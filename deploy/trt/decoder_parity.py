#!/usr/bin/env python3
"""Compare TensorRT decoder prefill/step engines with official calls."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt


CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
EXPECTED_OUTPUTS = {
    "decoder_output",
    "logits",
    "decision_values",
    "self_k_cache_out",
    "self_v_cache_out",
}


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

    @staticmethod
    def check(code: int, operation: str) -> None:
        if code != CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), nbytes), "cudaMalloc")
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaFree(pointer), "cudaFree")

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream


class TensorRtRunner:
    def __init__(self, engine_path: Path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create context for {engine_path}")
        self.cuda = CudaRuntime()

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        allocations: list[ctypes.c_void_p] = []
        stream = self.cuda.create_stream()
        try:
            for name, array in inputs.items():
                contiguous = np.ascontiguousarray(array)
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if contiguous.dtype != dtype:
                    contiguous = contiguous.astype(dtype)
                if not self.context.set_input_shape(name, contiguous.shape):
                    raise RuntimeError(f"TensorRT rejected {name}={contiguous.shape}")
                pointer = self.cuda.malloc(contiguous.nbytes)
                allocations.append(pointer)
                self.cuda.check(
                    self.cuda.lib.cudaMemcpyAsync(
                        pointer,
                        ctypes.c_void_p(contiguous.ctypes.data),
                        contiguous.nbytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        stream,
                    ),
                    f"copy input {name}",
                )
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError(f"Failed to bind input {name}")

            missing = self.context.infer_shapes()
            if missing:
                raise RuntimeError(f"Unresolved TensorRT input shapes: {missing}")

            outputs: dict[str, np.ndarray] = {}
            output_pointers: dict[str, ctypes.c_void_p] = {}
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                    continue
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"Unresolved output {name}: {shape}")
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                output = np.empty(shape, dtype=dtype)
                pointer = self.cuda.malloc(output.nbytes)
                allocations.append(pointer)
                outputs[name] = output
                output_pointers[name] = pointer
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError(f"Failed to bind output {name}")

            if set(outputs) != EXPECTED_OUTPUTS:
                raise RuntimeError(f"Unexpected decoder outputs: {sorted(outputs)}")
            if not self.context.execute_async_v3(int(stream.value)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            for name, output in outputs.items():
                self.cuda.check(
                    self.cuda.lib.cudaMemcpyAsync(
                        ctypes.c_void_p(output.ctypes.data),
                        output_pointers[name],
                        output.nbytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        stream,
                    ),
                    f"copy output {name}",
                )
            self.cuda.check(
                self.cuda.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize"
            )
            return outputs
        finally:
            for allocation in allocations:
                self.cuda.free(allocation)
            self.cuda.check(self.cuda.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-engine", type=Path, required=True)
    parser.add_argument("--step-engine", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-decision-error", type=float, default=0.02)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    reference_f32 = reference.astype(np.float32).reshape(-1)
    candidate_f32 = candidate.astype(np.float32).reshape(-1)
    denominator = float(
        np.linalg.norm(reference_f32) * np.linalg.norm(candidate_f32)
    )
    cosine = (
        float(np.dot(reference_f32, candidate_f32) / denominator)
        if denominator
        else 1.0
    )
    delta = np.abs(reference_f32 - candidate_f32)
    return {
        "shape": list(reference.shape),
        "cosine": cosine,
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "finite": bool(np.isfinite(candidate_f32).all()),
    }


def stack_cache(capture: Any, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    keys = np.stack([capture[f"self_k_{suffix}_{index:02d}"] for index in range(24)])
    values = np.stack([capture[f"self_v_{suffix}_{index:02d}"] for index in range(24)])
    return keys, values


def causal_mask(target_length: int) -> np.ndarray:
    allowed = np.arange(target_length)[:, None] >= np.arange(target_length)[None, :]
    return np.where(allowed, 0.0, -np.inf).astype(np.float16)


def decision_layer_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> list[dict[str, Any]]:
    reference_layers = reference.astype(np.float32).reshape(24, 16)
    candidate_layers = candidate.astype(np.float32).reshape(24, 16)
    layers = []
    for index in range(24):
        delta = np.abs(reference_layers[index] - candidate_layers[index])
        layers.append(
            {
                "layer": index,
                "reference_min": float(reference_layers[index].min()),
                "candidate_min": float(candidate_layers[index].min()),
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
            }
        )
    return layers


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    prefill = TensorRtRunner(args.prefill_engine)
    step = TensorRtRunner(args.step_engine)
    cases: list[dict[str, Any]] = []
    failed = False

    for call in manifest["calls"]:
        path = args.reference_dir / call["file"]
        with np.load(path) as capture:
            expected_k, expected_v = stack_cache(capture, "out")
            is_prefill = int(call["self_cache_layers_in"]) == 0
            if is_prefill:
                target_length = int(capture["target_input"].shape[1])
                inputs = {
                    "target_input": capture["target_input"],
                    "encoder_output": capture["encoder_output"],
                    "causal_mask": causal_mask(target_length),
                }
                outputs = prefill.run(inputs)
                cache_length_in = 0
            else:
                input_k, input_v = stack_cache(capture, "in")
                inputs = {
                    "target_input": capture["target_input"],
                    "encoder_output": capture["encoder_output"],
                    "position": np.asarray(
                        [int(call["state_step_before_decode"])], dtype=np.int64
                    ),
                    "self_k_cache": input_k,
                    "self_v_cache": input_v,
                }
                outputs = step.run(inputs)
                cache_length_in = int(input_k.shape[3])

            decoder = float_metrics(
                capture["decoder_output"], outputs["decoder_output"]
            )
            logits = float_metrics(
                capture["logits"][:, -1], outputs["logits"]
            )
            decisions = float_metrics(
                capture["p_choose"][:, -1, -1], outputs["decision_values"]
            )
            decision_layers = decision_layer_metrics(
                capture["p_choose"][:, -1, -1], outputs["decision_values"]
            )
            cache_k = float_metrics(expected_k, outputs["self_k_cache_out"])
            cache_v = float_metrics(expected_v, outputs["self_v_cache_out"])
            selected_index = int(outputs["logits"][0].argmax())
            decision_probability = float(outputs["decision_values"].min())
            decision_error = abs(
                decision_probability - float(call["decision_probability"])
            )
            decision_probability_used = not bool(call["source_finished"])
            decision_probability_gate_passed = bool(
                not decision_probability_used
                or decision_error <= args.max_decision_error
            )
            passed = bool(
                selected_index == int(call["selected_index"])
                and decoder["finite"]
                and decoder["cosine"] >= args.min_cosine
                and cache_k["finite"]
                and cache_k["cosine"] >= args.min_cosine
                and cache_v["finite"]
                and cache_v["cosine"] >= args.min_cosine
                and decisions["finite"]
                and decision_probability_gate_passed
            )
            failed = failed or not passed
            cases.append(
                {
                    "call_index": int(call["call_index"]),
                    "reference": path.name,
                    "kind": "prefill" if is_prefill else "step",
                    "target_length": int(capture["target_input"].shape[1]),
                    "encoder_length": int(capture["encoder_output"].shape[1]),
                    "cache_length_in": cache_length_in,
                    "selected_index": selected_index,
                    "expected_index": int(call["selected_index"]),
                    "decision_probability": decision_probability,
                    "expected_decision_probability": float(
                        call["decision_probability"]
                    ),
                    "decision_probability_error": decision_error,
                    "source_finished": bool(call["source_finished"]),
                    "decision_probability_used_by_policy": decision_probability_used,
                    "decision_probability_gate_passed": decision_probability_gate_passed,
                    "decoder_output": decoder,
                    "logits": logits,
                    "decision_values": decisions,
                    "decision_layers": decision_layers,
                    "self_k_cache_out": cache_k,
                    "self_v_cache_out": cache_v,
                    "passed": passed,
                }
            )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prefill_engine": str(args.prefill_engine),
        "prefill_engine_sha256": sha256_file(args.prefill_engine),
        "step_engine": str(args.step_engine),
        "step_engine_sha256": sha256_file(args.step_engine),
        "minimum_cosine": args.min_cosine,
        "maximum_decision_probability_error": args.max_decision_error,
        "passed": not failed,
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
