#!/usr/bin/env python3
"""Compare the TensorRT encoder with all captured official encoder calls."""

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
            raise RuntimeError("Failed to create TensorRT execution context")
        self.cuda = CudaRuntime()

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        allocations: list[ctypes.c_void_p] = []
        stream = self.cuda.create_stream()
        try:
            for name, array in inputs.items():
                contiguous = np.ascontiguousarray(array)
                expected_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if contiguous.dtype != expected_dtype:
                    contiguous = contiguous.astype(expected_dtype)
                if not self.context.set_input_shape(name, contiguous.shape):
                    raise RuntimeError(f"TensorRT rejected {name} shape {contiguous.shape}")
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
                    raise RuntimeError(f"Failed to bind input tensor {name}")

            missing = self.context.infer_shapes()
            if missing:
                raise RuntimeError(f"TensorRT shape inference is missing: {missing}")

            outputs: dict[str, np.ndarray] = {}
            output_pointers: dict[str, ctypes.c_void_p] = {}
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                    continue
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"Unresolved output {name} shape: {shape}")
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                output = np.empty(shape, dtype=dtype)
                pointer = self.cuda.malloc(output.nbytes)
                allocations.append(pointer)
                output_pointers[name] = pointer
                outputs[name] = output
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError(f"Failed to bind output tensor {name}")

            if set(outputs) != {"encoder_output", "encoder_seq_lens"}:
                raise RuntimeError(f"Unexpected encoder outputs: {sorted(outputs)}")
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
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float32).reshape(-1)
    got = candidate.astype(np.float32).reshape(-1)
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(got))
    cosine = float(np.dot(ref, got) / denominator) if denominator else 1.0
    delta = np.abs(ref - got)
    return {
        "shape": list(reference.shape),
        "cosine": cosine,
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "finite": bool(np.isfinite(got).all()),
    }


def exact_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_equal = reference.shape == candidate.shape
    equal = bool(shape_equal and np.array_equal(reference, candidate))
    return {
        "shape_equal": shape_equal,
        "reference": reference.astype(np.int64).tolist(),
        "candidate": candidate.astype(np.int64).tolist(),
        "equal": equal,
    }


def main() -> None:
    args = parse_args()
    reference_files = sorted(args.reference_dir.glob("call_*.npz"))
    if not reference_files:
        raise FileNotFoundError(f"No encoder references in {args.reference_dir}")

    runner = TensorRtRunner(args.engine)
    cases = []
    failed = False
    for path in reference_files:
        with np.load(path) as data:
            outputs = runner.run(
                {
                    "fbank": data["fbank"],
                    "fbank_seq_lens": data["fbank_seq_lens"],
                }
            )
            output_metrics = float_metrics(
                data["encoder_output"], outputs["encoder_output"]
            )
            length_metrics = exact_metrics(
                data["encoder_seq_lens"], outputs["encoder_seq_lens"]
            )
            fbank_length = int(data["fbank"].shape[1])
            valid_fbank_length = int(data["fbank_seq_lens"][0])
        passed = bool(
            output_metrics["finite"]
            and output_metrics["cosine"] >= args.min_cosine
            and length_metrics["equal"]
        )
        failed = failed or not passed
        cases.append(
            {
                "reference": path.name,
                "fbank_length": fbank_length,
                "valid_fbank_length": valid_fbank_length,
                "encoder_output": output_metrics,
                "encoder_seq_lens": length_metrics,
                "passed": passed,
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "minimum_cosine": args.min_cosine,
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
