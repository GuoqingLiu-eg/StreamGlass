#!/usr/bin/env python3
"""Compare a TensorRT vocoder engine against exported PyTorch references."""

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


class TensorRtRunner:
    def __init__(self, engine_path: Path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.cuda = CudaRuntime()

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        allocations: list[ctypes.c_void_p] = []
        stream = self.cuda.create_stream()
        try:
            for name, array in inputs.items():
                contiguous = np.ascontiguousarray(array)
                expected_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if contiguous.dtype != expected_dtype:
                    contiguous = contiguous.astype(expected_dtype)
                if not self.context.set_input_shape(name, contiguous.shape):
                    raise RuntimeError(f"TensorRT rejected shape {contiguous.shape} for {name}")
                device_pointer = self.cuda.malloc(contiguous.nbytes)
                allocations.append(device_pointer)
                self.cuda.check(
                    self.cuda.lib.cudaMemcpyAsync(
                        device_pointer,
                        ctypes.c_void_p(contiguous.ctypes.data),
                        contiguous.nbytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        stream,
                    ),
                    f"copy input {name}",
                )
                if not self.context.set_tensor_address(name, int(device_pointer.value)):
                    raise RuntimeError(f"Failed to bind input tensor {name}")

            missing = self.context.infer_shapes()
            if missing:
                raise RuntimeError(f"TensorRT shape inference is missing tensors: {missing}")

            output_names = [
                self.engine.get_tensor_name(index)
                for index in range(self.engine.num_io_tensors)
                if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
                == trt.TensorIOMode.OUTPUT
            ]
            if output_names != ["waveform"]:
                raise RuntimeError(f"Unexpected TensorRT outputs: {output_names}")

            output_name = output_names[0]
            output_shape = tuple(self.context.get_tensor_shape(output_name))
            if any(dimension < 0 for dimension in output_shape):
                raise RuntimeError(f"Unresolved output shape: {output_shape}")
            output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(output_name)))
            output = np.empty(output_shape, dtype=output_dtype)
            output_pointer = self.cuda.malloc(output.nbytes)
            allocations.append(output_pointer)
            if not self.context.set_tensor_address(output_name, int(output_pointer.value)):
                raise RuntimeError(f"Failed to bind output tensor {output_name}")

            if not self.context.execute_async_v3(int(stream.value)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            self.cuda.check(
                self.cuda.lib.cudaMemcpyAsync(
                    ctypes.c_void_p(output.ctypes.data),
                    output_pointer,
                    output.nbytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                    stream,
                ),
                "copy output waveform",
            )
            self.cuda.check(self.cuda.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            return output
        finally:
            for allocation in allocations:
                self.cuda.free(allocation)
            self.cuda.check(self.cuda.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Shape mismatch: reference={reference.shape}, TensorRT={candidate.shape}")
    reference_f32 = reference.astype(np.float32).reshape(-1)
    candidate_f32 = candidate.astype(np.float32).reshape(-1)
    denominator = float(np.linalg.norm(reference_f32) * np.linalg.norm(candidate_f32))
    cosine = float(np.dot(reference_f32, candidate_f32) / denominator) if denominator else 1.0
    delta = np.abs(reference_f32 - candidate_f32)
    return {
        "shape": list(reference.shape),
        "cosine": cosine,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "reference_rms": float(np.sqrt(np.mean(np.square(reference_f32)))),
        "tensorrt_rms": float(np.sqrt(np.mean(np.square(candidate_f32)))),
        "finite": bool(np.isfinite(candidate_f32).all()),
    }


def main() -> None:
    args = parse_args()
    references = sorted(args.reference_dir.glob("reference_len*.npz"))
    if not references:
        raise FileNotFoundError(f"No reference files under {args.reference_dir}")

    runner = TensorRtRunner(args.engine)
    cases: list[dict[str, Any]] = []
    failed = False
    for path in references:
        with np.load(path) as data:
            inputs = {
                "unit_ids": data["unit_ids"],
                "lang_ids": data["lang_ids"],
                "speaker_ids": data["speaker_ids"],
            }
            result = metrics(data["waveform"], runner.run(inputs))
        result["reference"] = path.name
        result["unit_length"] = int(inputs["unit_ids"].shape[1])
        if result["cosine"] < args.min_cosine or not result["finite"]:
            failed = True
        cases.append(result)

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
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
