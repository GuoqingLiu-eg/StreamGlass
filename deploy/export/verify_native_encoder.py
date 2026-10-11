#!/usr/bin/env python3
"""Compare the native C++ encoder output with the official final capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-raw", type=Path, required=True)
    parser.add_argument("--native-length", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native = np.fromfile(args.native_raw, dtype=np.float16)
    if native.size % 1024 != 0:
        raise RuntimeError("Native encoder output is not divisible by 1024")
    native = native.reshape(1, -1, 1024)
    native_length_values = np.fromfile(args.native_length, dtype=np.int64)
    if native_length_values.size != 1:
        raise RuntimeError("Native encoder length file must contain one int64")

    with np.load(args.reference) as capture:
        expected = capture["encoder_output"]
        expected_lengths = capture["encoder_seq_lens"]
    if native.shape != expected.shape:
        raise RuntimeError(
            f"Encoder shape mismatch: native={native.shape} official={expected.shape}"
        )

    reference = expected.astype(np.float32).reshape(-1)
    candidate = native.astype(np.float32).reshape(-1)
    difference = np.abs(reference - candidate)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    cosine = float(np.dot(reference, candidate) / denominator)
    lengths_equal = bool(np.array_equal(native_length_values, expected_lengths))
    finite = bool(np.isfinite(candidate).all())
    report = {
        "schema_version": 1,
        "component": "native_cpp_encoder_official_parity",
        "shape": list(native.shape),
        "cosine": cosine,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "finite": finite,
        "native_seq_lens": native_length_values.tolist(),
        "official_seq_lens": expected_lengths.astype(np.int64).tolist(),
        "lengths_equal": lengths_equal,
        "minimum_cosine": args.minimum_cosine,
        "passed": bool(
            finite and lengths_equal and cosine >= args.minimum_cosine
        ),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
