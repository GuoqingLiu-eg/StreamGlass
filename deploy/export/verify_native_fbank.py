#!/usr/bin/env python3
"""Compare native segmented fbank output with the official final capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-raw", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native = np.fromfile(args.native_raw, dtype=np.float32)
    if native.size % 80 != 0:
        raise RuntimeError("Native fbank byte count is not divisible by 80 bins")
    native = native.reshape(-1, 80)

    with np.load(args.reference) as capture:
        expected = capture["fbank"].astype(np.float32)[0]
        valid_frames = int(capture["fbank_seq_lens"][0])
    expected_valid = expected[:valid_frames]
    if native.shape != expected_valid.shape:
        raise RuntimeError(
            f"Fbank shape mismatch: native={native.shape} "
            f"official_valid={expected_valid.shape} padded={expected.shape}"
        )

    difference = np.abs(native - expected_valid)
    cosine = float(
        np.dot(native.reshape(-1), expected_valid.reshape(-1))
        / (
            np.linalg.norm(native.reshape(-1))
            * np.linalg.norm(expected_valid.reshape(-1))
        )
    )
    report = {
        "schema_version": 1,
        "component": "official_segmented_fbank_native_parity",
        "native_shape": list(native.shape),
        "official_padded_shape": list(expected.shape),
        "official_valid_frames": valid_frames,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.quantile(difference, 0.99)),
        "cosine": cosine,
        "finite": bool(np.isfinite(native).all()),
    }
    report["passed"] = bool(
        report["finite"]
        and report["max_abs"] <= 0.0041
        and report["cosine"] >= 0.99999
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
