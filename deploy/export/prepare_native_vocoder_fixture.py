#!/usr/bin/env python3
"""Pack official vocoder references into a native replay fixture."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"SGVOC001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_array(handle: object, value: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    handle.write(array.tobytes(order="C"))


def main() -> None:
    args = parse_args()
    references = sorted(args.reference_dir.glob("reference_len*.npz"))
    if not references:
        raise FileNotFoundError("No vocoder references were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<8sII", MAGIC, len(references), 0))
        for path in references:
            with np.load(path) as data:
                units = np.asarray(data["unit_ids"], dtype=np.int64)
                language = np.asarray(data["lang_ids"], dtype=np.int64)
                speaker = np.asarray(data["speaker_ids"], dtype=np.int64)
                waveform = np.asarray(data["waveform"], dtype=np.float16)
            if units.ndim != 2 or units.shape[0] != 1:
                raise ValueError(f"Unexpected unit shape {units.shape}")
            if language.shape != (1, 1) or speaker.shape != (1, 1):
                raise ValueError("Unexpected vocoder conditioning shape")
            if waveform.shape != (1, 1, units.shape[1] * 320):
                raise ValueError(f"Unexpected waveform shape {waveform.shape}")
            handle.write(
                struct.pack(
                    "<IIqq",
                    int(units.shape[1]),
                    int(waveform.shape[2]),
                    int(language[0, 0]),
                    int(speaker[0, 0]),
                )
            )
            write_array(handle, units, "<i8")
            write_array(handle, waveform, "<f2")

    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "cases": len(references),
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
