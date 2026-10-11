#!/usr/bin/env python3
"""Pack official decoder captures into a compact native replay fixture."""

from __future__ import annotations

import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np


MAGIC = b"SGDEC001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_array(handle: object, value: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    handle.write(array.tobytes(order="C"))


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.capture_dir / "manifest.json").read_text())
    policies: OrderedDict[int, list[dict[str, object]]] = OrderedDict()
    for call in manifest["calls"]:
        policies.setdefault(int(call["policy_index"]), []).append(call)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<8sII", MAGIC, len(policies), len(manifest["calls"])))
        for policy_index, calls in policies.items():
            captures = [
                np.load(args.capture_dir / str(call["file"])) for call in calls
            ]
            encoder = np.asarray(captures[0]["encoder_output"], dtype=np.float16)
            if encoder.ndim != 3 or encoder.shape[0] != 1 or encoder.shape[2] != 1024:
                raise ValueError(f"Unexpected encoder shape {encoder.shape}")
            for capture in captures[1:]:
                if not np.array_equal(encoder, capture["encoder_output"]):
                    raise ValueError("Encoder output changed inside one policy call")

            handle.write(
                struct.pack(
                    "<IIII", policy_index, len(calls), int(encoder.shape[1]), 0
                )
            )
            write_array(handle, encoder, "<f2")

            for local_index, (call, capture) in enumerate(zip(calls, captures)):
                target = np.asarray(capture["target_input"], dtype=np.int64)
                decoder = np.asarray(capture["decoder_output"], dtype=np.float16)
                key = np.asarray(capture["self_k_out_00"], dtype=np.float16)
                kind = 0 if int(call["self_cache_layers_in"]) == 0 else 1
                if (local_index == 0) != (kind == 0):
                    raise ValueError("Every policy must start with exactly one prefill")
                if target.ndim != 2 or target.shape[0] != 1:
                    raise ValueError(f"Unexpected target shape {target.shape}")
                if decoder.shape != (1, target.shape[1], 1024):
                    raise ValueError(f"Unexpected decoder shape {decoder.shape}")
                if key.ndim != 4 or key.shape[0:2] != (1, 16) or key.shape[3] != 64:
                    raise ValueError(f"Unexpected cache shape {key.shape}")

                position = int(call["state_step_before_decode"] or 0)
                handle.write(
                    struct.pack(
                        "<IIIIIIIIf",
                        int(call["call_index"]),
                        kind,
                        int(target.shape[1]),
                        position,
                        int(call["selected_index"]),
                        int(decoder.shape[1]),
                        int(key.shape[2]),
                        int(bool(call["source_finished"])),
                        float(call["decision_probability"]),
                    )
                )
                write_array(handle, target, "<i8")
                write_array(handle, decoder, "<f2")

    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "policies": len(policies),
                "calls": len(manifest["calls"]),
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
