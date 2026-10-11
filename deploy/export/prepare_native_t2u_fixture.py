#!/usr/bin/env python3
"""Pack official T2U calls and split references for native replay."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"SGT2U001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--split-reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_array(handle: object, value: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    handle.write(array.tobytes(order="C"))


def find_split_reference(
    references: list[dict[str, np.ndarray]], capture: np.lib.npyio.NpzFile
) -> dict[str, np.ndarray]:
    for reference in references:
        if (
            reference["text_decoder_output"].shape
            == capture["text_decoder_output"].shape
            and np.array_equal(
                reference["text_decoder_output"],
                capture["text_decoder_output"],
            )
        ):
            return reference
    raise ValueError("No validated T2U split reference matches capture")


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.capture_dir / "manifest.json").read_text())
    references = [
        dict(np.load(path))
        for path in sorted(args.split_reference_dir.glob("call_*.npz"))
    ]
    if not references:
        raise ValueError("No T2U split references were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<8sII", MAGIC, len(manifest["calls"]), 0))
        for call in manifest["calls"]:
            capture = np.load(args.capture_dir / str(call["file"]))
            reference = find_split_reference(references, capture)
            text_features = np.asarray(
                capture["text_decoder_output"], dtype=np.float16
            )
            tokens = np.asarray(capture["text_seqs"], dtype=np.int64)
            char_seqs = np.asarray(capture["char_seqs"], dtype=np.int64)
            char_lens = np.asarray(capture["char_lens"], dtype=np.int64)
            durations = np.asarray(reference["durations"], dtype=np.int64)
            full_units = np.asarray(reference["unit_ids"], dtype=np.int32)
            policy_units = np.asarray(
                capture["policy_unit_ids"]
                if "policy_unit_ids" in capture.files
                else np.empty((1, 0), dtype=np.int32),
                dtype=np.int32,
            )

            if text_features.ndim != 3 or text_features.shape[0::2] != (1, 1024):
                raise ValueError(f"Unexpected T2U text shape {text_features.shape}")
            text_length = int(text_features.shape[1])
            character_count = int(char_seqs.shape[1])
            total_unit_count = int(full_units.shape[1])
            if tokens.shape != (1, text_length) or char_lens.shape != (1, text_length):
                raise ValueError("T2U token/character length shape mismatch")
            if durations.shape != (1, character_count):
                raise ValueError("T2U duration shape mismatch")
            if not np.array_equal(reference["char_seqs"], char_seqs):
                raise ValueError("Validated char sequence differs from capture")
            if not np.array_equal(reference["char_lens"], char_lens):
                raise ValueError("Validated char lengths differ from capture")

            policy = call["policy"]
            action_name = str(policy["action"])
            if action_name == "ReadAction":
                expected_action = 0
            elif policy_units.size > 0:
                expected_action = 1
            else:
                expected_action = 2
            duration_cursor = int(policy["duration_start_index_before"])
            adjusted_cursor = duration_cursor
            if bool(policy["source_finished"]) and duration_cursor > 0:
                remaining = int(durations[0, duration_cursor:].sum())
                if remaining > 0:
                    adjusted_cursor -= 1
            expected_unit_offset = int(durations[0, :adjusted_cursor].sum())
            if policy_units.size > 0 and not np.array_equal(
                policy_units, full_units[:, expected_unit_offset:]
            ):
                raise ValueError("Official T2U policy suffix differs from full units")

            handle.write(
                struct.pack(
                    "<IIIIIIIIIIII",
                    int(call["call_index"]),
                    int(bool(policy["source_finished"])),
                    text_length,
                    character_count,
                    duration_cursor,
                    expected_action,
                    total_unit_count,
                    expected_unit_offset,
                    int(policy_units.shape[1]),
                    int(policy["duration_start_index_after"]),
                    0,
                    0,
                )
            )
            write_array(handle, tokens, "<i8")
            write_array(handle, text_features, "<f2")
            write_array(handle, char_seqs, "<i8")
            write_array(handle, char_lens, "<i8")
            write_array(handle, durations, "<i8")
            write_array(handle, full_units, "<i4")
            write_array(handle, policy_units, "<i4")

    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "calls": len(manifest["calls"]),
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
