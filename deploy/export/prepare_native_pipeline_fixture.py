#!/usr/bin/env python3
"""Pack official S0 policy traces for native end-to-end replay validation."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = b"SGPIP001"
READ = 0
WRITE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-capture-dir", type=Path, required=True)
    parser.add_argument("--decoder-capture-dir", type=Path, required=True)
    parser.add_argument("--t2u-capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text())


def action_code(name: str) -> int:
    if name == "ReadAction":
        return READ
    if name == "WriteAction":
        return WRITE
    raise ValueError(f"Unsupported official action {name}")


def main() -> None:
    args = parse_args()
    encoder = load_manifest(args.encoder_capture_dir)
    decoder = load_manifest(args.decoder_capture_dir)
    t2u = load_manifest(args.t2u_capture_dir)

    decoder_groups: dict[int, list[dict[str, Any]]] = {}
    for call in decoder["calls"]:
        decoder_groups.setdefault(int(call["policy_index"]), []).append(call)
    ordered_decoder = [decoder_groups[index] for index in sorted(decoder_groups)]
    encoder_calls = encoder["calls"]
    t2u_calls = t2u["calls"]
    if len(encoder_calls) != len(ordered_decoder):
        raise ValueError("Encoder and decoder policy counts differ")

    records: list[dict[str, Any]] = []
    t2u_index = 0
    t2u_active = False
    for encoder_call, decoder_calls in zip(encoder_calls, ordered_decoder):
        policy = decoder_calls[0]["policy"]
        encoder_policy = encoder_call["policy"]
        if bool(policy["source_finished"]) != bool(
            encoder_policy["source_finished"]
        ):
            raise ValueError("Encoder and decoder finished flags differ")

        text_action = action_code(policy["action"])
        if text_action == WRITE:
            t2u_active = True

        t2u_record: dict[str, Any] | None = None
        unit_ids = np.empty(0, dtype=np.int32)
        if t2u_active:
            if t2u_index >= len(t2u_calls):
                raise ValueError("Missing downstream T2U policy call")
            t2u_record = t2u_calls[t2u_index]
            t2u_index += 1
            arrays = np.load(args.t2u_capture_dir / t2u_record["file"])
            if "policy_unit_ids" in arrays:
                unit_ids = np.asarray(
                    arrays["policy_unit_ids"], dtype=np.int32
                ).reshape(-1)

        selected = np.asarray(
            [int(call["selected_index"]) for call in decoder_calls],
            dtype=np.int64,
        )
        probabilities = np.asarray(
            [float(call["decision_probability"]) for call in decoder_calls],
            dtype=np.float32,
        )
        target_after = np.asarray(
            policy["target_indices_after"], dtype=np.int64
        )
        padded_frames = int(encoder_call["arrays"]["fbank"]["shape"][1])
        valid_frames = int(encoder_policy["source_feature_blocks"])
        encoder_frames = int(
            encoder_call["arrays"]["encoder_output"]["shape"][1]
        )

        records.append(
            {
                "fbank_valid": valid_frames,
                "fbank_padded": padded_frames,
                "encoder_frames": encoder_frames,
                "source_finished": int(bool(policy["source_finished"])),
                "text_action": text_action,
                "text_finished": int(bool(policy["action_finished"])),
                "target_after": target_after,
                "selected": selected,
                "probabilities": probabilities,
                "has_t2u": int(t2u_record is not None),
                "t2u_action": (
                    action_code(t2u_record["policy"]["action"])
                    if t2u_record is not None
                    else READ
                ),
                "t2u_finished": int(
                    bool(t2u_record["policy"]["action_finished"])
                    if t2u_record is not None
                    else False
                ),
                "duration_cursor": int(
                    t2u_record["policy"]["duration_start_index_after"]
                    if t2u_record is not None
                    else 0
                ),
                "unit_ids": unit_ids,
            }
        )

    if t2u_index != len(t2u_calls):
        raise ValueError("Unmapped T2U policy calls remain")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<8sII", MAGIC, len(records), 0))
        for record in records:
            handle.write(
                struct.pack(
                    "<13I",
                    record["fbank_valid"],
                    record["fbank_padded"],
                    record["encoder_frames"],
                    record["source_finished"],
                    record["text_action"],
                    record["text_finished"],
                    len(record["target_after"]),
                    len(record["selected"]),
                    record["has_t2u"],
                    record["t2u_action"],
                    record["t2u_finished"],
                    record["duration_cursor"],
                    len(record["unit_ids"]),
                )
            )
            handle.write(record["target_after"].astype("<i8").tobytes())
            handle.write(record["selected"].astype("<i8").tobytes())
            handle.write(record["probabilities"].astype("<f4").tobytes())
            handle.write(record["unit_ids"].astype("<i4").tobytes())

    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "policies": len(records),
                "t2u_calls": t2u_index,
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
