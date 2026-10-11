#!/usr/bin/env python3
"""Refresh an export manifest from the ONNX external-data metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onnx_artifacts import index_onnx_artifacts, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    artifact_index = index_onnx_artifacts(args.onnx)
    manifest.update(artifact_index)
    write_json_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "artifact_count": artifact_index["artifact_count"],
                "external_data_file_count": artifact_index[
                    "external_data_file_count"
                ],
                "artifact_bytes": artifact_index["artifact_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
