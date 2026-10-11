#!/usr/bin/env python3
"""Verify every file declared by an ONNX export manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onnx_artifacts import verify_manifest_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_manifest_artifacts(args.manifest, args.artifact_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
