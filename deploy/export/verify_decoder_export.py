#!/usr/bin/env python3
"""Verify all isolated external-data files in a decoder export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from onnx_artifacts import resolve_artifact, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def verify_graph(export_root: Path, graph: dict[str, Any]) -> dict[str, Any]:
    onnx_relative = Path(graph["file"])
    graph_root = resolve_artifact(export_root, str(onnx_relative)).parent
    artifacts = graph.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"No artifacts declared for {onnx_relative}")

    verified_bytes = 0
    for artifact in artifacts:
        path = resolve_artifact(graph_root, artifact["file"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing decoder ONNX artifact: {path}")
        size = path.stat().st_size
        if size != int(artifact["bytes"]):
            raise ValueError(
                f"Size mismatch for {path}: {size} != {artifact['bytes']}"
            )
        digest = sha256_file(path)
        if digest != artifact["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path}: {digest} != {artifact['sha256']}"
            )
        verified_bytes += size

    if len(artifacts) != int(graph["artifact_count"]):
        raise ValueError(f"Artifact count mismatch for {onnx_relative}")
    if verified_bytes != int(graph["artifact_bytes"]):
        raise ValueError(f"Artifact byte count mismatch for {onnx_relative}")
    return {
        "file": str(onnx_relative),
        "artifact_count": len(artifacts),
        "artifact_bytes": verified_bytes,
        "verified": True,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("mathematical_algorithm_changed") is not False:
        raise ValueError("Decoder export is not marked mathematically equivalent")
    if int(manifest.get("validated_call_count", 0)) < 20:
        raise ValueError("Decoder export did not pass all 20 official calls")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, dict) or set(graphs) != {"prefill", "step"}:
        raise ValueError("Decoder export must contain prefill and step graphs")

    export_root = args.manifest.parent.resolve()
    report = {
        name: verify_graph(export_root, graph)
        for name, graph in sorted(graphs.items())
    }
    print(json.dumps({"verified": True, "graphs": report}, indent=2))


if __name__ == "__main__":
    main()
