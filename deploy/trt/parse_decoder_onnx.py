#!/usr/bin/env python3
"""Parse decoder ONNX graphs with the installed TensorRT without building."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def tensor_description(tensor: trt.ITensor) -> Dict[str, Any]:
    return {
        "name": tensor.name,
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def parse_graph(path: Path) -> Dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(str(path.resolve()))
    errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    report = {
        "file": str(path),
        "parsed": bool(parsed),
        "errors": errors,
        "network_layers": int(network.num_layers),
        "inputs": [
            tensor_description(network.get_input(index))
            for index in range(network.num_inputs)
        ],
        "outputs": [
            tensor_description(network.get_output(index))
            for index in range(network.num_outputs)
        ],
    }
    if not parsed or errors:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    graphs = {
        name: parse_graph(root / graph["file"])
        for name, graph in sorted(manifest["graphs"].items())
    }
    report = {
        "schema_version": 1,
        "tensorrt_version": trt.__version__,
        "manifest": str(args.manifest),
        "passed": all(graph["parsed"] for graph in graphs.values()),
        "graphs": graphs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
