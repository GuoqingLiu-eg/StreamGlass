#!/usr/bin/env python3
"""Locate TensorRT layers that freeze decoder target-length dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--source-input", default="target_input")
    parser.add_argument("--fixed-length", type=int, default=6)
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def tensor_shape(tensor: trt.ITensor | None) -> list[int] | None:
    if tensor is None:
        return None
    return [int(dimension) for dimension in tensor.shape]


def tensor_record(tensor: trt.ITensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "name": tensor.name,
        "shape": tensor_shape(tensor),
        "dtype": str(tensor.dtype),
    }


def main() -> None:
    args = parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))

    source = None
    for index in range(network.num_inputs):
        candidate = network.get_input(index)
        if candidate.name == args.source_input:
            source = candidate
            break
    if source is None:
        raise KeyError(f"Network input not found: {args.source_input}")

    dependent = {source.name}
    transitions: list[dict[str, Any]] = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        inputs = [layer.get_input(i) for i in range(layer.num_inputs)]
        if not any(tensor is not None and tensor.name in dependent for tensor in inputs):
            continue

        outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
        dependent.update(
            tensor.name for tensor in outputs if tensor is not None
        )
        input_shapes = [tensor_shape(tensor) for tensor in inputs if tensor is not None]
        output_shapes = [tensor_shape(tensor) for tensor in outputs if tensor is not None]
        input_has_dynamic = any(
            shape is not None and -1 in shape for shape in input_shapes
        )
        output_has_fixed_length = any(
            shape is not None and args.fixed_length in shape for shape in output_shapes
        )
        output_has_dynamic = any(
            shape is not None and -1 in shape for shape in output_shapes
        )
        if input_has_dynamic and output_has_fixed_length and not output_has_dynamic:
            transitions.append(
                {
                    "index": index,
                    "name": layer.name,
                    "type": str(layer.type),
                    "inputs": [tensor_record(tensor) for tensor in inputs],
                    "outputs": [tensor_record(tensor) for tensor in outputs],
                }
            )
            if len(transitions) >= args.limit:
                break

    report = {
        "onnx": str(args.onnx),
        "source_input": args.source_input,
        "source_shape": tensor_shape(source),
        "network_layers": network.num_layers,
        "candidate_count": len(transitions),
        "candidates": transitions,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
