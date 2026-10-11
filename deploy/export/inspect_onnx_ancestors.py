#!/usr/bin/env python3
"""Inspect the bounded producer graph for an ONNX tensor or node output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--depth", type=int, default=8)
    return parser.parse_args()


def tensor_shape(value_info: Any) -> list[int | str | None] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def summarize_attribute(attribute: Any) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        summary: dict[str, Any] = {
            "dtype": onnx.TensorProto.DataType.Name(value.data_type),
            "shape": list(array.shape),
        }
        if array.size <= 16:
            summary["value"] = array.tolist()
        else:
            summary["range"] = [float(np.min(array)), float(np.max(array))]
        return summary
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.onnx), load_external_data=False)
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    shapes = {
        value.name: tensor_shape(value)
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
    }
    visited: set[str] = set()

    def walk(tensor: str, depth: int) -> dict[str, Any]:
        result: dict[str, Any] = {"tensor": tensor, "shape": shapes.get(tensor)}
        if tensor in visited:
            result["revisited"] = True
            return result
        visited.add(tensor)
        node = producers.get(tensor)
        if node is None:
            result["source"] = "graph_input_or_initializer"
            return result
        result.update(
            {
                "node": node.name,
                "op_type": node.op_type,
                "attributes": {
                    attribute.name: summarize_attribute(attribute)
                    for attribute in node.attribute
                },
            }
        )
        if depth > 0:
            result["inputs"] = [walk(name, depth - 1) for name in node.input]
        else:
            result["input_names"] = list(node.input)
        return result

    print(json.dumps(walk(args.tensor, args.depth), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
