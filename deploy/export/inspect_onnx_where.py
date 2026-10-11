#!/usr/bin/env python3
"""Print branch types and constant values for every ONNX Where node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    return parser.parse_args()


def constant_details(node: Any) -> dict[str, Any] | None:
    if node is None or node.op_type != "Constant":
        return None
    value_attribute = next(
        (attribute for attribute in node.attribute if attribute.name == "value"), None
    )
    if value_attribute is None:
        return None
    tensor = helper.get_attribute_value(value_attribute)
    value = numpy_helper.to_array(tensor)
    return {
        "data_type": int(tensor.data_type),
        "data_type_name": onnx.TensorProto.DataType.Name(tensor.data_type),
        "shape": list(value.shape),
        "value": value.tolist(),
    }


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.onnx), load_external_data=False)
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    nodes = []
    for node in model.graph.node:
        if node.op_type != "Where":
            continue
        nodes.append(
            {
                "name": node.name,
                "condition": node.input[0],
                "branches": [
                    {
                        "name": name,
                        "producer": producers[name].op_type
                        if name in producers
                        else None,
                        "constant": constant_details(producers.get(name)),
                    }
                    for name in node.input[1:]
                ],
            }
        )
    print(json.dumps({"where_count": len(nodes), "nodes": nodes}, indent=2))


if __name__ == "__main__":
    main()
