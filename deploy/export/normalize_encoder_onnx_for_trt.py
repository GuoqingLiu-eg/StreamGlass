#!/usr/bin/env python3
"""Restore dynamic encoder shapes and apply TensorRT compatibility casts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_artifacts import (
    index_onnx_artifacts,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-padding-casts", type=int, default=24)
    parser.add_argument("--expected-attention-casts", type=int, default=24)
    parser.add_argument("--expected-shaw-slice-bypasses", type=int, default=24)
    parser.add_argument("--expected-shaw-dynamic-restorations", type=int, default=24)
    parser.add_argument("--expected-traced-attention-length", type=int, default=198)
    return parser.parse_args()


def constant_tensor(node: Any) -> Any | None:
    if node is None or node.op_type != "Constant":
        return None
    value_attribute = next(
        (attribute for attribute in node.attribute if attribute.name == "value"), None
    )
    return helper.get_attribute_value(value_attribute) if value_attribute else None


def classify_cast(
    where_node: Any,
    branch_index: int,
    producers: dict[str, Any],
) -> str | None:
    branch_name = where_node.input[branch_index]
    tensor = constant_tensor(producers.get(branch_name))
    if tensor is None:
        return None
    value = numpy_helper.to_array(tensor)
    if value.size != 1:
        return None
    other_index = 2 if branch_index == 1 else 1
    other_producer = producers.get(where_node.input[other_index])

    if (
        where_node.name.endswith("/conv/Where")
        and tensor.data_type == TensorProto.INT64
        and int(value.item()) == 0
        and other_producer is not None
        and other_producer.op_type == "LayerNormalization"
    ):
        return "padding_zero_int64_to_float16"
    if (
        "/self_attn/sdpa/" in where_node.name
        and tensor.data_type == TensorProto.DOUBLE
        and np.isneginf(float(value.item()))
        and other_producer is not None
        and other_producer.op_type in ("Add", "Mul")
    ):
        return "attention_negative_infinity_double_to_float"
    return None


def constant_array(name: str, producers: dict[str, Any]) -> np.ndarray:
    tensor = constant_tensor(producers.get(name))
    if tensor is None:
        raise RuntimeError(f"Expected a Constant producer for {name}")
    return numpy_helper.to_array(tensor)


def einsum_equation(node: Any) -> str:
    attribute = next(
        (attribute for attribute in node.attribute if attribute.name == "equation"),
        None,
    )
    return attribute.s.decode() if attribute is not None else ""


def integer_scalar(name: str, producers: dict[str, Any]) -> int:
    value = constant_array(name, producers).astype(np.int64).reshape(-1)
    if value.size != 1:
        raise RuntimeError(f"Expected one integer at {name}, got {value.shape}")
    return int(value.item())


def gather_axis(node: Any) -> int:
    attribute = next(
        (attribute for attribute in node.attribute if attribute.name == "axis"), None
    )
    return int(attribute.i) if attribute is not None else 0


def int64_constant(name: str, output: str, value: int | list[int]) -> Any:
    array = np.asarray(value, dtype=np.int64)
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output],
        name=name,
        value=numpy_helper.from_array(array),
    )


def make_dynamic_relative_indices(
    namespace: str,
    query_tensor: str,
) -> tuple[list[Any], str]:
    shape = f"{namespace}/Shape_output_0"
    sequence_axis = f"{namespace}/SequenceAxis_output_0"
    sequence_length = f"{namespace}/SequenceLength_output_0"
    range_start = f"{namespace}/RangeStart_output_0"
    range_delta = f"{namespace}/RangeDelta_output_0"
    positions = f"{namespace}/Range_output_0"
    unsqueeze_axes = f"{namespace}/UnsqueezeAxes_output_0"
    row_positions = f"{namespace}/Unsqueeze_output_0"
    column_positions = f"{namespace}/Transpose_output_0"
    relative_indices = f"{namespace}/Sub_output_0"
    nodes = [
        helper.make_node(
            "Shape",
            inputs=[query_tensor],
            outputs=[shape],
            name=f"{namespace}/Shape",
        ),
        int64_constant(f"{namespace}/SequenceAxis", sequence_axis, 2),
        helper.make_node(
            "Gather",
            inputs=[shape, sequence_axis],
            outputs=[sequence_length],
            name=f"{namespace}/SequenceLength",
            axis=0,
        ),
        int64_constant(f"{namespace}/RangeStart", range_start, 0),
        int64_constant(f"{namespace}/RangeDelta", range_delta, 1),
        helper.make_node(
            "Range",
            inputs=[range_start, sequence_length, range_delta],
            outputs=[positions],
            name=f"{namespace}/Range",
        ),
        int64_constant(f"{namespace}/UnsqueezeAxes", unsqueeze_axes, [0]),
        helper.make_node(
            "Unsqueeze",
            inputs=[positions, unsqueeze_axes],
            outputs=[row_positions],
            name=f"{namespace}/Unsqueeze",
        ),
        helper.make_node(
            "Transpose",
            inputs=[row_positions],
            outputs=[column_positions],
            name=f"{namespace}/Transpose",
            perm=[1, 0],
        ),
        helper.make_node(
            "Sub",
            inputs=[row_positions, column_positions],
            outputs=[relative_indices],
            name=f"{namespace}/Sub",
        ),
    ]
    return nodes, relative_indices


def restore_dynamic_full_attention(
    node: Any,
    producers: dict[str, Any],
    traced_attention_length: int,
) -> tuple[dict[str, Any], str, list[Any]] | None:
    if (
        node.op_type != "Einsum"
        or "/speech_encoder/inner/layers." not in node.name
        or "/self_attn/sdpa/Einsum" not in node.name
    ):
        return None
    if einsum_equation(node) != "nhsk,stk->nhst":
        raise RuntimeError(f"Unexpected Shaw attention equation at {node.name}")
    slice_node = producers.get(node.input[1])
    if slice_node is None or slice_node.op_type != "Slice" or len(slice_node.input) != 5:
        raise RuntimeError(f"Expected a five-input Shaw Slice before {node.name}")
    starts, ends, axes, steps = (
        constant_array(name, producers).astype(np.int64).reshape(-1).tolist()
        for name in slice_node.input[1:]
    )
    expected = {
        "starts": [-traced_attention_length],
        "ends": [np.iinfo(np.int64).max],
        "axes": [0],
        "steps": [1],
    }
    actual = {"starts": starts, "ends": ends, "axes": axes, "steps": steps}
    if actual != expected:
        raise RuntimeError(
            f"Frozen Shaw Slice changed at {node.name}: {actual} != {expected}"
        )
    gather_node = producers.get(slice_node.input[0])
    if (
        gather_node is None
        or gather_node.op_type != "Gather"
        or len(gather_node.input) != 2
        or gather_axis(gather_node) != 0
    ):
        raise RuntimeError(f"Expected axis-0 relative embedding Gather at {node.name}")
    add_node = producers.get(gather_node.input[1])
    if add_node is None or add_node.op_type != "Add" or len(add_node.input) != 2:
        raise RuntimeError(f"Expected relative-index Add before {node.name}")
    clip_node = producers.get(add_node.input[0])
    if clip_node is None or clip_node.op_type != "Clip" or len(clip_node.input) != 3:
        raise RuntimeError(f"Expected three-input relative-index Clip at {node.name}")

    frozen_indices = constant_array(clip_node.input[0], producers).astype(np.int64)
    positions = np.arange(traced_attention_length, dtype=np.int64)
    expected_indices = positions[np.newaxis, :] - positions[:, np.newaxis]
    if not np.array_equal(frozen_indices, expected_indices):
        raise RuntimeError(
            f"Frozen relative-index matrix changed at {node.name}: "
            f"shape={frozen_indices.shape}"
        )
    clip_min = integer_scalar(clip_node.input[1], producers)
    clip_max = integer_scalar(clip_node.input[2], producers)
    offset = integer_scalar(add_node.input[1], producers)
    if (clip_min, clip_max, offset) != (-64, 8, 64):
        raise RuntimeError(
            f"Unexpected Shaw clipping at {node.name}: "
            f"min={clip_min} max={clip_max} offset={offset}"
        )

    namespace = f"{node.name.rsplit('/Einsum', 1)[0]}/streamglass_dynamic_rel"
    dynamic_nodes, dynamic_indices = make_dynamic_relative_indices(
        namespace, node.input[0]
    )
    frozen_input = clip_node.input[0]
    clip_node.input[0] = dynamic_indices
    sliced_input = node.input[1]
    node.input[1] = slice_node.input[0]
    report = {
        "einsum": node.name,
        "slice": slice_node.name,
        "slice_output": sliced_input,
        "replacement": slice_node.input[0],
        "validated_parameters": actual,
        "relative_index_constant": frozen_input,
        "relative_index_shape": list(frozen_indices.shape),
        "dynamic_relative_indices": dynamic_indices,
        "clip": [clip_min, clip_max],
        "offset": offset,
    }
    return report, clip_node.name, dynamic_nodes


def prepare_dynamic_shaw_attention(
    model: Any,
    producers: dict[str, Any],
    traced_attention_length: int,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    reports = []
    insertions: dict[str, list[Any]] = {}
    for node in model.graph.node:
        restoration = restore_dynamic_full_attention(
            node, producers, traced_attention_length
        )
        if restoration is None:
            continue
        report, insertion_target, dynamic_nodes = restoration
        if insertion_target in insertions:
            raise RuntimeError(f"Duplicate Shaw insertion target: {insertion_target}")
        reports.append(report)
        insertions[insertion_target] = dynamic_nodes
    return reports, insertions


def normalize(model: Any, traced_attention_length: int) -> dict[str, Any]:
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    cast_outputs: dict[str, str] = {}
    transformed_nodes: dict[str, list[str]] = {
        "padding_zero_int64_to_float16": [],
        "attention_negative_infinity_double_to_float": [],
    }
    shaw_restorations, shaw_insertions = prepare_dynamic_shaw_attention(
        model, producers, traced_attention_length
    )
    rewritten_nodes = []

    for node in model.graph.node:
        rewritten_nodes.extend(shaw_insertions.get(node.name, []))
        if node.op_type == "Where":
            for branch_index in (1, 2):
                category = classify_cast(node, branch_index, producers)
                if category is None:
                    continue
                source_name = node.input[branch_index]
                cast_output = cast_outputs.get(source_name)
                if cast_output is None:
                    cast_output = f"{source_name}__streamglass_float"
                    cast_outputs[source_name] = cast_output
                    cast_type = (
                        TensorProto.FLOAT16
                        if category == "padding_zero_int64_to_float16"
                        else TensorProto.FLOAT
                    )
                    rewritten_nodes.append(
                        helper.make_node(
                            "Cast",
                            inputs=[source_name],
                            outputs=[cast_output],
                            name=f"StreamGlassTypeCast_{len(cast_outputs):02d}",
                            to=cast_type,
                        )
                    )
                node.input[branch_index] = cast_output
                transformed_nodes[category].append(node.name)
        rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    return {
        "casts_inserted": len(cast_outputs),
        "cast_sources": sorted(cast_outputs),
        "transformed_nodes": transformed_nodes,
        "padding_cast_count": len(
            transformed_nodes["padding_zero_int64_to_float16"]
        ),
        "attention_cast_count": len(
            transformed_nodes["attention_negative_infinity_double_to_float"]
        ),
        "shaw_slice_bypass_count": len(shaw_restorations),
        "shaw_dynamic_restoration_count": len(shaw_restorations),
        "shaw_restorations": shaw_restorations,
        "full_self_attention_assumption": (
            "The exported cumulative encoder has q_len == key_len. The official "
            "relative-index formula is restored from the dynamic query length, and "
            "rel_keys[-q_len:] therefore selects the full rel_keys tensor."
        ),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("The TensorRT-normalized graph must not overwrite its source")
    model = onnx.load(str(source), load_external_data=False)
    transformation = normalize(model, args.expected_traced_attention_length)
    if transformation["padding_cast_count"] != args.expected_padding_casts:
        raise RuntimeError(
            "Unexpected Conformer padding cast count: "
            f"{transformation['padding_cast_count']} != {args.expected_padding_casts}"
        )
    if transformation["attention_cast_count"] != args.expected_attention_casts:
        raise RuntimeError(
            "Unexpected attention mask cast count: "
            f"{transformation['attention_cast_count']} != "
            f"{args.expected_attention_casts}"
        )
    if transformation["casts_inserted"] != 2:
        raise RuntimeError(
            f"Expected two shared scalar casts, got {transformation['casts_inserted']}"
        )
    if (
        transformation["shaw_slice_bypass_count"]
        != args.expected_shaw_slice_bypasses
    ):
        raise RuntimeError(
            "Unexpected frozen Shaw Slice count: "
            f"{transformation['shaw_slice_bypass_count']} != "
            f"{args.expected_shaw_slice_bypasses}"
        )
    if (
        transformation["shaw_dynamic_restoration_count"]
        != args.expected_shaw_dynamic_restorations
    ):
        raise RuntimeError(
            "Unexpected dynamic Shaw restoration count: "
            f"{transformation['shaw_dynamic_restoration_count']} != "
            f"{args.expected_shaw_dynamic_restorations}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.stem}.building{output.suffix}")
    onnx.save_model(model, str(temporary_output))
    onnx.checker.check_model(str(temporary_output))
    temporary_output.replace(output)
    artifact_index = index_onnx_artifacts(output)
    manifest = {
        "schema_version": 1,
        "component": "official_encoder_tensorrt_compatibility_normalization",
        "source_onnx": source.name,
        "source_onnx_sha256": sha256_file(source),
        "output_onnx": output.name,
        "mathematical_algorithm_changed": False,
        "deployment_scope": "full cumulative speech encoder self-attention",
        "normalization": transformation,
        **artifact_index,
    }
    write_json_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(args.manifest),
                "normalization": transformation,
                "artifact_count": artifact_index["artifact_count"],
                "artifact_bytes": artifact_index["artifact_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
