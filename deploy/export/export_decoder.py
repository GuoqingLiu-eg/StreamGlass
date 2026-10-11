#!/usr/bin/env python3
"""Export the official monotonic decoder with explicit self-attention KV I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from fairseq2.nn.transformer.attention import NaiveSDPA, TorchSDPA
from fairseq2.nn.transformer.attention_mask import CustomAttentionMask
from torch import Tensor, nn

from seamless_communication.models.monotonic_decoder import (
    MonotonicDecoderModel,
    load_monotonic_decoder_model,
)

from onnx_artifacts import index_onnx_artifacts, write_json_atomic


class DecoderCore(nn.Module):
    """Official decoder modules with tensor-only explicit cache plumbing."""

    def __init__(self, model: MonotonicDecoderModel) -> None:
        super().__init__()
        self.frontend = model.text_decoder_frontend
        self.decoder = model.text_decoder
        self.final_proj = model.final_proj

        if self.frontend.pos_encoder is None:
            raise ValueError("The official decoder frontend has no position encoder")
        if not hasattr(self.frontend.pos_encoder, "freqs"):
            raise TypeError("Expected the official sinusoidal position encoder")

        self.num_layers = len(self.decoder.layers)
        if self.num_layers != 24:
            raise ValueError(f"Expected 24 decoder layers, got {self.num_layers}")

    def embed(self, token_indices: Tensor, positions: Tensor) -> Tensor:
        seqs = self.frontend.embed(token_indices)
        if self.frontend.scale != 1.0:
            seqs = seqs * self.frontend.scale

        freqs = self.frontend.pos_encoder.freqs.index_select(0, positions.reshape(-1))
        seqs = (seqs.float() + freqs).type_as(seqs)

        if self.frontend.layer_norm is not None:
            seqs = self.frontend.layer_norm(seqs)
        if self.frontend.dropout is not None:
            seqs = self.frontend.dropout(seqs)
        return seqs

    @staticmethod
    def self_attention(
        layer: nn.Module,
        seqs: Tensor,
        cached_k: Optional[Tensor],
        cached_v: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        residual = seqs
        normalized = layer.self_attn_layer_norm(seqs)
        attention = layer.self_attn

        if attention.pos_encoder is not None:
            raise ValueError("Unexpected position encoder in decoder self-attention")

        q = attention.q_proj(normalized)
        q = q.unflatten(-1, (attention.num_heads, -1)).transpose(1, 2)
        k = attention.k_proj(normalized)
        k = k.unflatten(-1, (attention.num_key_value_heads, -1)).transpose(1, 2)
        v = attention.v_proj(normalized)
        v = v.unflatten(-1, (attention.num_key_value_heads, -1)).transpose(1, 2)

        if cached_k is not None:
            if cached_v is None:
                raise ValueError("K cache was provided without V cache")
            k = torch.cat((cached_k, k), dim=2)
            v = torch.cat((cached_v, v), dim=2)
        elif cached_v is not None:
            raise ValueError("V cache was provided without K cache")

        query_groups = attention.num_heads // attention.num_key_value_heads
        attention_k = k
        attention_v = v
        if query_groups > 1:
            attention_k = torch.repeat_interleave(
                attention_k, repeats=query_groups, dim=1
            )
            attention_v = torch.repeat_interleave(
                attention_v, repeats=query_groups, dim=1
            )

        materialized_mask = (
            None
            if attention_mask is None
            else CustomAttentionMask(attention_mask)
        )

        attended, _ = attention.sdpa(
            q,
            attention_k,
            None,
            attention_v,
            attn_mask=materialized_mask,
            needs_weights=False,
        )
        attended = attended.transpose(1, 2)
        if attention.head_scale_weight is not None:
            attended = torch.einsum(
                "nshv,h->nshv", attended, attention.head_scale_weight
            )
        attended = attended.flatten(2, 3)
        attended = attention.output_proj(attended)

        if layer.self_attn_dropout is not None:
            attended = layer.self_attn_dropout(attended)
        return attended + residual, k, v

    def decode(
        self,
        seqs: Tensor,
        encoder_output: Tensor,
        cached_k: Optional[Tensor],
        cached_v: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        p_choose_layers: List[Tensor] = []
        output_k: List[Tensor] = []
        output_v: List[Tensor] = []

        for layer_index, layer in enumerate(self.decoder.layers):
            layer_k = None if cached_k is None else cached_k[layer_index]
            layer_v = None if cached_v is None else cached_v[layer_index]
            seqs, key, value = self.self_attention(
                layer, seqs, layer_k, layer_v, attention_mask
            )
            seqs, p_choose = layer._forward_encoder_decoder_attn(
                seqs, None, encoder_output, None
            )
            seqs = layer._forward_ffn(seqs)
            output_k.append(key)
            output_v.append(value)
            p_choose_layers.append(p_choose)

        seqs = self.decoder.layer_norm(seqs)
        p_choose = torch.cat(p_choose_layers, dim=0).flatten(0, 1)
        return seqs, p_choose, torch.stack(output_k), torch.stack(output_v)

    def outputs(
        self,
        decoder_output: Tensor,
        p_choose: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits = self.final_proj(decoder_output[:, -1:])[:, 0]
        decision_values = p_choose[:, -1, -1]
        return (
            decoder_output,
            logits,
            decision_values,
            self_k_cache,
            self_v_cache,
        )


class DecoderPrefill(nn.Module):
    def __init__(self, core: DecoderCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        target_input: Tensor,
        encoder_output: Tensor,
        causal_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        positions = torch.arange(
            target_input.shape[1], device=target_input.device, dtype=torch.int64
        )
        seqs = self.core.embed(target_input, positions)
        outputs = self.core.decode(
            seqs, encoder_output, None, None, attention_mask=causal_mask
        )
        return self.core.outputs(*outputs)


class DecoderStep(nn.Module):
    def __init__(self, core: DecoderCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        target_input: Tensor,
        encoder_output: Tensor,
        position: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        seqs = self.core.embed(target_input, position)
        outputs = self.core.decode(
            seqs,
            encoder_output,
            self_k_cache,
            self_v_cache,
            attention_mask=None,
        )
        return self.core.outputs(*outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="seamless_streaming_monotonic_decoder"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def use_official_naive_sdpa_for_export(module: nn.Module) -> int:
    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, TorchSDPA):
            replacement = NaiveSDPA(attn_dropout_p=child.attn_dropout_p)
            replacement.train(child.training)
            setattr(module, name, replacement)
            replacements += 1
        else:
            replacements += use_official_naive_sdpa_for_export(child)
    return replacements


def max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def cosine_similarity(left: Tensor, right: Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().reshape(1, -1), right.float().reshape(1, -1)
        ).item()
    )


def captured_cache(
    capture: Any, suffix: str, device: torch.device, num_layers: int
) -> Tuple[Tensor, Tensor]:
    keys = [
        torch.from_numpy(capture[f"self_k_{suffix}_{index:02d}"]).to(device)
        for index in range(num_layers)
    ]
    values = [
        torch.from_numpy(capture[f"self_v_{suffix}_{index:02d}"]).to(device)
        for index in range(num_layers)
    ]
    return torch.stack(keys), torch.stack(values)


def causal_attention_mask(target_input: Tensor, dtype: torch.dtype) -> Tensor:
    target_length = target_input.shape[1]
    positions = torch.arange(
        target_length, device=target_input.device, dtype=torch.int64
    )
    allowed = positions[:, None] >= positions[None, :]
    zeros = torch.zeros((), device=target_input.device, dtype=dtype)
    negative_infinity = torch.full(
        (), float("-inf"), device=target_input.device, dtype=dtype
    )
    return torch.where(allowed, zeros, negative_infinity)


def validate_references(
    capture_dir: Path,
    prefill: DecoderPrefill,
    step: DecoderStep,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Tuple[Tensor, Tensor], Tuple[Tensor, ...]]:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    reports: List[Dict[str, Any]] = []
    prefill_example = None
    step_example = None

    with torch.no_grad():
        for call in manifest["calls"]:
            capture = np.load(capture_dir / call["file"])
            target_input = torch.from_numpy(capture["target_input"]).to(device)
            encoder_output = torch.from_numpy(capture["encoder_output"]).to(device)
            expected_decoder = torch.from_numpy(capture["decoder_output"]).to(device)
            expected_logits = torch.from_numpy(capture["logits"]).to(device)[:, -1]
            expected_p_choose = torch.from_numpy(capture["p_choose"]).to(device)[
                :, -1, -1
            ]
            expected_k, expected_v = captured_cache(
                capture, "out", device, prefill.core.num_layers
            )

            is_prefill = int(call["self_cache_layers_in"]) == 0
            if is_prefill:
                causal_mask = causal_attention_mask(
                    target_input, encoder_output.dtype
                )
                actual = prefill(target_input, encoder_output, causal_mask)
                if prefill_example is None or target_input.shape[1] > prefill_example[0].shape[1]:
                    prefill_example = (
                        target_input,
                        encoder_output,
                        causal_mask,
                    )
            else:
                input_k, input_v = captured_cache(
                    capture, "in", device, prefill.core.num_layers
                )
                position = torch.tensor(
                    [int(call["state_step_before_decode"])],
                    dtype=torch.int64,
                    device=device,
                )
                actual = step(
                    target_input, encoder_output, position, input_k, input_v
                )
                if step_example is None or input_k.shape[3] > step_example[3].shape[3]:
                    step_example = (
                        target_input,
                        encoder_output,
                        position,
                        input_k,
                        input_v,
                    )

            actual_decoder, actual_logits, actual_p_choose, actual_k, actual_v = actual
            selected_index = int(actual_logits[0].argmax().item())
            expected_index = int(call["selected_index"])
            decoder_cosine = cosine_similarity(actual_decoder, expected_decoder)
            cache_k_cosine = cosine_similarity(actual_k, expected_k)
            cache_v_cosine = cosine_similarity(actual_v, expected_v)
            probability_error = abs(
                float(actual_p_choose.min().item())
                - float(call["decision_probability"])
            )
            report = {
                "call_index": int(call["call_index"]),
                "kind": "prefill" if is_prefill else "step",
                "target_length": int(target_input.shape[1]),
                "encoder_length": int(encoder_output.shape[1]),
                "cache_length_out": int(actual_k.shape[3]),
                "selected_index": selected_index,
                "expected_index": expected_index,
                "decoder_max_abs": max_abs(actual_decoder, expected_decoder),
                "decoder_cosine": decoder_cosine,
                "logits_max_abs": max_abs(actual_logits, expected_logits),
                "p_choose_max_abs": max_abs(actual_p_choose, expected_p_choose),
                "decision_probability_error": probability_error,
                "cache_k_max_abs": max_abs(actual_k, expected_k),
                "cache_k_cosine": cache_k_cosine,
                "cache_v_max_abs": max_abs(actual_v, expected_v),
                "cache_v_cosine": cache_v_cosine,
            }
            if (
                selected_index != expected_index
                or decoder_cosine < 0.999
                or cache_k_cosine < 0.999
                or cache_v_cosine < 0.999
                or probability_error > 0.02
            ):
                raise RuntimeError(
                    "Explicit-cache wrapper diverged from the official decoder: "
                    + json.dumps(report, sort_keys=True)
                )
            reports.append(report)

    if prefill_example is None or step_example is None:
        raise RuntimeError("Capture must contain both prefill and incremental calls")
    return reports, prefill_example, step_example


def export_onnx(
    prefill: DecoderPrefill,
    step: DecoderStep,
    prefill_example: Tuple[Tensor, ...],
    step_example: Tuple[Tensor, ...],
    output_dir: Path,
    opset: int,
) -> Tuple[Path, Path]:
    output_names = (
        "decoder_output",
        "logits",
        "decision_values",
        "self_k_cache_out",
        "self_v_cache_out",
    )
    # The legacy PyTorch exporter names external tensors from graph values.
    # Prefill and step therefore need isolated directories; otherwise equal
    # names can truncate or overwrite the other graph's external data file.
    prefill_dir = output_dir / "prefill"
    step_dir = output_dir / "step"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = prefill_dir / "decoder_prefill.onnx"
    step_path = step_dir / "decoder_step.onnx"

    torch.onnx.export(
        prefill,
        prefill_example,
        str(prefill_path),
        input_names=("target_input", "encoder_output", "causal_mask"),
        output_names=output_names,
        dynamic_axes={
            "target_input": {1: "target_length"},
            "encoder_output": {1: "encoder_length"},
            "causal_mask": {0: "target_length", 1: "target_length"},
            "decoder_output": {1: "target_length"},
            "self_k_cache_out": {3: "target_length"},
            "self_v_cache_out": {3: "target_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    torch.onnx.export(
        step,
        step_example,
        str(step_path),
        input_names=(
            "target_input",
            "encoder_output",
            "position",
            "self_k_cache",
            "self_v_cache",
        ),
        output_names=output_names,
        dynamic_axes={
            "encoder_output": {1: "encoder_length"},
            "self_k_cache": {3: "cache_length"},
            "self_v_cache": {3: "cache_length"},
            "self_k_cache_out": {3: "cache_length_out"},
            "self_v_cache_out": {3: "cache_length_out"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    for path in (prefill_path, step_path):
        onnx.checker.check_model(str(path))
    return prefill_path, step_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model = load_monotonic_decoder_model(
        args.model_name, device=device, dtype=dtype
    ).eval()
    sdpa_replacement_count = use_official_naive_sdpa_for_export(model)
    if sdpa_replacement_count != 48:
        raise RuntimeError(
            f"Expected 48 official SDPA replacements, got {sdpa_replacement_count}"
        )

    core = DecoderCore(model).eval()
    prefill = DecoderPrefill(core).eval()
    step = DecoderStep(core).eval()
    reports, prefill_example, step_example = validate_references(
        args.capture_dir, prefill, step, device
    )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "component": "official_seamless_streaming_monotonic_decoder",
        "model_name": args.model_name,
        "dtype": args.dtype,
        "opset": args.opset,
        "algorithm": (
            "official monotonic decoder with tensor-only explicit self-attention "
            "KV cache plumbing"
        ),
        "mathematical_algorithm_changed": False,
        "policy_semantics": (
            "state resets for each official policy call; prefill is followed by "
            "zero or more cached single-token steps"
        ),
        "cache_layout": "[24, batch=1, heads=16, cache_length, head_dim=64]",
        "sdpa_export_mode": "official fairseq2 NaiveSDPA fallback",
        "sdpa_replacement_count": sdpa_replacement_count,
        "references": reports,
        "validated_call_count": len(reports),
        "graphs": {},
    }

    if not args.validate_only:
        prefill_path, step_path = export_onnx(
            prefill,
            step,
            prefill_example,
            step_example,
            args.output_dir,
            args.opset,
        )
        manifest["graphs"] = {
            "prefill": {
                "file": str(prefill_path.relative_to(args.output_dir)),
                **index_onnx_artifacts(prefill_path),
            },
            "step": {
                "file": str(step_path.relative_to(args.output_dir)),
                **index_onnx_artifacts(step_path),
            },
        }

    write_json_atomic(args.output_dir / "export_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
