#!/usr/bin/env python3
"""Export the official cumulative Seamless Streaming speech encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import torch
from fairseq2.assets import asset_store
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer.attention import NaiveSDPA, TorchSDPA
from torch import Tensor, nn

from seamless_communication.models.unity import load_unity_config, load_unity_model
from seamless_communication.models.unity.model import UnitYModel

from onnx_artifacts import index_onnx_artifacts, write_json_atomic


class SpeechEncoderWrapper(nn.Module):
    """Expose official encode_speech() with explicit sequence-length tensors."""

    def __init__(self, model: UnitYModel) -> None:
        super().__init__()
        self.speech_encoder_frontend = model.speech_encoder_frontend
        self.speech_encoder = model.speech_encoder

    def forward(self, fbank: Tensor, fbank_seq_lens: Tensor) -> Tuple[Tensor, Tensor]:
        padding_mask = PaddingMask(fbank_seq_lens, fbank.shape[1])
        seqs, padding_mask = self.speech_encoder_frontend(fbank, padding_mask)
        encoder_output, encoder_padding_mask = self.speech_encoder(seqs, padding_mask)
        if encoder_padding_mask is None:
            raise RuntimeError("Explicit input lengths must produce encoder lengths")
        return encoder_output, encoder_padding_mask.seq_lens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="seamless_streaming_unity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--sdpa-mode",
        choices=("official-default", "official-naive-export"),
        default="official-naive-export",
    )
    return parser.parse_args()


def max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def cosine_similarity(left: Tensor, right: Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().reshape(1, -1), right.float().reshape(1, -1)
        ).item()
    )


def load_model(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> UnitYModel:
    config = load_unity_config(model_name)
    config.use_text_decoder = False
    config.use_text_encoder = False
    config.t2u_config = None
    card = asset_store.retrieve_card(model_name)
    card.field("model_config").set(config)
    return load_unity_model(card, device=device, dtype=dtype).eval()


def use_official_naive_sdpa_for_export(module: nn.Module) -> int:
    """Use fairseq2's official tensor-op fallback so sequence axes stay dynamic."""
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


def validate_and_save_references(
    capture_dir: Path,
    output_dir: Path,
    encoder: SpeechEncoderWrapper,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Tuple[Tensor, Tensor]]:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    reference_dir = output_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reports: List[Dict[str, Any]] = []
    export_example = None

    with torch.no_grad():
        for call in manifest["calls"]:
            capture = np.load(capture_dir / call["file"])
            fbank = torch.from_numpy(capture["fbank"]).to(device)
            if "fbank_seq_lens" in capture:
                seq_lens = torch.from_numpy(capture["fbank_seq_lens"]).to(device)
            else:
                seq_lens = torch.full(
                    (fbank.shape[0],),
                    fbank.shape[1],
                    dtype=torch.int64,
                    device=device,
                )
            expected_output = torch.from_numpy(capture["encoder_output"]).to(device)
            if "encoder_seq_lens" in capture:
                expected_lens = torch.from_numpy(capture["encoder_seq_lens"]).to(device)
            else:
                expected_lens = torch.full(
                    (expected_output.shape[0],),
                    expected_output.shape[1],
                    dtype=torch.int64,
                    device=device,
                )

            actual_output, actual_lens = encoder(fbank, seq_lens)
            error = max_abs(actual_output, expected_output)
            cosine = cosine_similarity(actual_output, expected_output)
            lengths_equal = bool(torch.equal(actual_lens, expected_lens))
            if error > 0.08 or cosine < 0.9999 or not lengths_equal:
                raise RuntimeError(
                    "Encoder wrapper changed official output: "
                    f"call={call['call_index']} max_abs={error} cosine={cosine} "
                    f"lengths_equal={lengths_equal}"
                )

            filename = f"call_{int(call['call_index']):04d}.npz"
            np.savez_compressed(
                reference_dir / filename,
                fbank=fbank.cpu().numpy(),
                fbank_seq_lens=seq_lens.cpu().numpy(),
                encoder_output=actual_output.cpu().numpy(),
                encoder_seq_lens=actual_lens.cpu().numpy(),
            )
            reports.append(
                {
                    "call_index": int(call["call_index"]),
                    "reference": f"references/{filename}",
                    "fbank_length": int(fbank.shape[1]),
                    "valid_fbank_length": int(seq_lens[0].item()),
                    "encoder_length": int(actual_output.shape[1]),
                    "valid_encoder_length": int(actual_lens[0].item()),
                    "max_abs": error,
                    "cosine": cosine,
                    "lengths_equal": lengths_equal,
                }
            )
            if export_example is None or int(seq_lens[0].item()) < fbank.shape[1]:
                export_example = (fbank, seq_lens)

    if export_example is None:
        raise RuntimeError(f"No encoder calls found in {capture_dir}")
    return reports, export_example


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = load_model(args.model_name, device, dtype)
    sdpa_replacement_count = 0
    if args.sdpa_mode == "official-naive-export":
        sdpa_replacement_count = use_official_naive_sdpa_for_export(model)
        if sdpa_replacement_count == 0:
            raise RuntimeError("No TorchSDPA modules were found for export")
    encoder = SpeechEncoderWrapper(model).eval()

    reports, example = validate_and_save_references(
        args.capture_dir, args.output_dir, encoder, device
    )
    onnx_path = args.output_dir / "encoder.onnx"
    torch.onnx.export(
        encoder,
        example,
        str(onnx_path),
        input_names=("fbank", "fbank_seq_lens"),
        output_names=("encoder_output", "encoder_seq_lens"),
        dynamic_axes={
            "fbank": {1: "fbank_length"},
            "encoder_output": {1: "encoder_length"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(onnx_path))

    artifact_index = index_onnx_artifacts(onnx_path)
    export_manifest = {
        "schema_version": 1,
        "component": "official_seamless_streaming_cumulative_speech_encoder",
        "model_name": args.model_name,
        "dtype": args.dtype,
        "opset": args.opset,
        "algorithm": "official UnitYModel speech_encoder_frontend + speech_encoder",
        "sdpa_export_mode": args.sdpa_mode,
        "sdpa_replacement_count": sdpa_replacement_count,
        "sdpa_math": (
            "official fairseq2 NaiveSDPA fallback with equivalent scaled "
            "dot-product attention"
            if args.sdpa_mode == "official-naive-export"
            else "official runtime default"
        ),
        "dynamic_axes": {
            "fbank": {"axis": 1, "name": "fbank_length"},
            "encoder_output": {"axis": 1, "name": "encoder_length"},
        },
        "references": reports,
        **artifact_index,
    }
    write_json_atomic(args.output_dir / "export_manifest.json", export_manifest)
    print(json.dumps(export_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
