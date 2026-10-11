#!/usr/bin/env python3
"""Export the official NAR T2U model as two TensorRT-friendly ONNX graphs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from fairseq2.assets import asset_store
from torch import Tensor, nn

from seamless_communication.models.unity import (
    load_unity_config,
    load_unity_model,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.unity.model import UnitYNART2UModel


class T2UFrontend(nn.Module):
    """Official T2U encoder, character expansion, and duration prediction."""

    def __init__(self, model: UnitYNART2UModel) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.frontend = model.decoder_frontend

        adaptor = self.frontend.variance_adaptor
        if adaptor.duration_predictor is None:
            raise ValueError("The official NAR T2U model has no duration predictor")
        if any(
            predictor is not None
            for predictor in (
                adaptor.pitch_predictor,
                adaptor.vuv_predictor,
                adaptor.energy_predictor,
            )
        ):
            raise ValueError("This split requires the official duration-only adaptor")

    @staticmethod
    def _hard_upsample_to_chars(seqs: Tensor, char_lens: Tensor, char_count: int) -> Tensor:
        # Equivalent to HardUpsampling.repeat_interleave(), expressed with
        # tensor-only operations so S and C remain dynamic in ONNX/TensorRT.
        ends = char_lens.cumsum(dim=1)
        positions = torch.arange(
            char_count, device=char_lens.device, dtype=char_lens.dtype
        )
        source_indices = (positions.view(1, -1, 1) >= ends.unsqueeze(1)).sum(dim=2)
        gather_indices = source_indices.unsqueeze(2).expand(-1, -1, seqs.shape[2])
        return torch.gather(seqs, 1, gather_indices)

    def forward(
        self,
        text_decoder_output: Tensor,
        char_seqs: Tensor,
        char_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.encoder is not None:
            encoder_output, _ = self.encoder(text_decoder_output, None)
        else:
            encoder_output = text_decoder_output

        frontend = self.frontend
        seqs = self._hard_upsample_to_chars(
            encoder_output, char_lens, char_seqs.shape[1]
        )
        pos_embeds = frontend.pos_emb_alpha_char * (
            frontend.char_pos_encoder(seqs, None) - seqs
        )
        char_embeds = frontend.embed_char(char_seqs)
        if frontend.scale != 1.0:
            char_embeds = char_embeds * frontend.scale
        char_features = seqs + pos_embeds + char_embeds

        predictor = frontend.variance_adaptor.duration_predictor
        assert predictor is not None
        log_durations = predictor(char_features, None, None)
        durations = torch.clamp(
            torch.round(torch.exp(log_durations) - 1).to(torch.int64), min=1
        )
        return char_features, durations


class T2UDecoder(nn.Module):
    """Official unit-position frontend, FFT decoder, projection, and decode."""

    def __init__(self, model: UnitYNART2UModel) -> None:
        super().__init__()
        official_frontend = model.decoder_frontend
        # Keep decoder precision independent from the frontend model.
        self.unit_pos_encoder = copy.deepcopy(official_frontend.unit_pos_encoder)
        self.pos_emb_alpha = nn.Parameter(
            official_frontend.pos_emb_alpha.detach().clone()
        )
        self.dropout = copy.deepcopy(official_frontend.dropout)
        self.decoder = copy.deepcopy(model.decoder)
        self.final_proj = copy.deepcopy(model.final_proj)

        vocab = model.target_vocab_info
        if vocab.eos_idx is None or vocab.pad_idx is None:
            raise ValueError("T2U vocabulary must define EOS and PAD")
        self.eos_idx = int(vocab.eos_idx)
        self.pad_idx = int(vocab.pad_idx)

    def forward(self, expanded_char_features: Tensor) -> Tensor:
        seqs_input = expanded_char_features.to(dtype=self.pos_emb_alpha.dtype)
        pos_embeds = self.pos_emb_alpha * (
            self.unit_pos_encoder(seqs_input, None) - seqs_input
        )
        seqs = seqs_input + pos_embeds
        if self.dropout is not None:
            seqs = self.dropout(seqs)
        decoder_output, _ = self.decoder(seqs, None, film_cond_emb=None)
        # Keep the large FFT stack in official FP16, but accumulate the output
        # head in FP32 so small logit margins do not flip discrete unit IDs in
        # TensorRT. Weights and the linear/argmax algorithm remain unchanged.
        bias = self.final_proj.bias
        logits = F.linear(
            decoder_output.float(),
            self.final_proj.weight.float(),
            bias.float() if bias is not None else None,
        )
        token_indices = logits.argmax(dim=-1)

        # Exact UnitTokenDecoder mapping for the official NAR UnitY tokenizer.
        units = torch.where(
            token_indices == self.eos_idx,
            torch.full_like(token_indices, self.pad_idx),
            token_indices,
        )
        units = torch.where(
            units == self.pad_idx,
            torch.full_like(units, self.pad_idx + 4),
            units,
        )
        return (units - 4).to(torch.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="seamless_streaming_unity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--decoder-precision", choices=("mixed", "fp32"), default="fp32"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_abs(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def cosine_similarity(left: Tensor, right: Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    return float(
        torch.nn.functional.cosine_similarity(
            left_flat.unsqueeze(0), right_flat.unsqueeze(0)
        ).item()
    )


def unique_reference_calls(capture_dir: Path) -> Iterable[Tuple[int, Path]]:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    seen: set[Tuple[int, int, int]] = set()
    for call in manifest["calls"]:
        arrays = call["arrays"]
        key = (
            arrays["text_decoder_output"]["shape"][1],
            arrays["char_seqs"]["shape"][1],
            arrays["decoder_output"]["shape"][1],
        )
        if key not in seen:
            seen.add(key)
            yield int(call["call_index"]), capture_dir / call["file"]


def load_model(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> Tuple[UnitYNART2UModel, object]:
    config = load_unity_config(model_name)
    config.use_text_decoder = False
    config.use_text_encoder = False
    unit_tokenizer = load_unity_unit_tokenizer(model_name)
    card = asset_store.retrieve_card(model_name)
    card.field("model_config").set(config)
    unity_model = load_unity_model(card, device=device, dtype=dtype).eval()
    t2u = unity_model.t2u_model
    if not isinstance(t2u, UnitYNART2UModel):
        raise TypeError(f"Expected UnitYNART2UModel, got {type(t2u).__name__}")
    return t2u, unit_tokenizer


def validate_and_save_references(
    capture_dir: Path,
    output_dir: Path,
    frontend: T2UFrontend,
    decoder: T2UDecoder,
    unit_tokenizer: object,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Tuple[Tensor, Tensor, Tensor], Tensor]:
    reference_dir = output_dir / "split_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reports: List[Dict[str, object]] = []
    frontend_example = None
    decoder_example = None
    official_unit_decoder = unit_tokenizer.create_decoder()  # type: ignore[attr-defined]

    # ONNX tracing reuses the selected examples after this validation block.
    # no_grad keeps them ordinary tensors; inference_mode tensors cannot be
    # saved by the PyTorch 2.2 tracer even though no training is performed.
    with torch.no_grad():
        for call_index, capture_file in unique_reference_calls(capture_dir):
            capture = np.load(capture_file)
            text = torch.from_numpy(capture["text_decoder_output"]).to(device)
            char_seqs = torch.from_numpy(capture["char_seqs"]).to(device)
            char_lens = torch.from_numpy(capture["char_lens"]).to(device)
            expected_chars = torch.from_numpy(capture["char_features"]).to(device)
            expected_durations = torch.from_numpy(capture["durations"]).to(device)
            expected_token_indices = torch.from_numpy(capture["logits_argmax"]).to(
                device
            )

            char_features, durations = frontend(text, char_seqs, char_lens)
            expanded = char_features[0].repeat_interleave(
                durations[0], dim=0
            ).unsqueeze(0)
            unit_ids = decoder(expanded)
            expected_unit_ids = official_unit_decoder(expected_token_indices).to(
                torch.int32
            )

            char_error = max_abs(char_features, expected_chars)
            char_cosine = cosine_similarity(char_features, expected_chars)
            duration_equal = bool(torch.equal(durations, expected_durations))
            unit_equal = bool(torch.equal(unit_ids, expected_unit_ids))
            if (
                char_error > 0.05
                or char_cosine < 0.9999
                or not duration_equal
                or not unit_equal
            ):
                raise RuntimeError(
                    "T2U split changed official outputs: "
                    f"call={call_index} char_error={char_error} "
                    f"char_cosine={char_cosine} "
                    f"duration_equal={duration_equal} unit_equal={unit_equal}"
                )

            reference_file = reference_dir / f"call_{call_index:04d}.npz"
            np.savez_compressed(
                reference_file,
                text_decoder_output=text.cpu().numpy(),
                char_seqs=char_seqs.cpu().numpy(),
                char_lens=char_lens.cpu().numpy(),
                char_features=char_features.cpu().numpy(),
                durations=durations.cpu().numpy(),
                expanded_char_features=expanded.cpu().numpy(),
                unit_ids=unit_ids.cpu().numpy(),
            )
            reports.append(
                {
                    "call_index": call_index,
                    "reference": str(reference_file.relative_to(output_dir)),
                    "text_length": int(text.shape[1]),
                    "char_length": int(char_seqs.shape[1]),
                    "unit_length": int(expanded.shape[1]),
                    "char_max_abs": char_error,
                    "char_cosine": char_cosine,
                    "durations_equal": duration_equal,
                    "unit_ids_equal": unit_equal,
                }
            )
            if frontend_example is None:
                frontend_example = (text, char_seqs, char_lens)
                decoder_example = expanded

    if frontend_example is None or decoder_example is None:
        raise RuntimeError(f"No T2U calls found in {capture_dir}")
    return reports, frontend_example, decoder_example


def export_onnx(
    frontend: T2UFrontend,
    decoder: T2UDecoder,
    frontend_example: Tuple[Tensor, Tensor, Tensor],
    decoder_example: Tensor,
    output_dir: Path,
) -> Tuple[Path, Path]:
    frontend_path = output_dir / "t2u_frontend.onnx"
    decoder_path = output_dir / "t2u_decoder.onnx"

    torch.onnx.export(
        frontend,
        frontend_example,
        frontend_path,
        input_names=("text_decoder_output", "char_seqs", "char_lens"),
        output_names=("char_features", "durations"),
        dynamic_axes={
            "text_decoder_output": {1: "text_length"},
            "char_seqs": {1: "char_length"},
            "char_lens": {1: "text_length"},
            "char_features": {1: "char_length"},
            "durations": {1: "char_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    torch.onnx.export(
        decoder,
        (decoder_example,),
        decoder_path,
        input_names=("expanded_char_features",),
        output_names=("unit_ids",),
        dynamic_axes={
            "expanded_char_features": {1: "unit_length"},
            "unit_ids": {1: "unit_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    for path in (frontend_path, decoder_path):
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)
    return frontend_path, decoder_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model, unit_tokenizer = load_model(args.model_name, device, dtype)
    frontend = T2UFrontend(model).eval()
    decoder = T2UDecoder(model).eval()
    if args.decoder_precision == "fp32":
        decoder = decoder.float()

    reports, frontend_example, decoder_example = validate_and_save_references(
        args.capture_dir,
        args.output_dir,
        frontend,
        decoder,
        unit_tokenizer,
        device,
    )
    frontend_path, decoder_path = export_onnx(
        frontend, decoder, frontend_example, decoder_example, args.output_dir
    )

    manifest = {
        "schema_version": 1,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "algorithm": "official UnitYNART2UModel split at hard unit upsampling",
        "host_stage": "exact HardUpsampling(char_features, durations)",
        "decoder_precision": args.decoder_precision,
        "graphs": {
            "frontend": {
                "file": frontend_path.name,
                "sha256": sha256(frontend_path),
                "inputs": ["text_decoder_output", "char_seqs", "char_lens"],
                "outputs": ["char_features", "durations"],
            },
            "decoder": {
                "file": decoder_path.name,
                "sha256": sha256(decoder_path),
                "inputs": ["expanded_char_features"],
                "outputs": ["unit_ids"],
            },
        },
        "references": reports,
    }
    manifest_path = args.output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
