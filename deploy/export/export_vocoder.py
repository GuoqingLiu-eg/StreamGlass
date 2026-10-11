#!/usr/bin/env python3
"""Export the official Seamless Streaming CodeHiFiGAN vocoder to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import Tensor, nn

from seamless_communication.models.vocoder.loader import load_vocoder_model


class VocoderOnnxWrapper(nn.Module):
    """Expose the official code generator with tensor-only inputs."""

    def __init__(self, code_generator: nn.Module) -> None:
        super().__init__()
        self.code_generator = code_generator

    def forward(
        self, unit_ids: Tensor, lang_ids: Tensor, speaker_ids: Tensor
    ) -> Tensor:
        sample = {"code": unit_ids, "lang": lang_ids, "spkr": speaker_ids}
        return self.code_generator(sample, False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="vocoder_v2")
    parser.add_argument("--model-file", type=Path, default=Path("/models/vocoder_v2.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--speaker-id", type=int, default=-1)
    parser.add_argument("--trace-unit-length", type=int, default=64)
    parser.add_argument("--reference-unit-lengths", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_units(length: int, num_embeddings: int, device: torch.device) -> Tensor:
    values = (torch.arange(length, device=device, dtype=torch.int64) * 37 + 11) % num_embeddings
    return values.unsqueeze(0)


def resolve_conditioning(
    lang_spkr_idx_map: dict[str, Any], language: str, speaker_id: int
) -> tuple[int, int]:
    try:
        lang_id = int(lang_spkr_idx_map["multilingual"][language])
        default_speaker = int(lang_spkr_idx_map["multispkr"][language][0])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"No vocoder conditioning entry for language {language!r}") from exc

    return lang_id, default_speaker if speaker_id < 0 else speaker_id


def tensor_stats(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    reference_f32 = reference.detach().float()
    candidate_f32 = candidate.detach().float()
    delta = (reference_f32 - candidate_f32).abs()
    cosine = torch.nn.functional.cosine_similarity(
        reference_f32.flatten(), candidate_f32.flatten(), dim=0
    )
    return {
        "cosine": float(cosine.item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.trace_unit_length < 1 or any(length < 1 for length in args.reference_unit_lengths):
        raise ValueError("Unit lengths must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.reference_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = load_vocoder_model(args.model_name, device=device, dtype=dtype)
    model.eval()

    lang_id, speaker_id = resolve_conditioning(
        model.lang_spkr_idx_map, args.language, args.speaker_id
    )
    code_generator = model.code_generator
    num_embeddings = int(code_generator.dict.num_embeddings)
    lang_ids = torch.tensor([[lang_id]], device=device, dtype=torch.int64)
    speaker_ids = torch.tensor([[speaker_id]], device=device, dtype=torch.int64)

    official_outputs: dict[int, Tensor] = {}
    with torch.inference_mode():
        for length in args.reference_unit_lengths:
            unit_ids = make_units(length, num_embeddings, device)
            official_outputs[length] = model(
                unit_ids, args.language, args.speaker_id, dur_prediction=False
            ).detach()

    code_generator.remove_weight_norm()
    code_generator.eval()
    wrapper = VocoderOnnxWrapper(code_generator).eval()

    representation_checks: dict[str, dict[str, float]] = {}
    reference_files: list[dict[str, Any]] = []
    with torch.inference_mode():
        for length, official_output in official_outputs.items():
            unit_ids = make_units(length, num_embeddings, device)
            wrapper_output = wrapper(unit_ids, lang_ids, speaker_ids)
            stats = tensor_stats(official_output, wrapper_output)
            representation_checks[str(length)] = stats
            if stats["cosine"] < 0.999999 or stats["max_abs"] > 0.002:
                raise RuntimeError(
                    f"Weight-norm removal changed vocoder output at length {length}: {stats}"
                )

            reference_path = args.reference_dir / f"reference_len{length:04d}.npz"
            np.savez(
                reference_path,
                unit_ids=unit_ids.cpu().numpy(),
                lang_ids=lang_ids.cpu().numpy(),
                speaker_ids=speaker_ids.cpu().numpy(),
                waveform=official_output.cpu().numpy(),
            )
            reference_files.append(
                {
                    "path": reference_path.name,
                    "unit_length": length,
                    "waveform_shape": list(official_output.shape),
                    "sha256": sha256_file(reference_path),
                }
            )

    trace_units = make_units(args.trace_unit_length, num_embeddings, device)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (trace_units, lang_ids, speaker_ids),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["unit_ids", "lang_ids", "speaker_ids"],
            output_names=["waveform"],
            dynamic_axes={
                "unit_ids": {1: "unit_length"},
                "waveform": {2: "waveform_samples"},
            },
        )

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)

    manifest = {
        "schema_version": 1,
        "component": "official_seamless_streaming_vocoder",
        "model_name": args.model_name,
        "model_sha256": sha256_file(args.model_file),
        "onnx_path": args.output.name,
        "onnx_sha256": sha256_file(args.output),
        "opset": args.opset,
        "dtype": args.dtype,
        "dynamic_axes": {
            "unit_ids": {"axis": 1, "name": "unit_length"},
            "waveform": {"axis": 2, "name": "waveform_samples"},
        },
        "conditioning": {
            "language": args.language,
            "language_id": lang_id,
            "speaker_id": speaker_id,
        },
        "profile_contract": {
            "min_unit_length": 1,
            "opt_unit_length": args.trace_unit_length,
            "max_unit_length": 256,
            "waveform_samples_per_unit": 320,
        },
        "representation_checks": representation_checks,
        "references": reference_files,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
