#!/usr/bin/env python3
"""Emit the loaded official T2U module contract without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from fairseq2.assets import asset_store

from seamless_communication.models.unity import (
    load_unity_config,
    load_unity_model,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.unity.model import UnitYNART2UModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="seamless_streaming_unity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    config = load_unity_config(args.model_name)
    config.use_text_decoder = False
    config.use_text_encoder = False
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    card = asset_store.retrieve_card(args.model_name)
    card.field("model_config").set(config)
    unity_model = load_unity_model(card, device=device, dtype=dtype).eval()

    t2u = unity_model.t2u_model
    if not isinstance(t2u, UnitYNART2UModel):
        raise TypeError(f"Expected UnitYNART2UModel, got {type(t2u).__name__}")
    frontend = t2u.decoder_frontend
    adaptor = frontend.variance_adaptor

    report = {
        "schema_version": 1,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "t2u_type": type(t2u).__name__,
        "model_dim": int(t2u.model_dim),
        "encoder": {
            "type": type(t2u.encoder).__name__ if t2u.encoder is not None else None,
            "layers": len(t2u.encoder.layers) if t2u.encoder is not None else 0,
        },
        "frontend": {
            "type": type(frontend).__name__,
            "unit_embedding_shape": list(frontend.embed.weight.shape),
            "char_embedding_shape": list(frontend.embed_char.weight.shape),
            "char_length_regulator": type(frontend.char_length_regulator).__name__,
            "unit_length_regulator": type(adaptor.length_regulator).__name__,
            "duration_predictor": type(adaptor.duration_predictor).__name__
            if adaptor.duration_predictor is not None
            else None,
            "pitch_predictor": type(adaptor.pitch_predictor).__name__
            if adaptor.pitch_predictor is not None
            else None,
            "vuv_predictor": type(adaptor.vuv_predictor).__name__
            if adaptor.vuv_predictor is not None
            else None,
            "energy_predictor": type(adaptor.energy_predictor).__name__
            if adaptor.energy_predictor is not None
            else None,
        },
        "decoder": {
            "type": type(t2u.decoder).__name__,
            "layers": len(t2u.decoder.layers),
        },
        "projection_shape": list(t2u.final_proj.weight.shape),
        "unit_tokenizer": {
            "vocabulary_size": int(unit_tokenizer.vocab_info.size),
            "is_nar_decoder": bool(unit_tokenizer.is_nar_decoder),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
