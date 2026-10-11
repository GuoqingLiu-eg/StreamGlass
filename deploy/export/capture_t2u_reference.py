#!/usr/bin/env python3
"""Capture real official T2U calls while running streaming_evaluate.

This module only observes the official PyTorch execution. It does not replace
the model, alter READ/WRITE decisions, or change any tensor consumed by the
pipeline.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from seamless_communication.cli.streaming.evaluate import main as evaluate_main
from seamless_communication.models.unity.model import UnitYNART2UModel
from seamless_communication.models.unity.nar_decoder_frontend import (
    NARDecoderFrontend,
)
from seamless_communication.models.unity.unit_tokenizer import UnitTokenDecoder
from seamless_communication.streaming.agents.online_unit_decoder import (
    NARUnitDecoderAgentStates,
    NARUnitYUnitDecoderAgent,
)


CAPTURE_DIR = Path(os.environ["STREAMGLASS_T2U_CAPTURE_DIR"])
RECORDS: List[Dict[str, Any]] = []
ACTIVE_RECORD: Optional[Dict[str, Any]] = None
LAST_MODEL_RECORD: Optional[Dict[str, Any]] = None
FLUSHED = False


def _array(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def _capture_tensor(record: Dict[str, Any], name: str, tensor: Optional[Tensor]) -> None:
    if tensor is not None:
        record["arrays"][name] = _array(tensor)


def _padding_lengths(padding_mask: Any) -> Optional[Tensor]:
    if padding_mask is None:
        return None
    return getattr(padding_mask, "seq_lens", None)


ORIGINAL_MODEL_FORWARD = UnitYNART2UModel.forward
ORIGINAL_MODEL_ENCODE = UnitYNART2UModel.encode
ORIGINAL_MODEL_DECODE = UnitYNART2UModel.decode
ORIGINAL_TEXT_TO_CHARS = NARDecoderFrontend.text_to_char_seqs
ORIGINAL_CHAR_UPSAMPLING = NARDecoderFrontend.character_level_upsampling
ORIGINAL_FRONTEND_FORWARD = NARDecoderFrontend.forward
ORIGINAL_TOKEN_DECODE = UnitTokenDecoder.__call__
ORIGINAL_POLICY = NARUnitYUnitDecoderAgent.policy


def _model_forward(
    self: UnitYNART2UModel,
    text_decoder_output: Tensor,
    text_decoder_padding_mask: Any,
    text_seqs: Optional[Tensor],
    duration_factor: float = 1.0,
    film_cond_emb: Optional[Tensor] = None,
) -> Any:
    global ACTIVE_RECORD, LAST_MODEL_RECORD

    record: Dict[str, Any] = {
        "call_index": len(RECORDS),
        "arrays": {},
        "duration_factor": float(duration_factor),
    }
    _capture_tensor(record, "text_decoder_output", text_decoder_output)
    _capture_tensor(record, "text_seqs", text_seqs)
    _capture_tensor(
        record,
        "text_decoder_seq_lens",
        _padding_lengths(text_decoder_padding_mask),
    )
    _capture_tensor(record, "film_cond_emb", film_cond_emb)

    previous = ACTIVE_RECORD
    ACTIVE_RECORD = record
    try:
        output = ORIGINAL_MODEL_FORWARD(
            self,
            text_decoder_output,
            text_decoder_padding_mask,
            text_seqs,
            duration_factor,
            film_cond_emb,
        )
        model_output, decoder_padding_mask, durations = output
        _capture_tensor(record, "logits_argmax", model_output.logits.argmax(dim=-1))
        _capture_tensor(record, "durations", durations)
        _capture_tensor(
            record, "decoder_seq_lens", _padding_lengths(decoder_padding_mask)
        )
        return output
    finally:
        ACTIVE_RECORD = previous
        RECORDS.append(record)
        LAST_MODEL_RECORD = record


def _model_encode(
    self: UnitYNART2UModel,
    text_decoder_output: Tensor,
    text_decoder_padding_mask: Any,
) -> Any:
    output = ORIGINAL_MODEL_ENCODE(
        self, text_decoder_output, text_decoder_padding_mask
    )
    if ACTIVE_RECORD is not None:
        encoder_output, encoder_padding_mask = output
        _capture_tensor(ACTIVE_RECORD, "encoder_output", encoder_output)
        _capture_tensor(
            ACTIVE_RECORD,
            "encoder_seq_lens",
            _padding_lengths(encoder_padding_mask),
        )
    return output


def _text_to_char_seqs(
    self: NARDecoderFrontend, text_seqs: Tensor
) -> Any:
    output = ORIGINAL_TEXT_TO_CHARS(self, text_seqs)
    if ACTIVE_RECORD is not None:
        char_seqs, char_seq_lens, char_lens = output
        _capture_tensor(ACTIVE_RECORD, "char_seqs", char_seqs)
        _capture_tensor(ACTIVE_RECORD, "char_seq_lens", char_seq_lens)
        _capture_tensor(ACTIVE_RECORD, "char_lens", char_lens)
    return output


def _character_level_upsampling(
    self: NARDecoderFrontend,
    seqs: Tensor,
    padding_mask: Any,
    char_seqs: Tensor,
    char_lens: Tensor,
) -> Tensor:
    output = ORIGINAL_CHAR_UPSAMPLING(
        self, seqs, padding_mask, char_seqs, char_lens
    )
    if ACTIVE_RECORD is not None:
        _capture_tensor(ACTIVE_RECORD, "char_features", output)
    return output


def _frontend_forward(
    self: NARDecoderFrontend,
    encoder_output: Tensor,
    encoder_padding_mask: Any,
    text_seqs: Optional[Tensor],
    duration_factor: float = 1.0,
    film_cond_emb: Optional[Tensor] = None,
) -> Any:
    output = ORIGINAL_FRONTEND_FORWARD(
        self,
        encoder_output,
        encoder_padding_mask,
        text_seqs,
        duration_factor,
        film_cond_emb,
    )
    if ACTIVE_RECORD is not None:
        unit_features, unit_padding_mask, durations = output
        _capture_tensor(ACTIVE_RECORD, "unit_frontend_output", unit_features)
        _capture_tensor(ACTIVE_RECORD, "frontend_durations", durations)
        _capture_tensor(
            ACTIVE_RECORD,
            "unit_seq_lens",
            _padding_lengths(unit_padding_mask),
        )
    return output


def _model_decode(
    self: UnitYNART2UModel,
    encoder_output: Tensor,
    encoder_padding_mask: Any,
    text_seqs: Optional[Tensor],
    duration_factor: float = 1.0,
    film_cond_emb: Optional[Tensor] = None,
) -> Any:
    output = ORIGINAL_MODEL_DECODE(
        self,
        encoder_output,
        encoder_padding_mask,
        text_seqs,
        duration_factor,
        film_cond_emb,
    )
    if ACTIVE_RECORD is not None:
        decoder_output, decoder_padding_mask, durations = output
        _capture_tensor(ACTIVE_RECORD, "decoder_output", decoder_output)
        _capture_tensor(ACTIVE_RECORD, "decode_durations", durations)
        _capture_tensor(
            ACTIVE_RECORD,
            "decode_seq_lens",
            _padding_lengths(decoder_padding_mask),
        )
    return output


def _token_decode(self: UnitTokenDecoder, token_indices: Tensor) -> Tensor:
    global LAST_MODEL_RECORD

    output = ORIGINAL_TOKEN_DECODE(self, token_indices)
    if LAST_MODEL_RECORD is not None:
        _capture_tensor(LAST_MODEL_RECORD, "policy_token_indices", token_indices)
        _capture_tensor(LAST_MODEL_RECORD, "policy_unit_ids", output)
        LAST_MODEL_RECORD = None
    return output


def _policy(
    self: NARUnitYUnitDecoderAgent, states: NARUnitDecoderAgentStates
) -> Any:
    before_count = len(RECORDS)
    before_duration_index = int(states.duration_start_index)
    action = ORIGINAL_POLICY(self, states)

    if len(RECORDS) > before_count:
        record = RECORDS[-1]
        record["policy"] = {
            "source_finished": bool(states.source_finished),
            "source_token_count": len(states.source_token_list),
            "duration_start_index_before": before_duration_index,
            "duration_start_index_after": int(states.duration_start_index),
            "action": type(action).__name__,
            "action_finished": bool(getattr(action, "finished", False)),
        }
    return action


def _install_hooks() -> None:
    UnitYNART2UModel.forward = _model_forward  # type: ignore[method-assign]
    UnitYNART2UModel.encode = _model_encode  # type: ignore[method-assign]
    UnitYNART2UModel.decode = _model_decode  # type: ignore[method-assign]
    NARDecoderFrontend.text_to_char_seqs = _text_to_char_seqs  # type: ignore[method-assign]
    NARDecoderFrontend.character_level_upsampling = (  # type: ignore[method-assign]
        _character_level_upsampling
    )
    NARDecoderFrontend.forward = _frontend_forward  # type: ignore[method-assign]
    UnitTokenDecoder.__call__ = _token_decode  # type: ignore[method-assign]
    NARUnitYUnitDecoderAgent.policy = _policy  # type: ignore[method-assign]


def _flush() -> None:
    global FLUSHED
    if FLUSHED:
        return
    FLUSHED = True

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_calls = []
    for record in RECORDS:
        arrays = record.pop("arrays")
        filename = f"call_{record['call_index']:04d}.npz"
        np.savez_compressed(CAPTURE_DIR / filename, **arrays)
        manifest_calls.append(
            {
                **record,
                "file": filename,
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in sorted(arrays.items())
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "capture": "official-streaming-evaluate-t2u",
        "argv": sys.argv,
        "call_count": len(manifest_calls),
        "calls": manifest_calls,
    }
    (CAPTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Captured {len(manifest_calls)} official T2U call(s) in {CAPTURE_DIR}")


def main() -> None:
    _install_hooks()
    atexit.register(_flush)
    try:
        evaluate_main()
    finally:
        _flush()


if __name__ == "__main__":
    main()
