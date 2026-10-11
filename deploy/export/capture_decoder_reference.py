#!/usr/bin/env python3
"""Capture official monotonic text decoder calls during streaming evaluation.

The hooks in this module are observational. They call the unmodified official
PyTorch methods and only copy inputs, outputs, decisions, and incremental
self-attention state to the reference artifact directory.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.transformer.multihead_attention import FullAttentionState
from torch import Tensor

from seamless_communication.cli.streaming.evaluate import main as evaluate_main
from seamless_communication.models.monotonic_decoder import MonotonicDecoderModel
from seamless_communication.streaming.agents.online_text_decoder import (
    DecoderAgentStates,
    MMATextDecoderAgent,
)


CAPTURE_DIR = Path(os.environ["STREAMGLASS_DECODER_CAPTURE_DIR"])
RECORDS: List[Dict[str, Any]] = []
ACTIVE_RECORD: Optional[Dict[str, Any]] = None
CURRENT_POLICY_INDEX: Optional[int] = None
NEXT_POLICY_INDEX = 0
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


def _capture_self_attention_caches(
    record: Dict[str, Any],
    model: MonotonicDecoderModel,
    state_bag: Optional[IncrementalStateBag],
    suffix: str,
) -> int:
    if state_bag is None:
        return 0

    captured = 0
    for layer_index, layer in enumerate(model.text_decoder.layers):
        state = state_bag.get_state(layer.self_attn, FullAttentionState)
        if state is None:
            continue
        key, value = state.get()
        _capture_tensor(record, f"self_k_{suffix}_{layer_index:02d}", key)
        _capture_tensor(record, f"self_v_{suffix}_{layer_index:02d}", value)
        captured += 1
    return captured


ORIGINAL_DECODE = MonotonicDecoderModel.decode
ORIGINAL_PROJECT = MonotonicDecoderModel.project
ORIGINAL_RUN_DECODER = MMATextDecoderAgent.run_decoder
ORIGINAL_POLICY = MMATextDecoderAgent.policy


def _decode(
    self: MonotonicDecoderModel,
    seqs: Tensor,
    padding_mask: Any,
    encoder_output: Tensor,
    encoder_padding_mask: Any,
    *,
    state_bag: Optional[IncrementalStateBag] = None,
) -> Any:
    record = ACTIVE_RECORD
    if record is not None:
        _capture_tensor(record, "target_input", seqs)
        _capture_tensor(record, "target_seq_lens", _padding_lengths(padding_mask))
        _capture_tensor(record, "encoder_output", encoder_output)
        _capture_tensor(
            record,
            "encoder_seq_lens",
            _padding_lengths(encoder_padding_mask),
        )
        record["state_step_before_decode"] = (
            int(state_bag.step_nr) if state_bag is not None else None
        )
        record["self_cache_layers_in"] = _capture_self_attention_caches(
            record, self, state_bag, "in"
        )

    output = ORIGINAL_DECODE(
        self,
        seqs,
        padding_mask,
        encoder_output,
        encoder_padding_mask,
        state_bag=state_bag,
    )

    if record is not None:
        decoder_output, decoder_padding_mask, p_choose = output
        _capture_tensor(record, "decoder_output", decoder_output)
        _capture_tensor(
            record,
            "decoder_seq_lens",
            _padding_lengths(decoder_padding_mask),
        )
        _capture_tensor(record, "p_choose", p_choose)
        record["state_step_after_decode"] = (
            int(state_bag.step_nr) if state_bag is not None else None
        )
        record["self_cache_layers_out"] = _capture_self_attention_caches(
            record, self, state_bag, "out"
        )

    return output


def _project(self: MonotonicDecoderModel, decoder_output: Tensor) -> Tensor:
    logits = ORIGINAL_PROJECT(self, decoder_output)
    if ACTIVE_RECORD is not None:
        _capture_tensor(ACTIVE_RECORD, "logits", logits)
        _capture_tensor(ACTIVE_RECORD, "logits_argmax", logits.argmax(dim=-1))
    return logits


def _run_decoder(
    self: MMATextDecoderAgent,
    states: DecoderAgentStates,
    pred_indices: List[int],
) -> Any:
    global ACTIVE_RECORD

    record: Dict[str, Any] = {
        "call_index": len(RECORDS),
        "policy_index": CURRENT_POLICY_INDEX,
        "decode_index_in_policy": sum(
            1 for item in RECORDS if item.get("policy_index") == CURRENT_POLICY_INDEX
        ),
        "source_finished": bool(states.source_finished),
        "states_target_indices_before": list(states.target_indices),
        "pred_indices_before": list(pred_indices),
        "prefix_indices": list(self.prefix_indices),
        "state_step_before_call": int(self.state_bag.step_nr),
        "arrays": {},
    }

    previous = ACTIVE_RECORD
    ACTIVE_RECORD = record
    try:
        index, probability, decoder_output = ORIGINAL_RUN_DECODER(
            self, states, pred_indices
        )
        record["selected_index"] = int(index)
        record["decision_probability"] = float(probability)
        record["state_step_after_call"] = int(self.state_bag.step_nr)
        _capture_tensor(record, "returned_decoder_output", decoder_output)
        return index, probability, decoder_output
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        ACTIVE_RECORD = previous
        RECORDS.append(record)


def _policy(self: MMATextDecoderAgent, states: DecoderAgentStates) -> Any:
    global CURRENT_POLICY_INDEX, NEXT_POLICY_INDEX

    policy_index = NEXT_POLICY_INDEX
    NEXT_POLICY_INDEX += 1
    previous_policy_index = CURRENT_POLICY_INDEX
    CURRENT_POLICY_INDEX = policy_index
    first_record = len(RECORDS)
    target_before = list(states.target_indices)
    source_len_before = int(states.source.size(1)) if len(states.source) > 0 else 0

    try:
        action = ORIGINAL_POLICY(self, states)
        policy = {
            "policy_index": policy_index,
            "source_finished": bool(states.source_finished),
            "source_len": source_len_before,
            "target_indices_before": target_before,
            "target_indices_after": list(states.target_indices),
            "action": type(action).__name__,
            "action_finished": bool(getattr(action, "finished", False)),
        }
        for record in RECORDS[first_record:]:
            record["policy"] = policy
        return action
    finally:
        CURRENT_POLICY_INDEX = previous_policy_index


def _install_hooks() -> None:
    MonotonicDecoderModel.decode = _decode  # type: ignore[method-assign]
    MonotonicDecoderModel.project = _project  # type: ignore[method-assign]
    MMATextDecoderAgent.run_decoder = _run_decoder  # type: ignore[method-assign]
    MMATextDecoderAgent.policy = _policy  # type: ignore[method-assign]


def _flush() -> None:
    global FLUSHED
    if FLUSHED:
        return
    FLUSHED = True

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    calls = []
    for record in RECORDS:
        arrays = record.pop("arrays")
        filename = f"call_{record['call_index']:04d}.npz"
        np.savez_compressed(CAPTURE_DIR / filename, **arrays)
        calls.append(
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
        "capture": "official-streaming-evaluate-monotonic-decoder",
        "argv": sys.argv,
        "call_count": len(calls),
        "policy_count": len({call.get("policy_index") for call in calls}),
        "calls": calls,
    }
    (CAPTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Captured {len(calls)} official decoder call(s) in {CAPTURE_DIR}")


def main() -> None:
    _install_hooks()
    atexit.register(_flush)
    try:
        evaluate_main()
    finally:
        _flush()


if __name__ == "__main__":
    main()
