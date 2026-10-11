#!/usr/bin/env python3
"""Capture official cumulative W2v-BERT encoder calls during evaluation."""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor

from seamless_communication.cli.streaming.evaluate import main as evaluate_main
from seamless_communication.models.unity.model import UnitYModel
from seamless_communication.streaming.agents.offline_w2v_bert_encoder import (
    OfflineWav2VecBertEncoderAgent,
)


CAPTURE_DIR = Path(os.environ["STREAMGLASS_ENCODER_CAPTURE_DIR"])
RECORDS: List[Dict[str, Any]] = []
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


ORIGINAL_ENCODE_SPEECH = UnitYModel.encode_speech
ORIGINAL_POLICY = OfflineWav2VecBertEncoderAgent.policy


def _encode_speech(
    self: UnitYModel, seqs: Tensor, padding_mask: Any
) -> Any:
    record: Dict[str, Any] = {"call_index": len(RECORDS), "arrays": {}}
    _capture_tensor(record, "fbank", seqs)
    _capture_tensor(record, "fbank_seq_lens", _padding_lengths(padding_mask))

    output = ORIGINAL_ENCODE_SPEECH(self, seqs, padding_mask)
    encoder_output, encoder_padding_mask = output
    _capture_tensor(record, "encoder_output", encoder_output)
    _capture_tensor(
        record, "encoder_seq_lens", _padding_lengths(encoder_padding_mask)
    )
    RECORDS.append(record)
    return output


def _policy(self: OfflineWav2VecBertEncoderAgent, states: Any) -> Any:
    before_count = len(RECORDS)
    source_feature_blocks = len(states.source)
    action = ORIGINAL_POLICY(self, states)
    if len(RECORDS) > before_count:
        record = RECORDS[-1]
        record["policy"] = {
            "source_finished": bool(states.source_finished),
            "source_feature_blocks": source_feature_blocks,
            "action": type(action).__name__,
            "action_finished": bool(getattr(action, "finished", False)),
        }
    return action


def _install_hooks() -> None:
    UnitYModel.encode_speech = _encode_speech  # type: ignore[method-assign]
    OfflineWav2VecBertEncoderAgent.policy = _policy  # type: ignore[method-assign]


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
        "capture": "official-streaming-evaluate-encoder",
        "argv": sys.argv,
        "call_count": len(calls),
        "calls": calls,
    }
    (CAPTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Captured {len(calls)} official encoder call(s) in {CAPTURE_DIR}")


def main() -> None:
    _install_hooks()
    atexit.register(_flush)
    try:
        evaluate_main()
    finally:
        _flush()


if __name__ == "__main__":
    main()
