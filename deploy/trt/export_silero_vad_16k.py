#!/usr/bin/env python3
"""Export the official Silero 16 kHz core and an optional parity fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import wave
from pathlib import Path

import numpy as np
import onnx
import torch


EXPECTED_JIT_SHA256 = (
    "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
)
CHUNK_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)
SAMPLE_RATE = 16_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def export_onnx(jit_path: Path, onnx_path: Path) -> tuple[torch.jit.ScriptModule, float]:
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    core = model._model
    current_input = torch.zeros((1, CONTEXT_SAMPLES + CHUNK_SAMPLES), dtype=torch.float32)
    state = torch.zeros(STATE_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        probability, next_state = core(current_input, state)
    if tuple(probability.shape) != (1, 1) or tuple(next_state.shape) != STATE_SHAPE:
        raise RuntimeError("Official Silero 16 kHz submodule contract changed")

    temporary = onnx_path.with_name(onnx_path.name + f".tmp.{os.getpid()}")
    torch.onnx.export(
        core,
        (current_input, state),
        str(temporary),
        input_names=["input", "state"],
        output_names=["output", "stateN"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    exported = onnx.load(str(temporary))
    onnx.checker.check_model(exported, full_check=True)
    os.replace(temporary, onnx_path)
    return model, float(probability.item())


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Fixture WAV must be {SAMPLE_RATE} Hz")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Fixture WAV must be mono PCM16")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def write_fixture(
    model: torch.jit.ScriptModule, wav_path: Path, fixture_path: Path
) -> dict[str, object]:
    samples = read_pcm16_mono(wav_path)
    chunk_count = (samples.size + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    model.reset_states()
    probabilities: list[float] = []
    temporary = fixture_path.with_name(fixture_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        output.write(b"SGVAD001")
        output.write(
            struct.pack(
                "<IIII", SAMPLE_RATE, CHUNK_SAMPLES, int(chunk_count), 0
            )
        )
        with torch.inference_mode():
            for index in range(chunk_count):
                begin = index * CHUNK_SAMPLES
                chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                available = min(CHUNK_SAMPLES, samples.size - begin)
                chunk[:available] = samples[begin : begin + available]
                probability = float(
                    model(torch.from_numpy(chunk).unsqueeze(0), SAMPLE_RATE).item()
                )
                probabilities.append(probability)
                output.write(chunk.astype("<f4", copy=False).tobytes())
                output.write(struct.pack("<f", probability))
    os.replace(temporary, fixture_path)
    return {
        "audio_path": str(wav_path),
        "audio_sha256": sha256(wav_path),
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256(fixture_path),
        "chunk_count": int(chunk_count),
        "sample_count": int(samples.size),
        "probability_min": min(probabilities),
        "probability_max": max(probabilities),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jit", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fixture-wav", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument(
        "--expected-jit-sha256", default=EXPECTED_JIT_SHA256
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.fixture_wav) != bool(args.fixture_output):
        raise ValueError("--fixture-wav and --fixture-output must be used together")
    actual_jit_sha = sha256(args.jit)
    if actual_jit_sha != args.expected_jit_sha256:
        raise RuntimeError(
            f"Unexpected Silero JIT SHA256: {actual_jit_sha}; "
            f"expected {args.expected_jit_sha256}"
        )

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    model, zero_probability = export_onnx(args.jit, args.onnx)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": "snakers4/silero-vad",
        "source_tag": "v6.2.1",
        "jit_sha256": actual_jit_sha,
        "onnx_path": str(args.onnx),
        "onnx_sha256": sha256(args.onnx),
        "opset": 17,
        "sample_rate": SAMPLE_RATE,
        "chunk_samples": CHUNK_SAMPLES,
        "context_samples": CONTEXT_SAMPLES,
        "input_shape": [1, CONTEXT_SAMPLES + CHUNK_SAMPLES],
        "state_shape": list(STATE_SHAPE),
        "zero_input_probability": zero_probability,
        "torch_version": torch.__version__,
    }
    if args.fixture_wav:
        metadata["fixture"] = write_fixture(
            model, args.fixture_wav, args.fixture_output
        )
    atomic_json(args.metadata, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
