#!/usr/bin/env python3
"""Measure the physical full-duplex echo path without writing PCM to disk."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


SAMPLE_RATE = 16_000
CHANNELS = 1
FORMAT = "f32"


@dataclass(frozen=True)
class SignalStats:
    samples: int
    rms: float
    peak: float
    window_rms_p50: float
    window_rms_p95: float


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    values = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(values * values)))


def signal_stats(samples: np.ndarray, window_samples: int = 1_600) -> SignalStats:
    if samples.size == 0:
        return SignalStats(0, 0.0, 0.0, 0.0, 0.0)
    windows = [
        rms(samples[offset : offset + window_samples])
        for offset in range(0, samples.size, window_samples)
        if samples[offset : offset + window_samples].size == window_samples
    ]
    if not windows:
        windows = [rms(samples)]
    return SignalStats(
        samples=int(samples.size),
        rms=rms(samples),
        peak=float(np.max(np.abs(samples))),
        window_rms_p50=float(np.percentile(windows, 50)),
        window_rms_p95=float(np.percentile(windows, 95)),
    )


def read_official_sample(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != CHANNELS
            or wav.getsampwidth() != 2
            or wav.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(
                "official sample must be 16 kHz mono signed 16-bit PCM"
            )
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def trim_silence(samples: np.ndarray, threshold: float = 0.006) -> np.ndarray:
    window = SAMPLE_RATE // 100
    count = samples.size // window
    if count == 0:
        return samples
    framed = samples[: count * window].reshape(count, window)
    energy = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1))
    active = np.flatnonzero(energy >= threshold)
    if active.size == 0:
        raise ValueError("official sample contains no active speech")
    first = max(0, int(active[0]) * window)
    last = min(samples.size, (int(active[-1]) + 1) * window)
    return samples[first:last]


def repeat_with_crossfade(
    samples: np.ndarray,
    duration_seconds: float,
    gain: float,
    crossfade_ms: float = 10.0,
) -> np.ndarray:
    target = int(round(duration_seconds * SAMPLE_RATE))
    if target <= 0:
        raise ValueError("duration must be positive")
    fade = min(int(round(crossfade_ms * SAMPLE_RATE / 1000.0)), samples.size // 4)
    if fade <= 0:
        tiled = np.resize(samples, target)
    else:
        parts = [samples.copy()]
        total = samples.size
        fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        fade_in = 1.0 - fade_out
        while total < target:
            prior = parts[-1]
            blend = prior[-fade:] * fade_out + samples[:fade] * fade_in
            parts[-1] = np.concatenate((prior[:-fade], blend))
            parts.append(samples[fade:].copy())
            total += samples.size - fade
        tiled = np.concatenate(parts)[:target]
    return np.clip(tiled * gain, -0.98, 0.98).astype(np.float32, copy=False)


def pipewire_nodes() -> set[str]:
    result = subprocess.run(
        ["pw-dump"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    objects = json.loads(result.stdout)

    def node_name(obj: object) -> str | None:
        if not isinstance(obj, dict):
            return None
        info = obj.get("info")
        if not isinstance(info, dict):
            return None
        props = info.get("props")
        if not isinstance(props, dict):
            return None
        value = props.get("node.name")
        return str(value) if value else None

    return {
        name
        for obj in objects
        if (name := node_name(obj)) is not None
    }


def wait_for_nodes(expected: set[str], timeout_seconds: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if expected.issubset(pipewire_nodes()):
            return
        time.sleep(0.1)
    missing = sorted(expected - pipewire_nodes())
    raise RuntimeError(f"PipeWire AEC nodes did not appear: {missing}")


PLAY_DELAY_PATTERN = re.compile(
    r"\bbuffer\.play_delay\s*=\s*(\d+)\s*/\s*(\d+)"
)


def configured_play_delay_ms(module_args: str) -> int:
    match = PLAY_DELAY_PATTERN.search(module_args)
    if match is None:
        return 0
    numerator, denominator = (int(value) for value in match.groups())
    if denominator == 0:
        return 0
    return int(round(1000.0 * numerator / denominator))


def add_play_delay(module_args: str, play_delay_ms: int | None) -> str:
    if play_delay_ms is None:
        return module_args
    if play_delay_ms < 0:
        raise ValueError("play delay cannot be negative")
    stripped = module_args.strip()
    if not stripped.startswith("{"):
        raise ValueError("AEC module arguments must start with an object")
    replacement = f"buffer.play_delay = {play_delay_ms}/1000"
    if PLAY_DELAY_PATTERN.search(stripped):
        return PLAY_DELAY_PATTERN.sub(replacement, stripped, count=1)
    return (
        "{ " + replacement + " " + stripped[1:]
    )


def wait_for_nodes_gone(expected: set[str], timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if expected.isdisjoint(pipewire_nodes()):
            return
        time.sleep(0.1)


class AecHost:
    def __init__(self, args_file: Path, play_delay_ms: int | None) -> None:
        self._args_file = args_file
        self._play_delay_ms = play_delay_ms
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "AecHost":
        args = add_play_delay(
            self._args_file.read_text(encoding="utf-8"),
            self._play_delay_ms,
        )
        self._process = subprocess.Popen(
            ["pw-cli"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert self._process.stdin is not None
        command = f"load-module libpipewire-module-echo-cancel {args}\n"
        self._process.stdin.write(command.encode())
        self._process.stdin.flush()
        wait_for_nodes({"streamglass_aec_source", "streamglass_aec_sink"})
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3)
        wait_for_nodes_gone(
            {"streamglass_aec_source", "streamglass_aec_sink"}
        )


def start_capture(node: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "pw-record",
            "--target",
            node,
            "--rate",
            str(SAMPLE_RATE),
            "--channels",
            str(CHANNELS),
            "--format",
            FORMAT,
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def drain_stream(stream: BinaryIO, destination: bytearray) -> None:
    while True:
        block = stream.read(16_384)
        if not block:
            return
        destination.extend(block)


def stop_process(process: subprocess.Popen[bytes]) -> str:
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    assert process.stderr is not None
    return process.stderr.read().decode(errors="replace").strip()


def play_signal(node: str, samples: np.ndarray) -> str:
    process = subprocess.Popen(
        [
            "pw-play",
            "--target",
            node,
            "--rate",
            str(SAMPLE_RATE),
            "--channels",
            str(CHANNELS),
            "--format",
            FORMAT,
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(samples.astype("<f4", copy=False).tobytes())
    if process.returncode != 0:
        raise RuntimeError(
            f"pw-play failed with code {process.returncode}: "
            f"{stderr.decode(errors='replace').strip()}"
        )
    return stderr.decode(errors="replace").strip()


def capture_echo(
    raw_source: str,
    aec_source: str,
    aec_sink: str,
    playback: np.ndarray,
    settle_seconds: float,
    tail_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    capture_started = time.monotonic()
    raw_process = start_capture(raw_source)
    aec_process = start_capture(aec_source)
    assert raw_process.stdout is not None
    assert aec_process.stdout is not None
    raw_bytes = bytearray()
    aec_bytes = bytearray()
    raw_thread = threading.Thread(
        target=drain_stream, args=(raw_process.stdout, raw_bytes), daemon=True
    )
    aec_thread = threading.Thread(
        target=drain_stream, args=(aec_process.stdout, aec_bytes), daemon=True
    )
    raw_thread.start()
    aec_thread.start()
    try:
        time.sleep(settle_seconds)
        playback_started = time.monotonic()
        play_signal(aec_sink, playback)
        time.sleep(tail_seconds)
    finally:
        raw_error = stop_process(raw_process)
        aec_error = stop_process(aec_process)
        raw_thread.join(timeout=3)
        aec_thread.join(timeout=3)
    if not raw_bytes or not aec_bytes:
        raise RuntimeError(
            "capture returned no PCM: "
            f"raw={len(raw_bytes)}B ({raw_error!r}), "
            f"aec={len(aec_bytes)}B ({aec_error!r})"
        )
    raw = np.frombuffer(raw_bytes, dtype="<f4").copy()
    aec = np.frombuffer(aec_bytes, dtype="<f4").copy()
    return raw, aec, playback_started - capture_started


def db_ratio(numerator: float, denominator: float) -> float:
    floor = 1e-9
    return 20.0 * math.log10(max(numerator, floor) / max(denominator, floor))


def estimate_echo_delay_ms(
    capture: np.ndarray,
    playback: np.ndarray,
    playback_start_seconds: float,
    max_delay_ms: float = 600.0,
) -> tuple[float, float]:
    reference_samples = min(playback.size, 3 * SAMPLE_RATE)
    reference = playback[:reference_samples].astype(np.float64, copy=True)
    reference -= np.mean(reference)
    expected = int(round(playback_start_seconds * SAMPLE_RATE))
    early = int(round(0.2 * SAMPLE_RATE))
    search_start = max(0, expected - early)
    search_stop = min(
        capture.size,
        expected
        + reference_samples
        + int(round(max_delay_ms * SAMPLE_RATE / 1000.0)),
    )
    search = capture[search_start:search_stop].astype(np.float64, copy=True)
    search -= np.mean(search)
    valid = search.size - reference.size + 1
    if valid <= 0:
        return float("nan"), 0.0
    fft_size = 1 << (search.size + reference.size - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(search, fft_size)
        * np.conj(np.fft.rfft(reference, fft_size)),
        fft_size,
    )[:valid]
    energy = np.concatenate(([0.0], np.cumsum(search * search)))
    window_energy = energy[reference.size:] - energy[:-reference.size]
    denominator = np.sqrt(
        np.maximum(window_energy, 1e-18)
        * max(float(np.dot(reference, reference)), 1e-18)
    )
    normalized = np.abs(correlation) / denominator
    best = int(np.argmax(normalized))
    capture_index = search_start + best
    delay_samples = capture_index - expected
    return delay_samples * 1000.0 / SAMPLE_RATE, float(normalized[best])


def evaluate(
    raw: np.ndarray,
    aec: np.ndarray,
    settle_seconds: float,
    playback_seconds: float,
    vad_rms: float,
    playback: np.ndarray | None = None,
    playback_start_seconds: float | None = None,
) -> dict[str, object]:
    start = int(round((settle_seconds + 0.5) * SAMPLE_RATE))
    stop = int(round((settle_seconds + playback_seconds - 0.5) * SAMPLE_RATE))
    if stop <= start:
        raise ValueError("playback interval is too short for analysis")
    raw_window = raw[start:min(stop, raw.size)]
    aec_window = aec[start:min(stop, aec.size)]
    count = min(raw_window.size, aec_window.size)
    raw_stats = signal_stats(raw_window[:count])
    aec_stats = signal_stats(aec_window[:count])
    attenuation = db_ratio(aec_stats.rms, raw_stats.rms)
    passed = aec_stats.window_rms_p95 < vad_rms and attenuation <= -6.0
    result: dict[str, object] = {
        "passed": passed,
        "sample_rate": SAMPLE_RATE,
        "analysis_seconds": count / SAMPLE_RATE,
        "vad_rms": vad_rms,
        "attenuation_db": attenuation,
        "raw": asdict(raw_stats),
        "aec": asdict(aec_stats),
        "acceptance": {
            "aec_window_rms_p95_below_vad": aec_stats.window_rms_p95 < vad_rms,
            "attenuation_at_least_6_db": attenuation <= -6.0,
        },
    }
    if playback is not None and playback_start_seconds is not None:
        raw_delay, raw_correlation = estimate_echo_delay_ms(
            raw, playback, playback_start_seconds
        )
        aec_delay, aec_correlation = estimate_echo_delay_ms(
            aec, playback, playback_start_seconds
        )
        result["echo_path"] = {
            "raw_delay_ms": raw_delay,
            "raw_normalized_correlation": raw_correlation,
            "aec_delay_ms": aec_delay,
            "aec_normalized_correlation": aec_correlation,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--sample", type=Path, default=root / "tests/data/sample_spa.wav"
    )
    parser.add_argument(
        "--aec-args",
        type=Path,
        default=root / "config/pipewire/streamglass-echo-cancel.args",
    )
    parser.add_argument(
        "--raw-source", default="bluez_input.41_42_84_69_36_F8.0"
    )
    parser.add_argument("--aec-source", default="streamglass_aec_source")
    parser.add_argument("--aec-sink", default="streamglass_aec_sink")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--gain", type=float, default=1.8)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--vad-rms", type=float, default=0.02)
    parser.add_argument(
        "--play-delay-ms",
        type=int,
        help="delay the AEC playback reference by this many milliseconds",
    )
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module_args = args.aec_args.read_text(encoding="utf-8")
    effective_play_delay_ms = (
        args.play_delay_ms
        if args.play_delay_ms is not None
        else configured_play_delay_ms(module_args)
    )
    sample = trim_silence(read_official_sample(args.sample))
    playback = repeat_with_crossfade(sample, args.seconds, args.gain)
    print(
        "[aec-test] playback "
        f"seconds={playback.size / SAMPLE_RATE:.3f} "
        f"rms={rms(playback):.6f} "
        f"peak={float(np.max(np.abs(playback))):.6f}"
    )
    with AecHost(args.aec_args, args.play_delay_ms):
        print(
            "[aec-test] dynamic PipeWire AEC ready "
            f"play_delay_ms={effective_play_delay_ms}"
        )
        raw, aec, playback_start_seconds = capture_echo(
            args.raw_source,
            args.aec_source,
            args.aec_sink,
            playback,
            args.settle_seconds,
            args.tail_seconds,
        )
    result = evaluate(
        raw,
        aec,
        args.settle_seconds,
        args.seconds,
        args.vad_rms,
        playback,
        playback_start_seconds,
    )
    result["configured_play_delay_ms"] = effective_play_delay_ms
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(serialized + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
