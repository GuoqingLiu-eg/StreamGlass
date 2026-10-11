#!/usr/bin/env python3
"""Run a no-disk, physical 15-second model/playback feedback regression."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from validate_full_duplex_aec import (
    AecHost,
    FORMAT,
    SAMPLE_RATE,
    read_official_sample,
    repeat_with_crossfade,
    start_capture,
    trim_silence,
)


BLOCK_SAMPLES = 512
BLOCK_BYTES = BLOCK_SAMPLES * np.dtype("<f4").itemsize
UTTERANCE_PATTERN = re.compile(r"^\[realtime\] utterance (\d+):\s*(.*)$")


def read_exact(stream: object, count: int) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = stream.read(count - len(output))  # type: ignore[attr-defined]
        if not block:
            break
        output.extend(block)
    return bytes(output)


def terminate(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    fingerprint = os.environ.get(
        "STREAMGLASS_PLATFORM_FINGERPRINT",
        "87d587047967766c63f3de7012a479fbd4f3c471ffea70efe8454017bf908709",
    )
    parser.add_argument(
        "--sample", type=Path, default=root / "tests/data/sample_spa.wav"
    )
    parser.add_argument(
        "--aec-args",
        type=Path,
        default=root / "config/pipewire/streamglass-echo-cancel.args",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(
            f"/opt/streamglass/tmp/native-install/{fingerprint}/bin/"
            "streamglass-native-realtime"
        ),
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path(f"/opt/streamglass/engine-cache/{fingerprint}"),
    )
    parser.add_argument(
        "--runtime-assets",
        type=Path,
        default=Path("/opt/streamglass/tmp/runtime-assets/official-v1"),
    )
    parser.add_argument("--target-language", default="eng")
    parser.add_argument(
        "--aec-source", default="streamglass_aec_source"
    )
    parser.add_argument("--aec-sink", default="streamglass_aec_sink")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--post-playback-seconds", type=float, default=8.0)
    parser.add_argument("--maximum-seconds", type=float, default=120.0)
    parser.add_argument("--expected-utterances", type=int, default=1)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def run_regression(args: argparse.Namespace) -> dict[str, object]:
    official = repeat_with_crossfade(
        trim_silence(read_official_sample(args.sample)),
        args.seconds,
        args.gain,
    )
    official_bytes = official.astype("<f4", copy=False).tobytes()
    source_blocks = (len(official_bytes) + BLOCK_BYTES - 1) // BLOCK_BYTES
    audio_queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue(maxsize=4096)
    stop_event = threading.Event()
    state_lock = threading.Lock()
    logs: list[str] = []
    translations: list[str] = []
    source_finished_at: float | None = None
    playback_samples = 0
    last_playback_at: float | None = None
    max_queued_blocks = 0
    ready_event = threading.Event()

    environment = os.environ.copy()
    environment["STREAMGLASS_HALF_DUPLEX"] = "0"

    native = subprocess.Popen(
        [
            str(args.binary),
            str(args.engine_dir),
            str(args.runtime_assets),
            args.target_language,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    playback = subprocess.Popen(
        [
            "pw-play",
            "--target",
            args.aec_sink,
            "--rate",
            str(SAMPLE_RATE),
            "--channels",
            "1",
            "--format",
            FORMAT,
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert native.stdin is not None
    assert native.stdout is not None
    assert native.stderr is not None
    assert playback.stdin is not None

    capture: subprocess.Popen[bytes] | None = None

    def capture_worker() -> None:
        nonlocal max_queued_blocks
        assert capture is not None and capture.stdout is not None
        sequence = 0
        try:
            while not stop_event.is_set():
                block = read_exact(capture.stdout, BLOCK_BYTES)
                if len(block) != BLOCK_BYTES:
                    break
                audio_queue.put((sequence, block))
                sequence += 1
                with state_lock:
                    max_queued_blocks = max(max_queued_blocks, audio_queue.qsize())
        finally:
            try:
                audio_queue.put_nowait(None)
            except queue.Full:
                pass

    def input_worker() -> None:
        nonlocal source_finished_at
        try:
            while not stop_event.is_set():
                item = audio_queue.get()
                if item is None:
                    break
                sequence, captured = item
                offset = sequence * BLOCK_BYTES
                if sequence < source_blocks:
                    payload = official_bytes[offset : offset + BLOCK_BYTES]
                    if len(payload) < BLOCK_BYTES:
                        payload += bytes(BLOCK_BYTES - len(payload))
                else:
                    payload = captured
                    if source_finished_at is None:
                        source_finished_at = time.monotonic()
                        print(
                            "[feedback-test] official source finished "
                            f"seconds={args.seconds:.3f}"
                        )
                native.stdin.write(payload)
                native.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                native.stdin.close()
            except OSError:
                pass

    def log_worker() -> None:
        for raw_line in iter(native.stderr.readline, b""):
            line = raw_line.decode(errors="replace").rstrip()
            with state_lock:
                logs.append(line)
                match = UTTERANCE_PATTERN.match(line)
                if match:
                    translations.append(match.group(2))
            if line.startswith("[realtime] ready"):
                ready_event.set()
            if (
                line.startswith("[realtime]")
                or line.startswith("[playback]")
                or (line.startswith("[meter]") and "ACTIVE" in line)
                or line.startswith("ERROR:")
            ):
                print(line)

    def playback_worker() -> None:
        nonlocal playback_samples, last_playback_at
        try:
            while True:
                block = native.stdout.read(16_384)
                if not block:
                    break
                playback.stdin.write(block)
                playback.stdin.flush()
                with state_lock:
                    playback_samples += len(block) // np.dtype("<f4").itemsize
                    last_playback_at = time.monotonic()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                playback.stdin.close()
            except OSError:
                pass

    started = time.monotonic()
    log_thread = threading.Thread(target=log_worker, name="log", daemon=True)
    playback_thread = threading.Thread(
        target=playback_worker, name="playback", daemon=True
    )
    log_thread.start()
    playback_thread.start()
    if not ready_event.wait(timeout=args.maximum_seconds):
        stop_event.set()
        try:
            native.stdin.close()
        except OSError:
            pass
        terminate(native)
        try:
            playback.stdin.close()
        except OSError:
            pass
        terminate(playback)
        log_thread.join(timeout=5)
        playback_thread.join(timeout=5)
        raise RuntimeError("native runtime did not become ready")
    load_seconds = time.monotonic() - started
    print(f"[feedback-test] native ready in {load_seconds:.3f}s")
    capture = start_capture(args.aec_source)
    capture_thread = threading.Thread(
        target=capture_worker, name="capture", daemon=True
    )
    input_thread = threading.Thread(target=input_worker, name="input", daemon=True)
    capture_thread.start()
    input_thread.start()
    observation_started = time.monotonic()
    reason = "timeout"
    try:
        while time.monotonic() - observation_started < args.maximum_seconds:
            time.sleep(0.25)
            with state_lock:
                utterance_count = len(translations)
                played = playback_samples
                last_playback = last_playback_at
                queued = audio_queue.qsize()
            if native.poll() not in (None, 0):
                reason = f"native-exit-{native.returncode}"
                break
            if (
                source_finished_at is not None
                and utterance_count >= args.expected_utterances
                and played > 0
                and last_playback is not None
                and time.monotonic() - last_playback
                >= args.post_playback_seconds
                and queued < 4
            ):
                reason = "post-playback-observation-complete"
                break
    finally:
        stop_event.set()
        if capture is not None:
            terminate(capture)
        for thread in (capture_thread, input_thread):
            thread.join(timeout=5)
        try:
            native.stdin.close()
        except OSError:
            pass
        try:
            native.wait(timeout=15)
        except subprocess.TimeoutExpired:
            terminate(native)
        log_thread.join(timeout=5)
        playback_thread.join(timeout=5)
        try:
            playback.stdin.close()
        except OSError:
            pass
        try:
            playback.wait(timeout=10)
        except subprocess.TimeoutExpired:
            terminate(playback)

    with state_lock:
        final_translations = list(translations)
        final_logs = list(logs)
        final_playback_samples = playback_samples
        final_max_queue = max_queued_blocks
    errors = [line for line in final_logs if line.startswith("ERROR:")]
    passed = (
        reason == "post-playback-observation-complete"
        and len(final_translations) == args.expected_utterances
        and final_playback_samples > 0
        and not errors
    )
    return {
        "passed": passed,
        "reason": reason,
        "source_seconds": args.seconds,
        "elapsed_seconds": time.monotonic() - started,
        "load_seconds": load_seconds,
        "utterance_count": len(final_translations),
        "expected_utterances": args.expected_utterances,
        "translations": final_translations,
        "playback_samples": final_playback_samples,
        "playback_seconds": final_playback_samples / SAMPLE_RATE,
        "max_input_queue_blocks": final_max_queue,
        "max_input_queue_seconds": final_max_queue * BLOCK_SAMPLES / SAMPLE_RATE,
        "errors": errors,
        "log_tail": final_logs[-30:],
    }


def main() -> int:
    args = parse_args()
    with AecHost(args.aec_args, None):
        print("[feedback-test] PipeWire AEC ready")
        result = run_regression(args)
    serialized = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
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
