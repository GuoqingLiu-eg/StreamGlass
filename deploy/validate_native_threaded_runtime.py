#!/usr/bin/env python3
"""Validate deterministic threaded inference in one long-lived AGX process."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_TRANSLATION = "Have you ever considered writing a column of advice?"
SAMPLES_PER_UTTERANCE = 63_488
OUTPUT_SAMPLES_PER_UTTERANCE = 43_840
POLICIES_PER_UTTERANCE = 7
WRITES_PER_UTTERANCE = 2


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        type=Path,
        default=project_root / "deploy/run_native_threaded_file.sh",
    )
    parser.add_argument("--utterances", type=int, default=3)
    parser.add_argument("--max-inference-ms", type=float, default=240.0)
    parser.add_argument("--max-realtime-queue-ms", type=float, default=5.0)
    parser.add_argument("--max-rss-growth-mib", type=float, default=128.0)
    parser.add_argument(
        "--max-cuda-free-drop-mib",
        type=float,
        default=128.0,
        help=(
            "maximum tolerated change in device-wide CUDA free memory; "
            "Jetson unified-memory accounting is not process-local"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_mode(
    runner: Path, utterances: int, mode: str, verbose: bool
) -> dict[str, Any]:
    command = [str(runner), "--repeat", str(utterances)]
    if mode == "fast":
        command.append("--fast")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if verbose or completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} runtime exited with code {completed.returncode}"
        )
    result_lines = [
        line.removeprefix("RESULT ")
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT ")
    ]
    if len(result_lines) != 1:
        raise RuntimeError(
            f"{mode} runtime produced {len(result_lines)} RESULT records"
        )
    result = json.loads(result_lines[0])
    if not isinstance(result, dict):
        raise RuntimeError(f"{mode} RESULT is not a JSON object")
    return result


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_common(
    result: dict[str, Any],
    mode: str,
    utterances: int,
    max_inference_ms: float,
    failures: list[str],
) -> None:
    require(result.get("mode") == mode, f"{mode}: mode mismatch", failures)
    require(
        result.get("utterances") == utterances,
        f"{mode}: utterance count mismatch",
        failures,
    )
    require(
        result.get("translation") == EXPECTED_TRANSLATION,
        f"{mode}: translation changed",
        failures,
    )
    require(
        result.get("input_samples") == SAMPLES_PER_UTTERANCE * utterances,
        f"{mode}: input sample count changed",
        failures,
    )
    require(
        result.get("output_samples")
        == OUTPUT_SAMPLES_PER_UTTERANCE * utterances,
        f"{mode}: output sample count changed",
        failures,
    )
    require(
        result.get("writes") == WRITES_PER_UTTERANCE * utterances,
        f"{mode}: WRITE count changed",
        failures,
    )
    require(
        result.get("policies") == POLICIES_PER_UTTERANCE * utterances,
        f"{mode}: official 320 ms policy count changed",
        failures,
    )
    require(
        float(result.get("max_inference_ms", math.inf)) <= max_inference_ms,
        f"{mode}: max inference exceeded {max_inference_ms:.1f} ms",
        failures,
    )
    require(
        int(result.get("deadline_misses", utterances + 1)) <= utterances,
        f"{mode}: more than one 200 ms target miss per utterance",
        failures,
    )
    require(
        int(result.get("playback_queue_high_water", 99)) <= 2,
        f"{mode}: playback queue accumulated",
        failures,
    )
    require(
        math.isfinite(float(result.get("output_rms", math.nan)))
        and float(result.get("output_rms", 0.0)) > 0.0,
        f"{mode}: invalid output RMS",
        failures,
    )
    require(
        math.isfinite(float(result.get("output_peak", math.nan)))
        and 0.0 < float(result.get("output_peak", 0.0)) <= 1.0,
        f"{mode}: invalid output peak",
        failures,
    )


def validate_resources(
    result: dict[str, Any],
    mode: str,
    max_rss_growth_mib: float,
    max_cuda_free_drop_mib: float,
    failures: list[str],
) -> None:
    required = (
        "rss_before_kib",
        "rss_after_kib",
        "peak_rss_kib",
        "cuda_free_before_bytes",
        "cuda_free_after_bytes",
        "cuda_free_delta_bytes",
    )
    require(
        all(field in result for field in required),
        f"{mode}: resource telemetry is incomplete",
        failures,
    )
    require(
        int(result.get("rss_before_kib", 0)) > 0
        and int(result.get("rss_after_kib", 0)) > 0
        and int(result.get("cuda_free_before_bytes", 0)) > 0
        and int(result.get("cuda_free_after_bytes", 0)) > 0,
        f"{mode}: resource telemetry is invalid",
        failures,
    )
    rss_growth_kib = int(result.get("rss_after_kib", 0)) - int(
        result.get("rss_before_kib", 0)
    )
    # cudaMemGetInfo() reports device-wide free unified memory on Jetson. Keep
    # it as a coarse exhaustion guard; process leak detection is based on RSS
    # and repeated-run slope probes rather than treating this delta as owned by
    # the runtime process.
    cuda_free_drop_bytes = -int(result.get("cuda_free_delta_bytes", 0))
    require(
        rss_growth_kib <= max_rss_growth_mib * 1024.0,
        f"{mode}: RSS grew by {rss_growth_kib / 1024.0:.1f} MiB",
        failures,
    )
    require(
        cuda_free_drop_bytes <= max_cuda_free_drop_mib * 1024.0 * 1024.0,
        f"{mode}: CUDA free memory dropped by "
        f"{cuda_free_drop_bytes / (1024.0 * 1024.0):.1f} MiB",
        failures,
    )


def main() -> int:
    args = parse_args()
    if args.utterances <= 0:
        raise SystemExit("--utterances must be positive")
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"runner not found: {runner}")

    realtime = run_mode(runner, args.utterances, "realtime", args.verbose)
    fast = run_mode(runner, args.utterances, "fast", args.verbose)
    failures: list[str] = []
    validate_common(
        realtime,
        "realtime",
        args.utterances,
        args.max_inference_ms,
        failures,
    )
    validate_resources(
        realtime,
        "realtime",
        args.max_rss_growth_mib,
        args.max_cuda_free_drop_mib,
        failures,
    )
    validate_resources(
        fast,
        "fast",
        args.max_rss_growth_mib,
        args.max_cuda_free_drop_mib,
        failures,
    )
    validate_common(
        fast,
        "fast",
        args.utterances,
        args.max_inference_ms,
        failures,
    )
    require(
        float(realtime.get("max_capture_queue_ms", math.inf))
        <= args.max_realtime_queue_ms,
        "realtime: capture queue latency exceeded threshold",
        failures,
    )
    require(
        int(realtime.get("audio_queue_high_water", 99)) <= 2,
        "realtime: audio queue accumulated",
        failures,
    )
    require(
        int(fast.get("coalesced_batches", -1)) == 0,
        "fast: policy calls were coalesced",
        failures,
    )
    require(
        int(fast.get("max_coalesced_blocks", 0)) == 1,
        "fast: more than one 320 ms block entered a policy call",
        failures,
    )
    for field in ("output_samples", "output_checksum"):
        require(
            realtime.get(field) == fast.get(field),
            f"realtime/fast: {field} is not deterministic",
            failures,
        )
    for field in ("output_rms", "output_peak"):
        require(
            math.isclose(
                float(realtime.get(field, math.nan)),
                float(fast.get(field, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-7,
            ),
            f"realtime/fast: {field} changed",
            failures,
        )

    report = {
        "passed": not failures,
        "utterances_per_mode": args.utterances,
        "hard_inference_limit_ms": args.max_inference_ms,
        "target_inference_ms": 200.0,
        "max_rss_growth_mib": args.max_rss_growth_mib,
        "max_cuda_free_drop_mib": args.max_cuda_free_drop_mib,
        "cuda_memory_metric_scope": "device-global-advisory",
        "realtime": realtime,
        "fast": fast,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
