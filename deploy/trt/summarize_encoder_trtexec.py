#!/usr/bin/env python3
"""Summarize cumulative encoder trtexec traces and enforce its local budget."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPORT_PATTERN = re.compile(r"^(compute|end_to_end)_f(\d{4})\.json$")
METRICS = ("latencyMs", "computeMs", "h2dMs", "d2hMs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    cases: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.glob("*.json")):
        match = REPORT_PATTERN.match(path.name)
        if match is None:
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty or invalid trtexec trace: {path}")
        metrics = {
            metric: summarize([float(row[metric]) for row in rows])
            for metric in METRICS
        }
        mode = match.group(1)
        p99 = metrics["latencyMs"]["p99"]
        cases.append(
            {
                "mode": mode,
                "fbank_length": int(match.group(2)),
                "samples": len(rows),
                "metrics_ms": metrics,
                "budget_passed": mode != "end_to_end" or p99 < args.budget_ms,
            }
        )
    if not cases:
        raise FileNotFoundError(f"No encoder trtexec reports under {args.input_dir}")

    end_to_end_cases = [case for case in cases if case["mode"] == "end_to_end"]
    passed = bool(end_to_end_cases and all(case["budget_passed"] for case in end_to_end_cases))
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "cumulative_encoder_only",
        "frontend_included": False,
        "decoder_included": False,
        "budget_ms": args.budget_ms,
        "passed": passed,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
