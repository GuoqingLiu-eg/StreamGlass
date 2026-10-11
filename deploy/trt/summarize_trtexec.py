#!/usr/bin/env python3
"""Summarize raw trtexec timing traces into stable percentile reports."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPORT_PATTERN = re.compile(r"^(compute|end_to_end)_len(\d{4})\.json$")
METRICS = ("latencyMs", "computeMs", "h2dMs", "d2hMs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        case = {
            "mode": match.group(1),
            "unit_length": int(match.group(2)),
            "samples": len(rows),
            "metrics_ms": {
                metric: summarize([float(row[metric]) for row in rows])
                for metric in METRICS
            },
        }
        cases.append(case)

    if not cases:
        raise FileNotFoundError(f"No trtexec reports under {args.input_dir}")

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_output.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
