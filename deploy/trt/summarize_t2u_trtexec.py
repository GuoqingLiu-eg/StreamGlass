#!/usr/bin/env python3
"""Summarize T2U TensorRT traces and enforce a component latency gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPORT_PATTERN = re.compile(
    r"^(frontend|decoder)_(compute|end_to_end)_"
    r"(?:s(?P<text>\d{4})_c(?P<char>\d{4})|u(?P<unit>\d{4}))\.json$"
)
METRICS = ("latencyMs", "computeMs", "h2dMs", "d2hMs")
PATHS = {
    "real_short": ((7, 26), 57),
    "real_final": ((14, 54), 131),
    "profile_opt": ((32, 128), 256),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend-engine", type=Path, required=True)
    parser.add_argument("--decoder-engine", type=Path, required=True)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def case_key(case: dict[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    shape = case["shape"]
    if case["module"] == "frontend":
        dimensions = (shape["text_length"], shape["char_length"])
    else:
        dimensions = (shape["unit_length"],)
    return case["module"], case["mode"], dimensions


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

        module = match.group(1)
        if module == "frontend":
            shape = {
                "text_length": int(match.group("text")),
                "char_length": int(match.group("char")),
            }
        else:
            shape = {"unit_length": int(match.group("unit"))}

        cases.append(
            {
                "module": module,
                "mode": match.group(2),
                "shape": shape,
                "samples": len(rows),
                "metrics_ms": {
                    metric: summarize([float(row[metric]) for row in rows])
                    for metric in METRICS
                },
            }
        )

    expected_cases = len(PATHS) * 2 * 2
    if len(cases) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} T2U traces, found {len(cases)} under "
            f"{args.input_dir}"
        )

    by_key = {case_key(case): case for case in cases}
    paired_paths: list[dict[str, Any]] = []
    for name, (frontend_shape, unit_length) in PATHS.items():
        frontend = by_key[("frontend", "end_to_end", frontend_shape)]
        decoder = by_key[("decoder", "end_to_end", (unit_length,))]
        p99_sum = (
            frontend["metrics_ms"]["latencyMs"]["p99"]
            + decoder["metrics_ms"]["latencyMs"]["p99"]
        )
        paired_paths.append(
            {
                "name": name,
                "text_length": frontend_shape[0],
                "char_length": frontend_shape[1],
                "unit_length": unit_length,
                "frontend_p99_ms": frontend["metrics_ms"]["latencyMs"]["p99"],
                "decoder_p99_ms": decoder["metrics_ms"]["latencyMs"]["p99"],
                "summed_p99_ms": p99_sum,
                "budget_ms": args.budget_ms,
                "passed": p99_sum < args.budget_ms,
            }
        )

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "t2u_frontend_plus_decoder_only",
        "host_upsample_included": False,
        "frontend_engine": str(args.frontend_engine),
        "frontend_engine_sha256": sha256(args.frontend_engine),
        "decoder_engine": str(args.decoder_engine),
        "decoder_engine_sha256": sha256(args.decoder_engine),
        "cases": cases,
        "paired_paths": paired_paths,
        "passed": all(path["passed"] for path in paired_paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_output.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
