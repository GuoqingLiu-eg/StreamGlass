#!/usr/bin/env python3
"""Create and verify the immutable Seamless Streaming model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROLES = {
    "seamless_streaming_unity.pt": "speech_encoder_and_t2u",
    "seamless_streaming_monotonic_decoder.pt": "monotonic_text_decoder",
    "vocoder_v2.pt": "unit_vocoder",
    "tokenizer.model": "text_tokenizer",
    "spm_char_lang38_tc.model": "unit_tokenizer",
}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(model_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name, role in ROLES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing required model artifact: {path}")
        files.append(
            {
                "name": name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "model_family": "facebook/seamless-streaming",
        "files": files,
    }


def verify_manifest(model_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("unsupported manifest schema_version")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return errors + ["manifest files must be a list"]

    found_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            errors.append("invalid manifest entry")
            continue
        name = entry["name"]
        found_names.add(name)
        path = model_dir / name
        if not path.is_file():
            errors.append(f"missing: {name}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != entry.get("bytes"):
            errors.append(f"size mismatch: {name}: expected={entry.get('bytes')} actual={actual_bytes}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != entry.get("sha256"):
            errors.append(f"sha256 mismatch: {name}: expected={entry.get('sha256')} actual={actual_sha}")

    missing_entries = set(ROLES) - found_names
    errors.extend(f"manifest entry missing: {name}" for name in sorted(missing_entries))
    return errors


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--model-dir", required=True, type=Path)
    create_parser.add_argument("--output", required=True, type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--model-dir", required=True, type=Path)
    verify_parser.add_argument("--manifest", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "create":
        manifest = create_manifest(args.model_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 0

    errors = verify_manifest(args.model_dir, load_json(args.manifest))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"verified {len(ROLES)} model artifacts in {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

