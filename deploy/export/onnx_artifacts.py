#!/usr/bin/env python3
"""Index and verify ONNX models that use external tensor data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact must remain below {root}: {relative_name!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Artifact escapes {resolved_root}: {relative_name!r}"
        ) from error
    return resolved


def _iter_tensor_protos(message: Any, tensor_type: type) -> Iterator[Any]:
    if isinstance(message, tensor_type):
        yield message
        return
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.label == field.LABEL_REPEATED:
            for child in value:
                yield from _iter_tensor_protos(child, tensor_type)
        else:
            yield from _iter_tensor_protos(value, tensor_type)


def external_data_locations(onnx_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in _iter_tensor_protos(model, onnx.TensorProto):
        entries = {entry.key: entry.value for entry in tensor.external_data}
        location = entries.get("location")
        if location:
            locations.add(location)
    return sorted(locations)


def index_onnx_artifacts(onnx_path: Path) -> dict[str, Any]:
    onnx_path = onnx_path.resolve()
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing ONNX protobuf: {onnx_path}")
    root = onnx_path.parent
    relative_names = [onnx_path.name, *external_data_locations(onnx_path)]
    artifacts = []
    seen: set[str] = set()
    for relative_name in relative_names:
        if relative_name in seen:
            continue
        seen.add(relative_name)
        path = resolve_artifact(root, relative_name)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing ONNX artifact: {path}")
        artifacts.append(
            {
                "file": relative_name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "external_data_file_count": len(artifacts) - 1,
        "artifact_bytes": sum(item["bytes"] for item in artifacts),
    }


def verify_manifest_artifacts(
    manifest_path: Path, artifact_root: Path | None = None
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    root = (artifact_root or manifest_path.parent).resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"No artifacts indexed in {manifest_path}")
    verified_bytes = 0
    for artifact in artifacts:
        relative_name = artifact.get("file")
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError(f"Invalid artifact entry: {artifact!r}")
        path = resolve_artifact(root, relative_name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing indexed artifact: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != int(artifact["bytes"]):
            raise ValueError(
                f"Size mismatch for {path}: {actual_bytes} != {artifact['bytes']}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != artifact["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path}: {actual_sha256} != "
                f"{artifact['sha256']}"
            )
        verified_bytes += actual_bytes
    declared_count = manifest.get("artifact_count")
    if declared_count is not None and int(declared_count) != len(artifacts):
        raise ValueError(
            f"Artifact count mismatch: {len(artifacts)} != {declared_count}"
        )
    declared_bytes = manifest.get("artifact_bytes")
    if declared_bytes is not None and int(declared_bytes) != verified_bytes:
        raise ValueError(
            f"Artifact byte count mismatch: {verified_bytes} != {declared_bytes}"
        )
    return {
        "manifest": str(manifest_path),
        "artifact_count": len(artifacts),
        "external_data_file_count": int(
            manifest.get("external_data_file_count", max(0, len(artifacts) - 1))
        ),
        "artifact_bytes": verified_bytes,
        "verified": True,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
