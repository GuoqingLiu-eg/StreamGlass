#!/usr/bin/env python3
"""Collect a deterministic Jetson deployment fingerprint using stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKAGE_NAMES = (
    "nvidia-l4t-core",
    "nvidia-l4t-cuda",
    "tensorrt",
    "libnvinfer10",
    "libnvinfer-bin",
    "libcudnn9-cuda-13",
    "libcudnn9-dev-cuda-13",
    "pipewire",
    "wireplumber",
    "bluez",
    "python3",
    "python3-pip",
    "python3-venv",
)


def run(command: Iterable[str], timeout: float = 10.0) -> dict[str, Any]:
    argv = list(command)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": argv,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def parse_os_release(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def package_versions(names: Iterable[str]) -> dict[str, dict[str, str]]:
    packages: dict[str, dict[str, str]] = {}
    for name in names:
        result = run(
            [
                "dpkg-query",
                "-W",
                "-f=${db:Status-Abbrev}\t${Version}",
                name,
            ]
        )
        fields = result["stdout"].split("\t", 1)
        installed = result["returncode"] == 0 and fields and fields[0].startswith("ii")
        packages[name] = {
            "status": fields[0].strip() if fields else "missing",
            "version": fields[1].strip() if len(fields) == 2 else "",
            "installed": str(installed).lower(),
        }
    return packages


def parse_cuda_version(raw: str) -> str:
    match = re.search(r"release\s+([0-9.]+)", raw)
    return match.group(1) if match else ""


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("generated_at", None)
    stable.pop("fingerprint", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect() -> dict[str, Any]:
    nvcc = run(["/usr/local/cuda/bin/nvcc", "--version"])
    nvpmodel = run(["nvpmodel", "-q"])
    lsblk = run(["lsblk", "--json", "--bytes", "--output", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL,UUID"])
    memory = run(["free", "--bytes"])
    pipewire = run(["pw-cli", "--version"])
    wireplumber = run(["wireplumber", "--version"])
    bluetooth = run(["bluetoothctl", "show"])
    default_target = run(["systemctl", "get-default"])

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "machine_id": read_text("/etc/machine-id"),
        },
        "os": parse_os_release(read_text("/etc/os-release")),
        "jetson": {
            "l4t_release": read_text("/etc/nv_tegra_release"),
            "device_tree_model": read_text("/proc/device-tree/model").rstrip("\x00"),
            "power_mode": nvpmodel["stdout"],
            "default_systemd_target": default_target["stdout"],
        },
        "accelerators": {
            "cuda_version": parse_cuda_version(nvcc["stdout"]),
            "nvcc": nvcc,
        },
        "packages": package_versions(PACKAGE_NAMES),
        "storage": {
            "lsblk": json.loads(lsblk["stdout"]) if lsblk["returncode"] == 0 else lsblk,
            "memory": memory["stdout"],
        },
        "audio": {
            "pipewire": pipewire,
            "wireplumber": wireplumber,
            "bluetooth_controller": bluetooth,
        },
        "toolchain": {
            "python": platform.python_version(),
            "cmake": run(["cmake", "--version"]),
            "compiler": run(["c++", "--version"]),
        },
        "environment": {
            "cuda_home": os.environ.get("CUDA_HOME", ""),
        },
    }
    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = collect()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

