#!/usr/bin/env bash
set -euo pipefail

data_root="${STREAMGLASS_DOCKER_ROOT:-/opt/streamglass/docker}"
containerd_root="${STREAMGLASS_CONTAINERD_ROOT:-/opt/streamglass/containerd}"
proxy_url="${STREAMGLASS_BOOTSTRAP_PROXY:-}"
daemon_json=/etc/docker/daemon.json
proxy_dir=/etc/systemd/system/docker.service.d
proxy_file="$proxy_dir/streamglass-proxy.conf"
containerd_dropin_dir=/etc/systemd/system/containerd.service.d
containerd_dropin="$containerd_dropin_dir/streamglass-root.conf"
legacy_containerd_root=/var/lib/containerd

sudo install -d -m 0755 \
    "$data_root" \
    "$containerd_root" \
    /etc/docker \
    "$proxy_dir" \
    "$containerd_dropin_dir"

tmp_json="$(mktemp)"
trap 'rm -f "$tmp_json"' EXIT

python3 - "$daemon_json" "$tmp_json" "$data_root" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
data_root = sys.argv[3]

if source.exists():
    with source.open(encoding="utf-8") as handle:
        config = json.load(handle)
else:
    config = {}

config["data-root"] = data_root
config.setdefault("runtimes", {}).setdefault(
    "nvidia",
    {"args": [], "path": "nvidia-container-runtime"},
)

with target.open("w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ -f "$daemon_json" ]]; then
    sudo cp -a "$daemon_json" "${daemon_json}.streamglass.bak"
fi
sudo install -m 0644 "$tmp_json" "$daemon_json"

if [[ -n "$proxy_url" ]]; then
    tmp_proxy="$(mktemp)"
    trap 'rm -f "$tmp_json" "$tmp_proxy"' EXIT
    cat >"$tmp_proxy" <<EOF
[Service]
Environment="HTTP_PROXY=$proxy_url"
Environment="HTTPS_PROXY=$proxy_url"
Environment="NO_PROXY=localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8"
EOF
    sudo install -m 0644 "$tmp_proxy" "$proxy_file"
else
    sudo rm -f "$proxy_file"
fi

tmp_containerd="$(mktemp)"
trap 'rm -f "$tmp_json" "${tmp_proxy:-}" "$tmp_containerd"' EXIT
cat >"$tmp_containerd" <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/containerd --root $containerd_root
EOF
sudo install -m 0644 "$tmp_containerd" "$containerd_dropin"

sudo systemctl stop docker.service docker.socket
sudo systemctl stop containerd.service

legacy_root_resolved="$(readlink -f "$legacy_containerd_root")"
target_root_resolved="$(readlink -f "$containerd_root")"
if [[ "$legacy_root_resolved" != "$target_root_resolved" ]] \
    && [[ -d "$legacy_containerd_root" ]]; then
    sudo rsync -aHAX --numeric-ids \
        "$legacy_containerd_root/" "$containerd_root/"
fi

sudo systemctl daemon-reload
sudo systemctl start containerd.service
sudo systemctl start docker.service

if ! systemctl is-active --quiet containerd.service; then
    echo "containerd failed to start" >&2
    exit 1
fi
if ! systemctl is-active --quiet docker.service; then
    echo "Docker failed to start" >&2
    exit 1
fi

actual_root="$(sudo docker info --format '{{.DockerRootDir}}')"
expected_root="$(readlink -f "$data_root")"
actual_root_resolved="$(readlink -f "$actual_root")"
if [[ "$actual_root_resolved" != "$expected_root" ]]; then
    echo "Docker data-root mismatch: expected $data_root, got $actual_root" >&2
    exit 1
fi

if ! sudo docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
    echo "NVIDIA Docker runtime is unavailable" >&2
    exit 1
fi

containerd_pid="$(systemctl show --property MainPID --value containerd.service)"
containerd_cmd="$(sudo tr '\0' ' ' <"/proc/$containerd_pid/cmdline")"
if [[ "$containerd_cmd" != *"--root $containerd_root"* ]]; then
    echo "containerd did not start with NVMe root: $containerd_cmd" >&2
    exit 1
fi

echo "Docker data-root: $actual_root"
echo "containerd root: $target_root_resolved"
echo "NVIDIA runtime: available"
if [[ -n "$proxy_url" ]]; then
    echo "Bootstrap proxy: configured"
else
    echo "Bootstrap proxy: disabled"
fi
