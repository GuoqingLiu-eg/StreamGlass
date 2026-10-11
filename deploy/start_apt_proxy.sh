#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="streamglass/apt-proxy:noble"
CONTAINER="streamglass-apt-proxy"

docker build -t "$IMAGE" "$SCRIPT_DIR/apt-proxy"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --rm --name "$CONTAINER" -p 0.0.0.0:3128:3128 "$IMAGE"

echo "temporary apt proxy listening on TCP 3128"

