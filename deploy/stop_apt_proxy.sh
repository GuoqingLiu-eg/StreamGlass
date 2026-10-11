#!/usr/bin/env bash
set -euo pipefail

docker rm -f streamglass-apt-proxy >/dev/null 2>&1 || true
echo "temporary apt proxy stopped"
