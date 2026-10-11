#!/usr/bin/env bash
set -euo pipefail

url="http://127.0.0.1:8080/"
data_root="/mnt/streamglass-data/browser"

mkdir -p "$data_root/cache" "$data_root/profile"
systemctl --user start streamglass-web.service

export XDG_CACHE_HOME="$data_root/cache"
exec /usr/bin/epiphany --profile="$data_root/profile" "$url"
