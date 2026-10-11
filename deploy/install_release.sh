#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="/opt/streamglass"

usage() {
  echo "Usage: sudo $0 --source DIR [--release-id ID] [--root DIR]" >&2
}

while (($#)); do
  case "$1" in
    --source) shift; SOURCE="${1:?missing source}" ;;
    --release-id) shift; RELEASE_ID="${1:?missing release id}" ;;
    --root) shift; ROOT="${1:?missing root}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "install_release.sh must run as root" >&2
  exit 1
fi
if [[ -z "$SOURCE" || ! -d "$SOURCE" ]]; then
  echo "valid --source is required" >&2
  exit 1
fi

destination="$ROOT/releases/$RELEASE_ID"
if [[ -e "$destination" ]]; then
  echo "release already exists: $destination" >&2
  exit 1
fi

install -d -o root -g streamglass -m 2755 "$destination"
cp -a "$SOURCE/." "$destination/"
chown -R root:streamglass "$destination"
ln -sfn "$destination" "$ROOT/current.next"
mv -Tf "$ROOT/current.next" "$ROOT/current"

echo "activated release $RELEASE_ID"

