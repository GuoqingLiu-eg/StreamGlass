#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_HOST="${TARGET_HOST:-192.168.2.107}"
TARGET_PORT="${TARGET_PORT:-2222}"
TARGET_USER="${TARGET_USER:-jetson}"
TARGET_KEY="${TARGET_KEY:-$HOME/.ssh/id_ed25519}"
TARGET_STAGING="${TARGET_STAGING:-/home/$TARGET_USER/streamglass-deploy-staging}"

SSH=(ssh -i "$TARGET_KEY" -o BatchMode=yes -o ServerAliveInterval=15 -p "$TARGET_PORT")
RSYNC_SSH="ssh -i $TARGET_KEY -o BatchMode=yes -o ServerAliveInterval=15 -p $TARGET_PORT"

"${SSH[@]}" "$TARGET_USER@$TARGET_HOST" "mkdir -p '$TARGET_STAGING'"

rsync -a --partial --checksum \
  --exclude '/.git/' \
  --exclude '__pycache__/' \
  --exclude '/artifacts/' \
  --exclude '/engines/' \
  --exclude '/logs/' \
  --exclude '/models/' \
  -e "$RSYNC_SSH" \
  "$PROJECT_ROOT/" "$TARGET_USER@$TARGET_HOST:$TARGET_STAGING/"

echo "project synchronized to $TARGET_USER@$TARGET_HOST:$TARGET_STAGING"
