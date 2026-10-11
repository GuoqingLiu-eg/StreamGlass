#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_HOST="${TARGET_HOST:-192.168.2.107}"
TARGET_PORT="${TARGET_PORT:-2222}"
TARGET_USER="${TARGET_USER:-jetson}"
TARGET_KEY="${TARGET_KEY:-$HOME/.ssh/id_ed25519}"
MODEL_DIR="${MODEL_DIR:-/Users/Zhuanz/Desktop/model/streamspeech_mlx_deploy/.models/seamless_streaming}"
REMOTE_MODEL_DIR="${REMOTE_MODEL_DIR:-/opt/streamglass/models}"

SSH=(ssh -i "$TARGET_KEY" -o BatchMode=yes -o ServerAliveInterval=15 -p "$TARGET_PORT")
RSYNC_SSH="ssh -i $TARGET_KEY -o BatchMode=yes -o ServerAliveInterval=15 -p $TARGET_PORT"

python3 "$SCRIPT_DIR/model_manifest.py" verify \
  --model-dir "$MODEL_DIR" \
  --manifest "$PROJECT_ROOT/config/models.manifest.json"

"${SSH[@]}" "$TARGET_USER@$TARGET_HOST" \
  "sudo install -d -o '$TARGET_USER' -g streamglass -m 2775 '$REMOTE_MODEL_DIR'"

rsync -a --partial --progress \
  -e "$RSYNC_SSH" \
  "$MODEL_DIR/" "$TARGET_USER@$TARGET_HOST:$REMOTE_MODEL_DIR/"

"${SSH[@]}" "$TARGET_USER@$TARGET_HOST" \
  "python3 /home/$TARGET_USER/streamglass-deploy-staging/deploy/model_manifest.py verify --model-dir '$REMOTE_MODEL_DIR' --manifest /home/$TARGET_USER/streamglass-deploy-staging/config/models.manifest.json"

"${SSH[@]}" "$TARGET_USER@$TARGET_HOST" \
  "sudo chown -R root:streamglass '$REMOTE_MODEL_DIR' && sudo chmod -R g-w,o-rwx '$REMOTE_MODEL_DIR'"

echo "model artifacts verified and sealed at $REMOTE_MODEL_DIR"
