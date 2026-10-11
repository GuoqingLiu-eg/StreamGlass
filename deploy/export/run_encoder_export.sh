#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
model_dir="${STREAMGLASS_MODEL_DIR:-/opt/streamglass/models}"
encoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/encoder"
capture_id="${STREAMGLASS_ENCODER_CAPTURE_ID:?Set STREAMGLASS_ENCODER_CAPTURE_ID to an official capture ID}"
sdpa_mode="${STREAMGLASS_ENCODER_SDPA_MODE:-official-naive-export}"
export_id="${STREAMGLASS_ENCODER_EXPORT_ID:-${capture_id}-${sdpa_mode}-fp16}"
capture_dir="$encoder_root/references/$capture_id/tensors"
output_dir="$encoder_root/exports/$export_id"

if [[ ! -s "$capture_dir/manifest.json" ]]; then
    echo "Missing official encoder capture: $capture_dir/manifest.json" >&2
    exit 1
fi

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$output_dir"

sudo docker run --rm \
    --runtime nvidia \
    --network none \
    --ipc host \
    --shm-size 16g \
    --volume "$model_dir:/models:ro" \
    --volume "$project_root:/workspace:ro" \
    --volume "$encoder_root:/encoder" \
    --env FAIRSEQ2_USER_ASSET_DIR=/workspace/config/assets \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$image" \
    python /workspace/deploy/export/export_encoder.py \
        --capture-dir "/encoder/references/$capture_id/tensors" \
        --output-dir "/encoder/exports/$export_id" \
        --device cuda \
        --dtype fp16 \
        --opset 17 \
        --sdpa-mode "$sdpa_mode"

echo "Encoder ONNX export: $output_dir"
