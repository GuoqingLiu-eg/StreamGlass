#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
model_dir="${STREAMGLASS_MODEL_DIR:-/opt/streamglass/models}"
capture_id="${STREAMGLASS_T2U_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
capture_dir="${STREAMGLASS_T2U_CAPTURE_DIR:-/opt/streamglass/onnx/s0/t2u/references/$capture_id/tensors}"
output_dir="${STREAMGLASS_RUNTIME_ASSET_DIR:-/opt/streamglass/tmp/runtime-assets/official-v1}"

mkdir -p "$output_dir"

sudo docker run --rm \
    --runtime nvidia \
    --network none \
    --ipc host \
    --shm-size 4g \
    --volume "$model_dir:/models:ro" \
    --volume "$project_root:/workspace:ro" \
    --volume "$capture_dir:/capture:ro" \
    --volume "$output_dir:/output" \
    --env FAIRSEQ2_USER_ASSET_DIR=/workspace/config/assets \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$image" \
    python /workspace/deploy/export/export_runtime_assets.py \
        --capture-dir /capture \
        --output-dir /output

echo "C++ runtime tokenizer assets: $output_dir"

