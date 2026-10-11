#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
model_dir="${STREAMGLASS_MODEL_DIR:-/opt/streamglass/models}"
output_dir="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/t2u"
capture_id="${STREAMGLASS_T2U_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
decoder_precision="${STREAMGLASS_T2U_DECODER_PRECISION:-fp32}"
capture_dir="$output_dir/references/$capture_id/tensors"

if [[ ! -s "$capture_dir/manifest.json" ]]; then
    echo "Missing official T2U capture: $capture_dir/manifest.json" >&2
    exit 1
fi

sudo install -d -m 0775 "$output_dir"

sudo docker run --rm \
    --runtime nvidia \
    --network none \
    --ipc host \
    --shm-size 8g \
    --volume "$model_dir:/models:ro" \
    --volume "$project_root:/workspace:ro" \
    --volume "$output_dir:/output" \
    --env FAIRSEQ2_USER_ASSET_DIR=/workspace/config/assets \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$image" \
    python /workspace/deploy/export/export_t2u.py \
        --capture-dir "/output/references/$capture_id/tensors" \
        --output-dir /output \
        --dtype fp16 \
        --decoder-precision "$decoder_precision"

echo "T2U ONNX export: $output_dir"
