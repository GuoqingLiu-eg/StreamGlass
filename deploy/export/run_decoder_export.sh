#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
model_dir="${STREAMGLASS_MODEL_DIR:-/opt/streamglass/models}"
decoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/decoder"
capture_id="${STREAMGLASS_DECODER_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
export_id="${STREAMGLASS_DECODER_EXPORT_ID:-${capture_id}-explicit-kv-fp16}"
capture_dir="$decoder_root/references/$capture_id/tensors"
output_dir="$decoder_root/exports/$export_id"

if [[ ! -s "$capture_dir/manifest.json" ]]; then
    echo "Missing official decoder capture: $capture_dir/manifest.json" >&2
    exit 1
fi

sudo install -d -m 0775 "$output_dir"

extra_args=()
if [[ "${STREAMGLASS_VALIDATE_ONLY:-0}" == "1" ]]; then
    extra_args+=(--validate-only)
fi

sudo docker run --rm \
    --runtime nvidia \
    --network none \
    --ipc host \
    --shm-size 16g \
    --volume "$model_dir:/models:ro" \
    --volume "$project_root:/workspace:ro" \
    --volume "$decoder_root:/decoder" \
    --env FAIRSEQ2_USER_ASSET_DIR=/workspace/config/assets \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$image" \
    python /workspace/deploy/export/export_decoder.py \
        --capture-dir "/decoder/references/$capture_id/tensors" \
        --output-dir "/decoder/exports/$export_id" \
        --dtype fp16 \
        "${extra_args[@]}"

echo "Decoder export: $output_dir"
