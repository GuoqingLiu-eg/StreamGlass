#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
encoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/encoder"
export_id="${STREAMGLASS_ENCODER_EXPORT_ID:?Set STREAMGLASS_ENCODER_EXPORT_ID}"
output_dir="$encoder_root/exports/$export_id"

if [[ ! -s "$output_dir/encoder.onnx" || ! -s "$output_dir/export_manifest.json" ]]; then
    echo "Missing encoder export below $output_dir" >&2
    exit 1
fi

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$encoder_root:/encoder" \
    "$image" \
    python /workspace/deploy/export/index_onnx_artifacts.py \
        --onnx "/encoder/exports/$export_id/encoder.onnx" \
        --manifest "/encoder/exports/$export_id/export_manifest.json"

python3 "$script_dir/verify_onnx_artifacts.py" \
    --manifest "$output_dir/export_manifest.json"
