#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
encoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/encoder"
export_id="${STREAMGLASS_ENCODER_EXPORT_ID:?Set STREAMGLASS_ENCODER_EXPORT_ID}"
output_dir="$encoder_root/exports/$export_id"
expected_attention_casts="${STREAMGLASS_ENCODER_EXPECTED_ATTENTION_CASTS:-25}"

if [[ ! -s "$output_dir/encoder.onnx" ]]; then
    echo "Missing encoder export: $output_dir/encoder.onnx" >&2
    exit 1
fi

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$encoder_root:/encoder" \
    "$image" \
    python /workspace/deploy/export/normalize_encoder_onnx_for_trt.py \
        --source "/encoder/exports/$export_id/encoder.onnx" \
        --output "/encoder/exports/$export_id/encoder_trt.onnx" \
        --manifest "/encoder/exports/$export_id/encoder_trt_manifest.json" \
        --expected-padding-casts 24 \
        --expected-attention-casts "$expected_attention_casts" \
        --expected-shaw-slice-bypasses 24 \
        --expected-shaw-dynamic-restorations 24 \
        --expected-traced-attention-length 198

python3 "$script_dir/verify_onnx_artifacts.py" \
    --manifest "$output_dir/encoder_trt_manifest.json"
