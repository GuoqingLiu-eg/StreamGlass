#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
engine="${STREAMGLASS_ENCODER_ENGINE:-/opt/streamglass/engine-cache/$fingerprint/encoder_cumulative_fp16.engine}"
capture_id="${STREAMGLASS_ENCODER_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
reference="${STREAMGLASS_ENCODER_REFERENCE:-/opt/streamglass/onnx/s0/encoder/references/$capture_id/tensors/call_0006.npz}"
output_dir="${STREAMGLASS_ENCODER_NATIVE_PARITY_DIR:-/opt/streamglass/tmp/native-encoder-parity/$capture_id}"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"

mkdir -p "$output_dir"

"$install_root/bin/streamglass-encoder-replay" \
    "$project_root/tests/data/sample_spa.wav" \
    "$engine" \
    "$output_dir/encoder_output.f16"

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$reference:/reference.npz:ro" \
    --volume "$output_dir:/output" \
    "$image" \
    python /workspace/deploy/export/verify_native_encoder.py \
        --native-raw /output/encoder_output.f16 \
        --native-length /output/encoder_output.f16.len.i64 \
        --reference /reference.npz \
        --output /output/report.json \
        --minimum-cosine 0.999

echo "Native encoder parity report: $output_dir/report.json"
