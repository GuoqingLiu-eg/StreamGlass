#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache/$fingerprint}"
capture_id="${STREAMGLASS_T2U_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
capture_dir="${STREAMGLASS_T2U_CAPTURE_DIR:-/opt/streamglass/onnx/s0/t2u/references/$capture_id/tensors}"
split_reference_dir="${STREAMGLASS_T2U_SPLIT_REFERENCE_DIR:-/opt/streamglass/onnx/s0/t2u/split_references}"
runtime_assets="${STREAMGLASS_RUNTIME_ASSETS:-/opt/streamglass/tmp/runtime-assets/official-v1}"
output_dir="${STREAMGLASS_T2U_NATIVE_PARITY_DIR:-/opt/streamglass/tmp/native-t2u-parity/$capture_id}"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"

mkdir -p "$output_dir"

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$capture_dir:/capture:ro" \
    --volume "$split_reference_dir:/split:ro" \
    --volume "$output_dir:/output" \
    "$image" \
    python /workspace/deploy/export/prepare_native_t2u_fixture.py \
        --capture-dir /capture \
        --split-reference-dir /split \
        --output /output/t2u_fixture.bin

"$install_root/bin/streamglass-t2u-replay" \
    "$engine_dir/t2u_frontend_fp16.engine" \
    "$engine_dir/t2u_decoder_fp32.engine" \
    "$runtime_assets" \
    "$output_dir/t2u_fixture.bin" \
    "$output_dir/report.json"

echo "Native T2U parity report: $output_dir/report.json"
