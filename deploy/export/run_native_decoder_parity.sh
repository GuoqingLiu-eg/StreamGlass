#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache/$fingerprint}"
capture_id="${STREAMGLASS_DECODER_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
capture_dir="${STREAMGLASS_DECODER_CAPTURE_DIR:-/opt/streamglass/onnx/s0/decoder/references/$capture_id/tensors}"
output_dir="${STREAMGLASS_DECODER_NATIVE_PARITY_DIR:-/opt/streamglass/tmp/native-decoder-parity/$capture_id}"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"

mkdir -p "$output_dir"

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$capture_dir:/capture:ro" \
    --volume "$output_dir:/output" \
    "$image" \
    python /workspace/deploy/export/prepare_native_decoder_fixture.py \
        --capture-dir /capture \
        --output /output/decoder_fixture.bin

"$install_root/bin/streamglass-decoder-replay" \
    "$engine_dir/decoder_prefill_mixed.engine" \
    "$engine_dir/decoder_step_mixed.engine" \
    "$output_dir/decoder_fixture.bin" \
    "$output_dir/report.json"

echo "Native decoder parity report: $output_dir/report.json"
