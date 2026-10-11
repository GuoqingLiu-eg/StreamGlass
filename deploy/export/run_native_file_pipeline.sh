#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache/$fingerprint}"
capture_id="${STREAMGLASS_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
encoder_capture="${STREAMGLASS_ENCODER_CAPTURE_DIR:-/opt/streamglass/onnx/s0/encoder/references/$capture_id/tensors}"
decoder_capture="${STREAMGLASS_DECODER_CAPTURE_DIR:-/opt/streamglass/onnx/s0/decoder/references/$capture_id/tensors}"
t2u_capture="${STREAMGLASS_T2U_CAPTURE_DIR:-/opt/streamglass/onnx/s0/t2u/references/$capture_id/tensors}"
runtime_assets="${STREAMGLASS_RUNTIME_ASSETS:-/opt/streamglass/tmp/runtime-assets/official-v1}"
output_dir="${STREAMGLASS_NATIVE_PIPELINE_DIR:-/opt/streamglass/tmp/native-file-pipeline/$capture_id}"
input_wav="${STREAMGLASS_INPUT_WAV:-$project_root/tests/data/sample_spa.wav}"
target_language="${STREAMGLASS_TARGET_LANGUAGE:-eng}"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"

mkdir -p "$output_dir"

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$encoder_capture:/encoder:ro" \
    --volume "$decoder_capture:/decoder:ro" \
    --volume "$t2u_capture:/t2u:ro" \
    --volume "$output_dir:/output" \
    "$image" \
    python /workspace/deploy/export/prepare_native_pipeline_fixture.py \
        --encoder-capture-dir /encoder \
        --decoder-capture-dir /decoder \
        --t2u-capture-dir /t2u \
        --output /output/pipeline_fixture.bin

"$install_root/bin/streamglass-native-file-replay" \
    "$input_wav" \
    "$engine_dir" \
    "$runtime_assets" \
    "$target_language" \
    "$output_dir/pipeline_fixture.bin" \
    "$output_dir/output.wav" \
    "$output_dir/report.json"

echo "Native file pipeline report: $output_dir/report.json"
echo "Native file pipeline audio:  $output_dir/output.wav"
