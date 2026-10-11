#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache/$fingerprint}"
reference_dir="${STREAMGLASS_VOCODER_REFERENCE_DIR:-/opt/streamglass/onnx/s0/vocoder/references}"
output_dir="${STREAMGLASS_VOCODER_NATIVE_PARITY_DIR:-/opt/streamglass/tmp/native-vocoder-parity}"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"

mkdir -p "$output_dir"

sudo docker run --rm \
    --network none \
    --volume "$project_root:/workspace:ro" \
    --volume "$reference_dir:/references:ro" \
    --volume "$output_dir:/output" \
    "$image" \
    python /workspace/deploy/export/prepare_native_vocoder_fixture.py \
        --reference-dir /references \
        --output /output/vocoder_fixture.bin

"$install_root/bin/streamglass-vocoder-replay" \
    "$engine_dir/vocoder_fp16.engine" \
    "$output_dir/vocoder_fixture.bin" \
    "$output_dir/report.json"

echo "Native vocoder parity report: $output_dir/report.json"
