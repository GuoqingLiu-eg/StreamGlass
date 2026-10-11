#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_path="${STREAMGLASS_VOCODER_ENGINE:-/opt/streamglass/engine-cache/$fingerprint/vocoder_fp16.engine}"
reference_dir="${STREAMGLASS_VOCODER_REFERENCES:-/opt/streamglass/onnx/s0/vocoder/references}"
report="${STREAMGLASS_VOCODER_PARITY_REPORT:-/opt/streamglass/golden/trt/vocoder_parity.json}"

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$(dirname -- "$report")"

python3 "$script_dir/vocoder_parity.py" \
    --engine "$engine_path" \
    --reference-dir "$reference_dir" \
    --report "$report" \
    --min-cosine 0.999
