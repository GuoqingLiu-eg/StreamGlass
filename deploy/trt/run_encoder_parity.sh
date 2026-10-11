#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
export_id="${STREAMGLASS_ENCODER_EXPORT_ID:?Set STREAMGLASS_ENCODER_EXPORT_ID}"
encoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/encoder"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
reference_dir="${STREAMGLASS_ENCODER_REFERENCES:-$encoder_root/exports/$export_id/references}"
report="${STREAMGLASS_ENCODER_PARITY_REPORT:-/opt/streamglass/golden/trt/encoder_parity.json}"

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$(dirname -- "$report")"

python3 "$script_dir/encoder_parity.py" \
    --engine "$engine_dir/encoder_cumulative_fp16.engine" \
    --reference-dir "$reference_dir" \
    --report "$report" \
    --min-cosine 0.999
