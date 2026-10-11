#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
reference_dir="${STREAMGLASS_T2U_REFERENCES:-/opt/streamglass/onnx/s0/t2u/split_references}"
report="${STREAMGLASS_T2U_PARITY_REPORT:-/opt/streamglass/golden/trt/t2u_parity.json}"

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$(dirname -- "$report")"

python3 "$script_dir/t2u_parity.py" \
    --frontend-engine "$engine_dir/t2u_frontend_fp16.engine" \
    --decoder-engine "$engine_dir/t2u_decoder_fp32.engine" \
    --reference-dir "$reference_dir" \
    --report "$report" \
    --min-cosine 0.999
