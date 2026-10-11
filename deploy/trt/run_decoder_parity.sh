#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_tag="${STREAMGLASS_DECODER_ENGINE_TAG:-mixed}"
capture_id="${STREAMGLASS_DECODER_CAPTURE_ID:-official-s0-320ms-20260720-r1}"
decoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/decoder"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
reference_dir="${STREAMGLASS_DECODER_REFERENCES:-$decoder_root/references/$capture_id/tensors}"
report="${STREAMGLASS_DECODER_PARITY_REPORT:-/opt/streamglass/golden/trt/decoder_parity.json}"
prefill_engine="${STREAMGLASS_DECODER_PREFILL_ENGINE:-$engine_dir/decoder_prefill_$engine_tag.engine}"
step_engine="${STREAMGLASS_DECODER_STEP_ENGINE:-$engine_dir/decoder_step_$engine_tag.engine}"

if [[ ! "$engine_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid STREAMGLASS_DECODER_ENGINE_TAG: $engine_tag" >&2
    exit 1
fi

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$(dirname -- "$report")"

python3 "$script_dir/decoder_parity.py" \
    --prefill-engine "$prefill_engine" \
    --step-engine "$step_engine" \
    --reference-dir "$reference_dir" \
    --report "$report" \
    --min-cosine 0.999 \
    --max-decision-error 0.02
