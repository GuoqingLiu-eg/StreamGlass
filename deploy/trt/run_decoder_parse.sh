#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export_id="${STREAMGLASS_DECODER_EXPORT_ID:-official-s0-320ms-20260720-r1-explicit-kv-fp16}"
export_dir="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/decoder/exports/$export_id"
manifest="$export_dir/export_manifest.json"
report="${STREAMGLASS_DECODER_PARSE_REPORT:-/opt/streamglass/golden/trt/decoder_parse.json}"

if [[ ! -s "$manifest" ]]; then
    echo "Missing decoder export manifest: $manifest" >&2
    exit 1
fi

python3 "$script_dir/parse_decoder_onnx.py" \
    --manifest "$manifest" \
    --report "$report"
