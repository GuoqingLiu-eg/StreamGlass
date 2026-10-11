#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
binary="${STREAMGLASS_ENGINE_INSPECT_BIN:-/opt/streamglass/tmp/native-install/$fingerprint/bin/streamglass-engine-inspect}"

exec "$binary" \
    "encoder=$engine_dir/encoder_cumulative_fp16.engine" \
    "decoder_prefill=$engine_dir/decoder_prefill_mixed.engine" \
    "decoder_step=$engine_dir/decoder_step_mixed.engine" \
    "t2u_frontend=$engine_dir/t2u_frontend_fp16.engine" \
    "t2u_decoder=$engine_dir/t2u_decoder_fp32.engine" \
    "vocoder=$engine_dir/vocoder_fp16.engine"
