#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}" 
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache/$fingerprint}"
runtime_assets="${STREAMGLASS_RUNTIME_ASSETS:-/opt/streamglass/tmp/runtime-assets/official-v1}"
input_wav="${STREAMGLASS_INPUT_WAV:-$project_root/tests/data/sample_spa.wav}"
target_language="${STREAMGLASS_TARGET_LANGUAGE:-eng}"

exec "$install_root/bin/streamglass-native-threaded-file" \
    "$input_wav" \
    "$engine_dir" \
    "$runtime_assets" \
    "$target_language" \
    "$@"
