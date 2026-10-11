#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
experiment_id="silero-vad-v6.2.1"
engine_root="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}"
source_engine="${STREAMGLASS_SILERO_SOURCE_ENGINE:-$engine_root/experiments/$experiment_id/silero_vad_16k_fp16.engine}"
source_checksum="$source_engine.sha256"
production_dir="$engine_root/$fingerprint"
production_engine="$production_dir/silero_vad_16k_fp16.engine"

[[ -s "$source_engine" ]] || {
    echo "Missing verified Silero experiment engine: $source_engine" >&2
    exit 1
}

actual_checksum=$(sha256sum "$source_engine" | awk '{print $1}')
if [[ -s "$source_checksum" ]]; then
    expected_checksum=$(awk 'NR == 1 {print $1}' "$source_checksum")
    [[ "$actual_checksum" == "$expected_checksum" ]] || {
        echo "Silero engine checksum mismatch" >&2
        exit 1
    }
fi

sudo install -d -m 2775 -o "$(id -un)" -g "$(id -gn)" "$production_dir"
temporary="$production_engine.installing.$$"
cleanup() {
    rm -f "$temporary"
}
trap cleanup EXIT
install -m 0644 "$source_engine" "$temporary"
mv -f "$temporary" "$production_engine"
printf '%s  %s\n' "$actual_checksum" "$production_engine" \
    >"$production_engine.sha256"
trap - EXIT

echo "Promoted Silero TensorRT engine: $production_engine"
echo "SHA256: $actual_checksum"
