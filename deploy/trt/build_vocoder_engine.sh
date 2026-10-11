#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
onnx_path="${STREAMGLASS_VOCODER_ONNX:-/opt/streamglass/onnx/s0/vocoder/vocoder.onnx}"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
engine_path="$engine_dir/vocoder_fp16.engine"
timing_cache="$engine_dir/vocoder_fp16.timing.cache"
engine_tmp="$engine_path.building.$$"
timing_cache_tmp="$timing_cache.building.$$"

if [[ ! -f "$onnx_path" ]]; then
    echo "Missing vocoder ONNX: $onnx_path" >&2
    exit 1
fi

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$engine_dir"

cleanup() {
    rm -f "$engine_tmp" "$timing_cache_tmp" "$timing_cache_tmp.lock"
}
trap cleanup EXIT

if [[ -s "$timing_cache" ]]; then
    cp "$timing_cache" "$timing_cache_tmp"
fi

trtexec \
    --onnx="$onnx_path" \
    --saveEngine="$engine_tmp" \
    --timingCacheFile="$timing_cache_tmp" \
    --fp16 \
    --minShapes=unit_ids:1x1,lang_ids:1x1,speaker_ids:1x1 \
    --optShapes=unit_ids:1x64,lang_ids:1x1,speaker_ids:1x1 \
    --maxShapes=unit_ids:1x256,lang_ids:1x1,speaker_ids:1x1 \
    --memPoolSize=workspace:8192 \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --skipInference

test -s "$engine_tmp"
test -s "$timing_cache_tmp"
mv -f "$engine_tmp" "$engine_path"
mv -f "$timing_cache_tmp" "$timing_cache"

echo "Vocoder TensorRT engine: $engine_path"
