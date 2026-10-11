#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
encoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/encoder"
export_id="${STREAMGLASS_ENCODER_EXPORT_ID:?Set STREAMGLASS_ENCODER_EXPORT_ID}"
onnx_dir="$encoder_root/exports/$export_id"
onnx_name="${STREAMGLASS_ENCODER_ONNX_NAME:-encoder_trt.onnx}"
manifest_name="${STREAMGLASS_ENCODER_MANIFEST_NAME:-encoder_trt_manifest.json}"
onnx_path="$onnx_dir/$onnx_name"
manifest_path="$onnx_dir/$manifest_name"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
report_dir="${STREAMGLASS_ENCODER_BUILD_REPORT_DIR:-/opt/streamglass/golden/trt/encoder_build}"
cache_dir="${STREAMGLASS_CACHE_DIR:-/opt/streamglass/cache}"
tmp_dir="${STREAMGLASS_TMP_DIR:-/opt/streamglass/tmp}"
build_mode="${STREAMGLASS_ENCODER_BUILD_MODE:-production}"
precision_args=()
if [[ "$build_mode" == "production" ]]; then
    engine_name="encoder_cumulative_fp16"
    precision_args=(--fp16)
elif [[ "$build_mode" == "strong-probe" ]]; then
    engine_name="encoder_cumulative_strongly_typed"
    precision_args=(--stronglyTyped)
else
    echo "Unknown STREAMGLASS_ENCODER_BUILD_MODE: $build_mode" >&2
    exit 2
fi
engine_path="$engine_dir/$engine_name.engine"
timing_cache="$engine_dir/$engine_name.timing.cache"
engine_tmp="$engine_path.building.$$"
timing_cache_tmp="$timing_cache.building.$$"
build_log="$report_dir/$engine_name.log"
build_log_tmp="$build_log.building.$$"

if [[ ! -s "$onnx_path" || ! -s "$manifest_path" ]]; then
    echo "Missing encoder export or manifest below $onnx_dir" >&2
    exit 1
fi

python3 "$project_root/deploy/export/verify_onnx_artifacts.py" \
    --manifest "$manifest_path"

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" \
    "$engine_dir" "$report_dir" "$cache_dir/cuda" "$cache_dir/xdg" "$tmp_dir"

export CUDA_CACHE_PATH="$cache_dir/cuda"
export XDG_CACHE_HOME="$cache_dir/xdg"
export TMPDIR="$tmp_dir"

cleanup_build() {
    rm -f "$engine_tmp" "$timing_cache_tmp" "$timing_cache_tmp.lock"
}
trap cleanup_build EXIT

if [[ -s "$timing_cache" ]]; then
    cp "$timing_cache" "$timing_cache_tmp"
fi

set +e
trtexec \
    --onnx="$onnx_path" \
    --saveEngine="$engine_tmp" \
    --timingCacheFile="$timing_cache_tmp" \
    "${precision_args[@]}" \
    --minShapes="fbank:1x222x80,fbank_seq_lens:1" \
    --optShapes="fbank:1x600x80,fbank_seq_lens:1" \
    --maxShapes="fbank:1x2000x80,fbank_seq_lens:1" \
    --memPoolSize=workspace:16384 \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --skipInference \
    2>&1 | tee "$build_log_tmp"
trtexec_status="${PIPESTATUS[0]}"
set -e

mv -f "$build_log_tmp" "$build_log"
if [[ "$trtexec_status" -ne 0 ]]; then
    echo "Encoder TensorRT build failed; see $build_log" >&2
    exit "$trtexec_status"
fi

test -s "$engine_tmp"
test -s "$timing_cache_tmp"
mv -f "$engine_tmp" "$engine_path"
mv -f "$timing_cache_tmp" "$timing_cache"
trap - EXIT
cleanup_build

sha256sum "$engine_path" > "$engine_path.sha256"
echo "Encoder TensorRT engine: $engine_path"
echo "Full build log: $build_log"
