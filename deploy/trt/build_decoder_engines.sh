#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
decoder_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/decoder"
export_id="${STREAMGLASS_DECODER_EXPORT_ID:?Set STREAMGLASS_DECODER_EXPORT_ID}"
engine_tag="${STREAMGLASS_DECODER_ENGINE_TAG:-mixed}"
build_components="${STREAMGLASS_DECODER_BUILD_COMPONENTS:-prefill,step}"
export_dir="$decoder_root/exports/$export_id"
manifest_path="$export_dir/export_manifest.json"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
report_dir="${STREAMGLASS_DECODER_BUILD_REPORT_DIR:-/opt/streamglass/golden/trt/decoder_build}"
cache_dir="${STREAMGLASS_CACHE_DIR:-/opt/streamglass/cache}"
tmp_dir="${STREAMGLASS_TMP_DIR:-/opt/streamglass/tmp}"

if [[ ! "$engine_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid STREAMGLASS_DECODER_ENGINE_TAG: $engine_tag" >&2
    exit 1
fi
if [[ ! "$build_components" =~ ^(prefill|step|prefill,step|step,prefill)$ ]]; then
    echo "Invalid STREAMGLASS_DECODER_BUILD_COMPONENTS: $build_components" >&2
    exit 1
fi

if [[ ! -s "$manifest_path" ]]; then
    echo "Missing decoder export manifest: $manifest_path" >&2
    exit 1
fi

python3 "$project_root/deploy/export/verify_decoder_export.py" \
    --manifest "$manifest_path"

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" \
    "$engine_dir" "$report_dir" "$cache_dir/cuda" "$cache_dir/xdg" "$tmp_dir"

export CUDA_CACHE_PATH="$cache_dir/cuda"
export XDG_CACHE_HOME="$cache_dir/xdg"
export TMPDIR="$tmp_dir"

build_engine() {
    local graph_name="$1"
    local onnx_relative="$2"
    local engine_name="$3"
    local min_shapes="$4"
    local opt_shapes="$5"
    local max_shapes="$6"
    local onnx_path="$export_dir/$onnx_relative"
    local engine_path="$engine_dir/$engine_name.engine"
    local timing_cache="$engine_dir/$engine_name.timing.cache"
    local build_log="$report_dir/$engine_name.log"
    local engine_tmp="$engine_path.building.$$"
    local timing_cache_tmp="$timing_cache.building.$$"
    local build_log_tmp="$build_log.building.$$"

    if [[ ! -s "$onnx_path" ]]; then
        echo "Missing decoder $graph_name ONNX: $onnx_path" >&2
        return 1
    fi

    cleanup_engine() {
        rm -f "$engine_tmp" "$timing_cache_tmp" "$timing_cache_tmp.lock" \
            "$build_log_tmp"
    }
    trap cleanup_engine RETURN

    if [[ -s "$timing_cache" ]]; then
        cp "$timing_cache" "$timing_cache_tmp"
    fi

    set +e
    trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$engine_tmp" \
        --timingCacheFile="$timing_cache_tmp" \
        --stronglyTyped \
        --minShapes="$min_shapes" \
        --optShapes="$opt_shapes" \
        --maxShapes="$max_shapes" \
        --memPoolSize=workspace:16384 \
        --builderOptimizationLevel=5 \
        --profilingVerbosity=detailed \
        --skipInference \
        2>&1 | tee "$build_log_tmp"
    local trtexec_status="${PIPESTATUS[0]}"
    set -e

    mv -f "$build_log_tmp" "$build_log"
    if [[ "$trtexec_status" -ne 0 ]]; then
        echo "Decoder $graph_name TensorRT build failed; see $build_log" >&2
        return "$trtexec_status"
    fi

    test -s "$engine_tmp"
    test -s "$timing_cache_tmp"
    mv -f "$engine_tmp" "$engine_path"
    mv -f "$timing_cache_tmp" "$timing_cache"
    sha256sum "$engine_path" > "$engine_path.sha256"
    trap - RETURN
    cleanup_engine
    echo "Decoder $graph_name TensorRT engine: $engine_path"
}

if [[ ",$build_components," == *,prefill,* ]]; then
    build_engine \
        prefill \
        prefill/decoder_prefill.onnx \
        "decoder_prefill_$engine_tag" \
        "target_input:1x2,encoder_output:1x1x1024,causal_mask:2x2" \
        "target_input:1x32,encoder_output:1x64x1024,causal_mask:32x32" \
        "target_input:1x512,encoder_output:1x256x1024,causal_mask:512x512"
fi

if [[ ",$build_components," == *,step,* ]]; then
    build_engine \
        step \
        step/decoder_step.onnx \
        "decoder_step_$engine_tag" \
        "target_input:1x1,encoder_output:1x1x1024,position:1,self_k_cache:24x1x16x2x64,self_v_cache:24x1x16x2x64" \
        "target_input:1x1,encoder_output:1x64x1024,position:1,self_k_cache:24x1x16x64x64,self_v_cache:24x1x16x64x64" \
        "target_input:1x1,encoder_output:1x256x1024,position:1,self_k_cache:24x1x16x512x64,self_v_cache:24x1x16x512x64"
fi

echo "Decoder TensorRT build logs: $report_dir"
