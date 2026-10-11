#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
onnx_dir="${STREAMGLASS_T2U_ONNX_DIR:-/opt/streamglass/onnx/s0/t2u}"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
build_mode="${STREAMGLASS_T2U_BUILD_MODE:-production}"

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$engine_dir"

build_engine() {
    local onnx_name="$1"
    local engine_name="$2"
    local min_shapes="$3"
    local opt_shapes="$4"
    local max_shapes="$5"
    local precision_mode="$6"
    local onnx_path="$onnx_dir/$onnx_name.onnx"
    local engine_path="$engine_dir/$engine_name.engine"
    local timing_cache="$engine_dir/$engine_name.timing.cache"
    local engine_tmp="$engine_path.building.$$"
    local timing_cache_tmp="$timing_cache.building.$$"
    local precision_args=()

    if [[ "$precision_mode" == "fp16" ]]; then
        precision_args=(--fp16)
    elif [[ "$precision_mode" == "strongly-typed" ]]; then
        precision_args=(--stronglyTyped)
    else
        echo "Unknown precision mode: $precision_mode" >&2
        return 1
    fi

    if [[ ! -s "$onnx_path" ]]; then
        echo "Missing T2U ONNX: $onnx_path" >&2
        return 1
    fi

    cleanup_engine() {
        rm -f "$engine_tmp" "$timing_cache_tmp" "$timing_cache_tmp.lock"
    }
    trap cleanup_engine RETURN

    if [[ -s "$timing_cache" ]]; then
        cp "$timing_cache" "$timing_cache_tmp"
    fi

    trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$engine_tmp" \
        --timingCacheFile="$timing_cache_tmp" \
        "${precision_args[@]}" \
        --minShapes="$min_shapes" \
        --optShapes="$opt_shapes" \
        --maxShapes="$max_shapes" \
        --memPoolSize=workspace:8192 \
        --builderOptimizationLevel=5 \
        --profilingVerbosity=detailed \
        --skipInference

    test -s "$engine_tmp"
    test -s "$timing_cache_tmp"
    mv -f "$engine_tmp" "$engine_path"
    mv -f "$timing_cache_tmp" "$timing_cache"
    trap - RETURN
    cleanup_engine
    echo "T2U TensorRT engine: $engine_path"
}

if [[ "$build_mode" == "production" || "$build_mode" == "all" ]]; then
    build_engine \
        t2u_frontend \
        t2u_frontend_fp16 \
        "text_decoder_output:1x2x1024,char_seqs:1x2,char_lens:1x2" \
        "text_decoder_output:1x32x1024,char_seqs:1x128,char_lens:1x32" \
        "text_decoder_output:1x128x1024,char_seqs:1x512,char_lens:1x128" \
        fp16

    build_engine \
        t2u_decoder \
        t2u_decoder_fp32 \
        "expanded_char_features:1x1x1024" \
        "expanded_char_features:1x256x1024" \
        "expanded_char_features:1x1024x1024" \
        strongly-typed
fi

if [[ "$build_mode" == "strong-frontend-probe" || "$build_mode" == "all" ]]; then
    build_engine \
        t2u_frontend \
        t2u_frontend_strongly_typed \
        "text_decoder_output:1x2x1024,char_seqs:1x2,char_lens:1x2" \
        "text_decoder_output:1x32x1024,char_seqs:1x128,char_lens:1x32" \
        "text_decoder_output:1x128x1024,char_seqs:1x512,char_lens:1x128" \
        strongly-typed
fi

if [[ "$build_mode" != "production" && "$build_mode" != "all" &&
      "$build_mode" != "strong-frontend-probe" ]]; then
    echo "Unknown STREAMGLASS_T2U_BUILD_MODE: $build_mode" >&2
    exit 2
fi
