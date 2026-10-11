#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
experiment_id="silero-vad-v6.2.1"
source_dir="$project_root/third_party/$experiment_id/data"
onnx_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}"
engine_root="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}"
golden_root="${STREAMGLASS_GOLDEN_DIR:-/opt/streamglass/golden}"
onnx_dir="$onnx_root/experiments/$experiment_id"
engine_dir="$engine_root/experiments/$experiment_id"
report_dir="$golden_root/experiments/$experiment_id"
container_image="${STREAMGLASS_SILERO_EXPORT_IMAGE:-nvcr.io/nvidia/pytorch:24.01-py3}"
trtexec_bin="${STREAMGLASS_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
jit_path="$source_dir/silero_vad.jit"
onnx_path="$onnx_dir/silero_vad_16k_static.onnx"
metadata_path="$report_dir/export.json"
engine_path="$engine_dir/silero_vad_16k_fp16.engine"
timing_cache="$engine_dir/silero_vad_16k_fp16.timing.cache"
build_log="$report_dir/build_fp16.log"
fixture_wav="${STREAMGLASS_SILERO_FIXTURE_WAV:-}"
fixture_path="$report_dir/silero_vad_reference.fixture"

for output_dir in "$onnx_dir" "$engine_dir" "$report_dir"; do
    case "$output_dir" in
        /opt/streamglass/*/experiments/silero-vad-v6.2.1) ;;
        *)
        if [[ "${STREAMGLASS_ALLOW_NONDEFAULT_EXPERIMENT_ROOT:-0}" != "1" ]]; then
            echo "Refusing non-isolated experiment path: $output_dir" >&2
            exit 2
        fi
            ;;
    esac
done

if [[ ! -s "$jit_path" ]]; then
    echo "Missing official Silero JIT model: $jit_path" >&2
    exit 1
fi
if [[ ! -x "$trtexec_bin" ]]; then
    echo "TensorRT trtexec not found: $trtexec_bin" >&2
    exit 1
fi

sudo install -d -m 2775 -o "$(id -un)" -g "$(id -gn)" \
    "$onnx_dir" "$engine_dir" "$report_dir"
install -m 0644 "$source_dir/../LICENSE" "$onnx_dir/LICENSE"
install -m 0644 "$jit_path" "$onnx_dir/silero_vad.jit"

docker_args=(
    --rm
    --network none
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -v "$project_root:/workspace:ro"
    -v "$onnx_dir:/onnx-output"
    -v "$report_dir:/report-output"
)
export_args=(
    python3 /workspace/deploy/trt/export_silero_vad_16k.py
    --jit "/workspace/third_party/$experiment_id/data/silero_vad.jit"
    --onnx /onnx-output/silero_vad_16k_static.onnx
    --metadata /report-output/export.json
)
if [[ -n "$fixture_wav" ]]; then
    if [[ ! -s "$fixture_wav" ]]; then
        echo "Fixture WAV not found: $fixture_wav" >&2
        exit 1
    fi
    docker_args+=(-v "$fixture_wav:/fixture.wav:ro")
    export_args+=(
        --fixture-wav /fixture.wav
        --fixture-output /report-output/silero_vad_reference.fixture
    )
fi
sudo docker run "${docker_args[@]}" "$container_image" "${export_args[@]}"

engine_tmp="$engine_path.building.$$"
cache_tmp="$timing_cache.building.$$"
log_tmp="$build_log.building.$$"
cleanup() {
    rm -f "$engine_tmp" "$cache_tmp" "$cache_tmp.lock" "$log_tmp"
}
trap cleanup EXIT
if [[ -s "$timing_cache" ]]; then
    cp "$timing_cache" "$cache_tmp"
fi

set +e
"$trtexec_bin" \
    --onnx="$onnx_path" \
    --saveEngine="$engine_tmp" \
    --timingCacheFile="$cache_tmp" \
    --fp16 \
    --memPoolSize=workspace:2048 \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --skipInference \
    2>&1 | tee "$log_tmp"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" -ne 0 ]]; then
    mv -f "$log_tmp" "$build_log"
    echo "Silero TensorRT build failed; see $build_log" >&2
    exit "$status"
fi
test -s "$engine_tmp"
test -s "$cache_tmp"
mv -f "$engine_tmp" "$engine_path"
mv -f "$cache_tmp" "$timing_cache"
mv -f "$log_tmp" "$build_log"
sha256sum "$engine_path" > "$engine_path.sha256"
trap - EXIT
cleanup

echo "Silero ONNX: $onnx_path"
echo "Silero TensorRT engine: $engine_path"
echo "Export metadata: $metadata_path"
if [[ -n "$fixture_wav" ]]; then
    echo "Parity fixture: $fixture_path"
fi
echo "Build log: $build_log"
