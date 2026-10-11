#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_tag="${STREAMGLASS_DECODER_ENGINE_TAG:-mixed}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
prefill_engine="${STREAMGLASS_DECODER_PREFILL_ENGINE:-$engine_dir/decoder_prefill_$engine_tag.engine}"
step_engine="${STREAMGLASS_DECODER_STEP_ENGINE:-$engine_dir/decoder_step_$engine_tag.engine}"
report_dir="${STREAMGLASS_DECODER_BENCHMARK_DIR:-/opt/streamglass/golden/trt/decoder_benchmark}"
duration="${STREAMGLASS_BENCHMARK_DURATION:-3}"
budget_ms="${STREAMGLASS_DECODER_BUDGET_MS:-200}"

for engine in "$prefill_engine" "$step_engine"; do
    if [[ ! -s "$engine" ]]; then
        echo "Missing decoder engine: $engine" >&2
        exit 1
    fi
done
sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$report_dir"

run_case() {
    local mode="$1"
    local component="$2"
    local case_name="$3"
    local shapes="$4"
    local engine="$5"
    local report="$report_dir/${mode}_${component}_${case_name}.json"
    local temporary_report="$report.building.$$"
    local transfer_args=()

    if [[ "$mode" == "compute" ]]; then
        transfer_args+=(--noDataTransfers)
    fi
    rm -f "$temporary_report"
    trtexec \
        --loadEngine="$engine" \
        --shapes="$shapes" \
        --warmUp=1000 \
        --duration="$duration" \
        --iterations=100 \
        --avgRuns=100 \
        --percentile=50,90,95,99 \
        --useCudaGraph \
        --useSpinWait \
        --exportTimes="$temporary_report" \
        "${transfer_args[@]}"
    test -s "$temporary_report"
    mv -f "$temporary_report" "$report"
}

for mode in compute end_to_end; do
    run_case "$mode" prefill t0002_s0014 \
        "target_input:1x2,encoder_output:1x14x1024,causal_mask:2x2" \
        "$prefill_engine"
    run_case "$mode" prefill t0006_s0025 \
        "target_input:1x6,encoder_output:1x25x1024,causal_mask:6x6" \
        "$prefill_engine"
    run_case "$mode" prefill t0032_s0064 \
        "target_input:1x32,encoder_output:1x64x1024,causal_mask:32x32" \
        "$prefill_engine"
    run_case "$mode" prefill t0128_s0128 \
        "target_input:1x128,encoder_output:1x128x1024,causal_mask:128x128" \
        "$prefill_engine"
    run_case "$mode" prefill t0512_s0256 \
        "target_input:1x512,encoder_output:1x256x1024,causal_mask:512x512" \
        "$prefill_engine"

    run_case "$mode" step c0002_s0014 \
        "target_input:1x1,encoder_output:1x14x1024,position:1,self_k_cache:24x1x16x2x64,self_v_cache:24x1x16x2x64" \
        "$step_engine"
    run_case "$mode" step c0006_s0025 \
        "target_input:1x1,encoder_output:1x25x1024,position:1,self_k_cache:24x1x16x6x64,self_v_cache:24x1x16x6x64" \
        "$step_engine"
    run_case "$mode" step c0032_s0064 \
        "target_input:1x1,encoder_output:1x64x1024,position:1,self_k_cache:24x1x16x32x64,self_v_cache:24x1x16x32x64" \
        "$step_engine"
    run_case "$mode" step c0128_s0128 \
        "target_input:1x1,encoder_output:1x128x1024,position:1,self_k_cache:24x1x16x128x64,self_v_cache:24x1x16x128x64" \
        "$step_engine"
    run_case "$mode" step c0512_s0256 \
        "target_input:1x1,encoder_output:1x256x1024,position:1,self_k_cache:24x1x16x512x64,self_v_cache:24x1x16x512x64" \
        "$step_engine"
done

python3 "$script_dir/summarize_decoder_trtexec.py" \
    --input-dir "$report_dir" \
    --output "$report_dir/summary.json" \
    --budget-ms "$budget_ms"

echo "Decoder benchmark reports: $report_dir"
