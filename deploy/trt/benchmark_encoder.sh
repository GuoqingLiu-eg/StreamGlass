#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
engine="${STREAMGLASS_ENCODER_ENGINE:-$engine_dir/encoder_cumulative_fp16.engine}"
report_dir="${STREAMGLASS_ENCODER_BENCHMARK_DIR:-/opt/streamglass/golden/trt/encoder_benchmark}"
duration="${STREAMGLASS_BENCHMARK_DURATION:-5}"
budget_ms="${STREAMGLASS_ENCODER_BUDGET_MS:-200}"

if [[ ! -s "$engine" ]]; then
    echo "Missing encoder engine: $engine" >&2
    exit 1
fi
sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$report_dir"

run_case() {
    local mode="$1"
    local fbank_length="$2"
    local case_name
    local report
    local temporary_report
    local transfer_args=()
    printf -v case_name "f%04d" "$fbank_length"
    report="$report_dir/${mode}_${case_name}.json"
    temporary_report="$report.building.$$"

    if [[ "$mode" == "compute" ]]; then
        transfer_args+=(--noDataTransfers)
    fi
    rm -f "$temporary_report"
    trtexec \
        --loadEngine="$engine" \
        --shapes="fbank:1x${fbank_length}x80,fbank_seq_lens:1" \
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
    for fbank_length in 222 396 600 2000; do
        run_case "$mode" "$fbank_length"
    done
done

python3 "$script_dir/summarize_encoder_trtexec.py" \
    --input-dir "$report_dir" \
    --output "$report_dir/summary.json" \
    --budget-ms "$budget_ms"

echo "Encoder benchmark reports: $report_dir"
