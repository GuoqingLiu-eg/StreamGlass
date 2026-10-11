#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_dir="${STREAMGLASS_ENGINE_DIR:-/opt/streamglass/engine-cache}/$fingerprint"
frontend_engine="${STREAMGLASS_T2U_FRONTEND_ENGINE:-$engine_dir/t2u_frontend_fp16.engine}"
decoder_engine="${STREAMGLASS_T2U_DECODER_ENGINE:-$engine_dir/t2u_decoder_fp32.engine}"
report_dir="${STREAMGLASS_T2U_BENCHMARK_DIR:-/opt/streamglass/golden/trt/t2u_benchmark}"
duration="${STREAMGLASS_BENCHMARK_DURATION:-5}"
budget_ms="${STREAMGLASS_T2U_BUDGET_MS:-200}"

for engine in "$frontend_engine" "$decoder_engine"; do
    if [[ ! -s "$engine" ]]; then
        echo "Missing T2U engine: $engine" >&2
        exit 1
    fi
done

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$report_dir"

run_case() {
    local module="$1"
    local mode="$2"
    local engine="$3"
    local shapes="$4"
    local case_name="$5"
    local report="$report_dir/${module}_${mode}_${case_name}.json"
    local temporary_report="$report.building.$$"
    local transfer_args=()

    if [[ "$mode" == "compute" ]]; then
        transfer_args+=(--noDataTransfers)
    fi

    cleanup_case() {
        rm -f "$temporary_report"
    }
    trap cleanup_case RETURN

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
    trap - RETURN
    cleanup_case
}

for mode in compute end_to_end; do
    run_case frontend "$mode" "$frontend_engine" \
        "text_decoder_output:1x7x1024,char_seqs:1x26,char_lens:1x7" \
        s0007_c0026
    run_case frontend "$mode" "$frontend_engine" \
        "text_decoder_output:1x14x1024,char_seqs:1x54,char_lens:1x14" \
        s0014_c0054
    run_case frontend "$mode" "$frontend_engine" \
        "text_decoder_output:1x32x1024,char_seqs:1x128,char_lens:1x32" \
        s0032_c0128

    run_case decoder "$mode" "$decoder_engine" \
        "expanded_char_features:1x57x1024" u0057
    run_case decoder "$mode" "$decoder_engine" \
        "expanded_char_features:1x131x1024" u0131
    run_case decoder "$mode" "$decoder_engine" \
        "expanded_char_features:1x256x1024" u0256
done

python3 "$script_dir/summarize_t2u_trtexec.py" \
    --input-dir "$report_dir" \
    --output "$report_dir/summary.json" \
    --frontend-engine "$frontend_engine" \
    --decoder-engine "$decoder_engine" \
    --budget-ms "$budget_ms"

echo "T2U benchmark reports: $report_dir"
