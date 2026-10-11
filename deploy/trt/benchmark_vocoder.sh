#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
engine_path="${STREAMGLASS_VOCODER_ENGINE:-/opt/streamglass/engine-cache/$fingerprint/vocoder_fp16.engine}"
report_dir="${STREAMGLASS_VOCODER_BENCHMARK_DIR:-/opt/streamglass/golden/trt/vocoder_benchmark}"
duration="${STREAMGLASS_BENCHMARK_DURATION:-5}"

if [[ ! -s "$engine_path" ]]; then
    echo "Missing vocoder engine: $engine_path" >&2
    exit 1
fi

sudo install -d -m 0775 -o "$(id -un)" -g "$(id -gn)" "$report_dir"

for unit_length in 1 8 64 256; do
    for mode in compute end_to_end; do
        report="$report_dir/${mode}_len$(printf '%04d' "$unit_length").json"
        temporary_report="$report.building.$$"
        transfer_args=()
        if [[ "$mode" == "compute" ]]; then
            transfer_args+=(--noDataTransfers)
        fi

        cleanup() {
            rm -f "$temporary_report"
        }
        trap cleanup EXIT

        trtexec \
            --loadEngine="$engine_path" \
            --shapes="unit_ids:1x${unit_length},lang_ids:1x1,speaker_ids:1x1" \
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
        trap - EXIT
    done
done

python3 "$script_dir/summarize_trtexec.py" \
    --input-dir "$report_dir" \
    --output "$report_dir/summary.json"

echo "Vocoder benchmark reports: $report_dir"
