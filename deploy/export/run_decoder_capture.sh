#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
model_dir="${STREAMGLASS_MODEL_DIR:-/opt/streamglass/models}"
output_root="${STREAMGLASS_ONNX_DIR:-/opt/streamglass/onnx}/s0/decoder/references"
run_id="${STREAMGLASS_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="$output_root/$run_id"

sudo install -d -m 0775 "$output_dir"

sudo docker run --rm \
    --runtime nvidia \
    --network none \
    --ipc host \
    --shm-size 8g \
    --volume "$model_dir:/models:ro" \
    --volume "$project_root:/workspace:ro" \
    --volume "$output_dir:/output" \
    --env FAIRSEQ2_USER_ASSET_DIR=/workspace/config/assets \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env STREAMGLASS_DECODER_CAPTURE_DIR=/output/tensors \
    "$image" \
    python /workspace/deploy/export/capture_decoder_reference.py \
        --task s2st \
        --data-file /workspace/tests/data/s0.tsv \
        --audio-root-dir /workspace/tests/data \
        --output /output/eval \
        --tgt-lang eng \
        --source-segment-size 320 \
        --dtype fp16 \
        --computation-aware \
        --no-scoring \
        --no-strip-silence

echo "Decoder official reference capture: $output_dir"
