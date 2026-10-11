#!/usr/bin/env bash
set -euo pipefail

image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"

sudo docker run --rm \
    --runtime nvidia \
    --network none \
    --ipc host \
    --shm-size 8g \
    "$image"
