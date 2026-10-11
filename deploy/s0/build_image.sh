#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
image="${STREAMGLASS_S0_IMAGE:-streamglass/seamless-s0:0.1}"
base_image="${STREAMGLASS_REFERENCE_IMAGE:-nvcr.io/nvidia/pytorch:24.01-py3}"
proxy_url="${STREAMGLASS_BOOTSTRAP_PROXY:-}"

build_args=(--build-arg "BASE_IMAGE=$base_image")
if [[ -n "$proxy_url" ]]; then
    build_args+=(
        --build-arg "HTTP_PROXY=$proxy_url"
        --build-arg "HTTPS_PROXY=$proxy_url"
        --build-arg "NO_PROXY=localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8"
    )
fi

sudo docker build \
    --progress=plain \
    --file "$script_dir/Dockerfile" \
    --tag "$image" \
    "${build_args[@]}" \
    "$project_root"

sudo docker image inspect "$image" --format \
    'reference={{index .RepoTags 0}} size={{.Size}} architecture={{.Architecture}}'
