#!/usr/bin/env bash
set -euo pipefail

image="${STREAMGLASS_REFERENCE_IMAGE:-nvcr.io/nvidia/pytorch:24.01-py3}"
max_attempts="${STREAMGLASS_PULL_ATTEMPTS:-12}"
retry_seconds="${STREAMGLASS_PULL_RETRY_SECONDS:-5}"

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo "Pulling $image (attempt $attempt/$max_attempts)"
    if sudo docker pull --quiet "$image"; then
        sudo docker image inspect "$image" --format \
            'reference={{index .RepoDigests 0}} size={{.Size}} architecture={{.Architecture}}'
        exit 0
    fi

    if ((attempt < max_attempts)); then
        echo "Pull interrupted; retrying in ${retry_seconds}s" >&2
        sleep "$retry_seconds"
    fi
done

echo "Failed to pull $image after $max_attempts attempts" >&2
exit 1
