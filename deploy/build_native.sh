#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
fingerprint="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$project_root/config/platform.lock.json")}"
build_root="${STREAMGLASS_NATIVE_BUILD_ROOT:-/opt/streamglass/tmp/native-build/$fingerprint}"
install_root="${STREAMGLASS_NATIVE_INSTALL_ROOT:-/opt/streamglass/tmp/native-install/$fingerprint}"
cuda_compiler="${STREAMGLASS_CUDA_COMPILER:-/usr/local/cuda/bin/nvcc}"
cuda_architectures="${STREAMGLASS_CUDA_ARCHITECTURES:-87}"

if [[ ! -x "$cuda_compiler" ]]; then
    printf 'CUDA compiler is not executable: %s\n' "$cuda_compiler" >&2
    exit 1
fi

mkdir -p "$build_root" "$install_root"

cmake \
    -S "$project_root" \
    -B "$build_root" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_root" \
    -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_architectures" \
    -DSTREAMGLASS_ENABLE_TENSORRT=ON \
    -DSTREAMGLASS_BUILD_TESTS=ON

cmake --build "$build_root" --parallel
ctest --test-dir "$build_root" --output-on-failure
cmake --install "$build_root"

realtime_binary="$install_root/bin/streamglass-native-realtime"
if [[ -x "$realtime_binary" && -x "$(command -v setcap || true)" ]]; then
    sudo -n setcap cap_net_raw=ep "$realtime_binary"
    getcap "$realtime_binary"
fi

printf '%s\n' "$install_root/bin/streamglass-engine-inspect"
