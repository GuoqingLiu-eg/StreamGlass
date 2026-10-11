#include "streamglass/cuda_kernels.hpp"

#include "streamglass/cuda_buffer.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace streamglass {
namespace {

__global__ void float_to_half_kernel(
    const float* input, __half* output, const std::size_t count) {
    const auto index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2half(input[index]);
    }
}

__global__ void causal_mask_kernel(
    __half* output, const std::size_t target_length) {
    const auto column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto row =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < target_length && column < target_length) {
        const auto value = row >= column ? 0.0F : -CUDART_INF_F;
        output[row * target_length + column] = __float2half(value);
    }
}

struct Maximum {
    float value;
    std::int32_t index;
};

__device__ Maximum choose_maximum(
    const Maximum left, const Maximum right) {
    if (isnan(left.value)) {
        return right;
    }
    if (isnan(right.value)) {
        return left;
    }
    if (right.value > left.value ||
        (right.value == left.value && right.index < left.index)) {
        return right;
    }
    return left;
}

__global__ void argmax_half_blocks_kernel(
    const __half* input,
    const std::size_t element_count,
    __half* block_values,
    std::int32_t* block_indices) {
    constexpr auto kThreads = 256U;
    __shared__ float values[kThreads];
    __shared__ std::int32_t indices[kThreads];

    Maximum local{-CUDART_INF_F, std::numeric_limits<std::int32_t>::max()};
    auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    while (index < element_count) {
        local = choose_maximum(
            local,
            Maximum{
                __half2float(input[index]), static_cast<std::int32_t>(index)});
        index += stride;
    }

    values[threadIdx.x] = local.value;
    indices[threadIdx.x] = local.index;
    __syncthreads();

    for (auto offset = blockDim.x / 2U; offset > 0U; offset /= 2U) {
        if (threadIdx.x < offset) {
            const auto selected = choose_maximum(
                Maximum{values[threadIdx.x], indices[threadIdx.x]},
                Maximum{
                    values[threadIdx.x + offset],
                    indices[threadIdx.x + offset]});
            values[threadIdx.x] = selected.value;
            indices[threadIdx.x] = selected.index;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0U) {
        block_values[blockIdx.x] = __float2half(values[0]);
        block_indices[blockIdx.x] = indices[0];
    }
}

__global__ void argmax_half_final_kernel(
    const __half* block_values,
    const std::int32_t* block_indices,
    const std::size_t block_count,
    std::int64_t* output_index) {
    constexpr auto kThreads = 256U;
    __shared__ float values[kThreads];
    __shared__ std::int32_t indices[kThreads];

    Maximum local{-CUDART_INF_F, std::numeric_limits<std::int32_t>::max()};
    auto index = static_cast<std::size_t>(threadIdx.x);
    while (index < block_count) {
        local = choose_maximum(
            local,
            Maximum{__half2float(block_values[index]), block_indices[index]});
        index += blockDim.x;
    }
    values[threadIdx.x] = local.value;
    indices[threadIdx.x] = local.index;
    __syncthreads();

    for (auto offset = blockDim.x / 2U; offset > 0U; offset /= 2U) {
        if (threadIdx.x < offset) {
            const auto selected = choose_maximum(
                Maximum{values[threadIdx.x], indices[threadIdx.x]},
                Maximum{
                    values[threadIdx.x + offset],
                    indices[threadIdx.x + offset]});
            values[threadIdx.x] = selected.value;
            indices[threadIdx.x] = selected.index;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        output_index[0] = static_cast<std::int64_t>(indices[0]);
    }
}

__device__ float choose_minimum(const float left, const float right) {
    if (isnan(left) || isnan(right)) {
        return CUDART_NAN_F;
    }
    return fminf(left, right);
}

__global__ void min_float_kernel(
    const float* input,
    const std::size_t element_count,
    float* output) {
    constexpr auto kThreads = 256U;
    __shared__ float values[kThreads];
    auto local = CUDART_INF_F;
    auto index = static_cast<std::size_t>(threadIdx.x);
    while (index < element_count) {
        local = choose_minimum(local, input[index]);
        index += blockDim.x;
    }
    values[threadIdx.x] = local;
    __syncthreads();

    for (auto offset = blockDim.x / 2U; offset > 0U; offset /= 2U) {
        if (threadIdx.x < offset) {
            values[threadIdx.x] = choose_minimum(
                values[threadIdx.x], values[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        output[0] = values[0];
    }
}

__global__ void duration_prefix_sum_kernel(
    const std::int64_t* durations,
    const std::size_t duration_count,
    std::int64_t* offsets,
    std::int64_t* total) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    std::int64_t sum = 0;
    offsets[0] = 0;
    for (std::size_t index = 0; index < duration_count; ++index) {
        const auto duration = durations[index];
        if (duration <= 0 ||
            sum > std::numeric_limits<std::int64_t>::max() - duration) {
            total[0] = -1;
            return;
        }
        sum += duration;
        offsets[index + 1U] = sum;
    }
    total[0] = sum;
}

__global__ void t2u_policy_slice_kernel(
    const std::int64_t* offsets,
    const std::size_t duration_count,
    const std::size_t duration_start_index,
    const bool source_finished,
    const std::size_t minimum_unit_chunk,
    std::int64_t* output_range,
    std::int32_t* output_state) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    const auto total = offsets[duration_count];
    auto adjusted_start = duration_start_index;
    auto remaining = total - offsets[adjusted_start];

    if (source_finished && adjusted_start > 0U) {
        if (remaining == 0) {
            output_range[0] = total;
            output_range[1] = 0;
            output_state[0] = 2;
            output_state[1] = static_cast<std::int32_t>(adjusted_start);
            return;
        }
        --adjusted_start;
        remaining = total - offsets[adjusted_start];
    }

    if (remaining < static_cast<std::int64_t>(minimum_unit_chunk) &&
        !source_finished) {
        output_range[0] = offsets[adjusted_start];
        output_range[1] = 0;
        output_state[0] = 0;
        output_state[1] = static_cast<std::int32_t>(duration_start_index);
        return;
    }
    if (source_finished && remaining == 0) {
        output_range[0] = total;
        output_range[1] = 0;
        output_state[0] = 2;
        output_state[1] = static_cast<std::int32_t>(adjusted_start);
        return;
    }

    output_range[0] = offsets[adjusted_start];
    output_range[1] = remaining;
    output_state[0] = 1;
    output_state[1] = static_cast<std::int32_t>(duration_count - 1U);
}

__global__ void expand_by_duration_kernel(
    const __half* char_features,
    const std::int64_t* offsets,
    const std::size_t character_count,
    const std::size_t feature_dimension,
    __half* expanded_features) {
    const auto character = static_cast<std::size_t>(blockIdx.x);
    const auto feature =
        static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (character >= character_count || feature >= feature_dimension) {
        return;
    }
    const auto start = offsets[character];
    const auto end = offsets[character + 1U];
    const auto value = char_features[character * feature_dimension + feature];
    for (auto position = start; position < end; ++position) {
        expanded_features[
            static_cast<std::size_t>(position) * feature_dimension + feature] =
            value;
    }
}

__global__ void int32_to_int64_kernel(
    const std::int32_t* input,
    std::int64_t* output,
    const std::size_t element_count) {
    const auto index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = static_cast<std::int64_t>(input[index]);
    }
}

__global__ void half_to_float_kernel(
    const __half* input,
    float* output,
    const std::size_t element_count) {
    const auto index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = __half2float(input[index]);
    }
}

}  // namespace

void launch_float_to_half(
    const float* input,
    __half* output,
    const std::size_t count,
    const cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    constexpr std::size_t threads = 256;
    const auto blocks = (count + threads - 1) / threads;
    float_to_half_kernel<<<
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(threads),
        0,
        stream>>>(input, output, count);
    check_cuda(cudaGetLastError(), "float_to_half_kernel");
}

void launch_causal_mask(
    __half* output,
    const std::size_t target_length,
    const cudaStream_t stream) {
    if (target_length == 0) {
        return;
    }
    const dim3 threads{16U, 16U, 1U};
    const dim3 blocks{
        static_cast<unsigned int>((target_length + threads.x - 1U) / threads.x),
        static_cast<unsigned int>((target_length + threads.y - 1U) / threads.y),
        1U};
    causal_mask_kernel<<<blocks, threads, 0, stream>>>(output, target_length);
    check_cuda(cudaGetLastError(), "causal_mask_kernel");
}

std::size_t argmax_half_scratch_count(
    const std::size_t element_count) noexcept {
    constexpr std::size_t kThreads = 256;
    constexpr std::size_t kMaximumBlocks = 256;
    return std::min(
        kMaximumBlocks, (element_count + kThreads - 1U) / kThreads);
}

void launch_argmax_half(
    const __half* input,
    const std::size_t element_count,
    __half* scratch_values,
    std::int32_t* scratch_indices,
    const std::size_t scratch_count,
    std::int64_t* output_index,
    const cudaStream_t stream) {
    const auto block_count = argmax_half_scratch_count(element_count);
    if (element_count == 0 || scratch_count < block_count) {
        throw std::invalid_argument("argmax_half scratch buffer is too small");
    }
    constexpr unsigned int kThreads = 256U;
    argmax_half_blocks_kernel<<<
        static_cast<unsigned int>(block_count), kThreads, 0, stream>>>(
        input, element_count, scratch_values, scratch_indices);
    check_cuda(cudaGetLastError(), "argmax_half_blocks_kernel");
    argmax_half_final_kernel<<<1U, kThreads, 0, stream>>>(
        scratch_values, scratch_indices, block_count, output_index);
    check_cuda(cudaGetLastError(), "argmax_half_final_kernel");
}

void launch_min_float(
    const float* input,
    const std::size_t element_count,
    float* output,
    const cudaStream_t stream) {
    if (element_count == 0) {
        throw std::invalid_argument("min_float requires at least one element");
    }
    constexpr unsigned int kThreads = 256U;
    min_float_kernel<<<1U, kThreads, 0, stream>>>(
        input, element_count, output);
    check_cuda(cudaGetLastError(), "min_float_kernel");
}

void launch_duration_prefix_sum(
    const std::int64_t* durations,
    const std::size_t duration_count,
    std::int64_t* offsets,
    std::int64_t* total,
    const cudaStream_t stream) {
    if (duration_count == 0) {
        throw std::invalid_argument(
            "duration_prefix_sum requires at least one duration");
    }
    duration_prefix_sum_kernel<<<1U, 1U, 0, stream>>>(
        durations, duration_count, offsets, total);
    check_cuda(cudaGetLastError(), "duration_prefix_sum_kernel");
}

void launch_t2u_policy_slice(
    const std::int64_t* offsets,
    const std::size_t duration_count,
    const std::size_t duration_start_index,
    const bool source_finished,
    const std::size_t minimum_unit_chunk,
    std::int64_t* output_range,
    std::int32_t* output_state,
    const cudaStream_t stream) {
    // The official final policy accepts a cursor exactly at the end of the
    // shortened character sequence (the provisional trailing comma is gone)
    // and returns FINISHED with no additional units.
    if (duration_count == 0 || duration_start_index > duration_count ||
        (duration_start_index == duration_count && !source_finished)) {
        throw std::invalid_argument("T2U duration cursor is outside the sequence");
    }
    t2u_policy_slice_kernel<<<1U, 1U, 0, stream>>>(
        offsets,
        duration_count,
        duration_start_index,
        source_finished,
        minimum_unit_chunk,
        output_range,
        output_state);
    check_cuda(cudaGetLastError(), "t2u_policy_slice_kernel");
}

void launch_expand_by_duration(
    const __half* char_features,
    const std::int64_t* offsets,
    const std::size_t character_count,
    const std::size_t feature_dimension,
    __half* expanded_features,
    const cudaStream_t stream) {
    if (character_count == 0 || feature_dimension == 0) {
        throw std::invalid_argument("Duration expansion dimensions must be positive");
    }
    constexpr unsigned int kThreads = 256U;
    const dim3 blocks{
        static_cast<unsigned int>(character_count),
        static_cast<unsigned int>((feature_dimension + kThreads - 1U) / kThreads),
        1U};
    expand_by_duration_kernel<<<blocks, kThreads, 0, stream>>>(
        char_features,
        offsets,
        character_count,
        feature_dimension,
        expanded_features);
    check_cuda(cudaGetLastError(), "expand_by_duration_kernel");
}

void launch_int32_to_int64(
    const std::int32_t* input,
    std::int64_t* output,
    const std::size_t element_count,
    const cudaStream_t stream) {
    if (element_count == 0) {
        return;
    }
    constexpr std::size_t kThreads = 256;
    const auto blocks = (element_count + kThreads - 1U) / kThreads;
    int32_to_int64_kernel<<<
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(kThreads),
        0,
        stream>>>(input, output, element_count);
    check_cuda(cudaGetLastError(), "int32_to_int64_kernel");
}

void launch_half_to_float(
    const __half* input,
    float* output,
    const std::size_t element_count,
    const cudaStream_t stream) {
    if (element_count == 0) {
        return;
    }
    constexpr std::size_t kThreads = 256;
    const auto blocks = (element_count + kThreads - 1U) / kThreads;
    half_to_float_kernel<<<
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(kThreads),
        0,
        stream>>>(input, output, element_count);
    check_cuda(cudaGetLastError(), "half_to_float_kernel");
}

}  // namespace streamglass
