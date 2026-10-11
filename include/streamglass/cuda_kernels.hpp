#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace streamglass {

void launch_float_to_half(
    const float* input,
    __half* output,
    std::size_t count,
    cudaStream_t stream);

void launch_causal_mask(
    __half* output, std::size_t target_length, cudaStream_t stream);

[[nodiscard]] std::size_t argmax_half_scratch_count(
    std::size_t element_count) noexcept;

void launch_argmax_half(
    const __half* input,
    std::size_t element_count,
    __half* scratch_values,
    std::int32_t* scratch_indices,
    std::size_t scratch_count,
    std::int64_t* output_index,
    cudaStream_t stream);

void launch_min_float(
    const float* input,
    std::size_t element_count,
    float* output,
    cudaStream_t stream);

void launch_duration_prefix_sum(
    const std::int64_t* durations,
    std::size_t duration_count,
    std::int64_t* offsets,
    std::int64_t* total,
    cudaStream_t stream);

void launch_t2u_policy_slice(
    const std::int64_t* offsets,
    std::size_t duration_count,
    std::size_t duration_start_index,
    bool source_finished,
    std::size_t minimum_unit_chunk,
    std::int64_t* output_range,
    std::int32_t* output_state,
    cudaStream_t stream);

void launch_expand_by_duration(
    const __half* char_features,
    const std::int64_t* offsets,
    std::size_t character_count,
    std::size_t feature_dimension,
    __half* expanded_features,
    cudaStream_t stream);

void launch_int32_to_int64(
    const std::int32_t* input,
    std::int64_t* output,
    std::size_t element_count,
    cudaStream_t stream);

void launch_half_to_float(
    const __half* input,
    float* output,
    std::size_t element_count,
    cudaStream_t stream);

}  // namespace streamglass
