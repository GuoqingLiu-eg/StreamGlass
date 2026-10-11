#pragma once

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/trt_engine.hpp"

#include <cuda_fp16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace streamglass {

struct DecoderResult {
    std::int64_t selected_index = 0;
    float decision_probability = 0.0F;
    std::size_t feature_offset = 0;
    std::size_t feature_count = 0;
    std::size_t cache_length = 0;
};

class DecoderRuntime final {
public:
    static constexpr std::size_t kVocabularySize = 256'102;
    static constexpr std::size_t kDecisionCount = 384;
    static constexpr std::size_t kDecoderDimension = 1'024;
    static constexpr std::size_t kDecoderLayers = 24;
    static constexpr std::size_t kAttentionHeads = 16;
    static constexpr std::size_t kHeadDimension = 64;
    static constexpr std::size_t kMinimumTargetLength = 2;
    static constexpr std::size_t kMaximumTargetLength = 512;
    static constexpr std::size_t kMaximumEncoderFrames = 256;

    DecoderRuntime(
        const std::filesystem::path& prefill_engine_path,
        const std::filesystem::path& step_engine_path);

    DecoderRuntime(const DecoderRuntime&) = delete;
    DecoderRuntime& operator=(const DecoderRuntime&) = delete;

    [[nodiscard]] DecoderResult begin_policy(
        std::span<const std::int64_t> target_prefix,
        const __half* encoder_output,
        std::size_t encoder_frames);

    [[nodiscard]] DecoderResult step(std::int64_t target_token);

    // Cross-policy KV cache (cached mode). The canonical cache stores the
    // committed prefix minus its last token; resume_policy re-evaluates that
    // last token against the new encoder output instead of a full prefill.
    // Prefix self-attention K/V stay frozen at commit time, which is an
    // accepted semantic deviation gated by reference/cached A/B.
    [[nodiscard]] bool canonical_matches(
        std::size_t prefix_length, std::int64_t last_token) const noexcept {
        return canonical_valid_ &&
               canonical_length_ + 1U == prefix_length &&
               canonical_last_token_ == last_token;
    }

    [[nodiscard]] DecoderResult resume_policy(
        const __half* encoder_output, std::size_t encoder_frames);

    void capture_canonical(std::int64_t last_token);

    void invalidate_canonical() noexcept {
        canonical_valid_ = false;
        canonical_length_ = 0;
        canonical_last_token_ = -1;
    }

    [[nodiscard]] const __half* decoder_features_device() const noexcept {
        return decoder_features_.data();
    }
    [[nodiscard]] std::size_t decoder_feature_count() const noexcept {
        return feature_count_;
    }
    [[nodiscard]] std::size_t cache_length() const noexcept {
        return cache_length_;
    }

    void copy_result_features(
        const DecoderResult& result, std::span<__half> destination) const;

private:
    [[nodiscard]] DecoderResult collect_result(
        std::size_t feature_offset,
        std::size_t feature_count,
        std::size_t cache_length);
    void validate_engines();

    static constexpr std::size_t kMaximumCacheElements =
        kDecoderLayers * kAttentionHeads * kMaximumTargetLength *
        kHeadDimension;

    TensorRtEngine prefill_engine_;
    TensorRtEngine step_engine_;
    CudaStream stream_;
    MappedHostBuffer<std::int64_t> target_input_;
    MappedHostBuffer<std::int64_t> position_;
    MappedHostBuffer<std::int64_t> selected_index_;
    MappedHostBuffer<float> decision_probability_;
    DeviceBuffer<__half> causal_mask_;
    DeviceBuffer<__half> decoder_features_;
    DeviceBuffer<__half> logits_;
    DeviceBuffer<float> decision_values_;
    std::array<DeviceBuffer<__half>, 2> key_cache_;
    std::array<DeviceBuffer<__half>, 2> value_cache_;
    DeviceBuffer<__half> canonical_key_;
    DeviceBuffer<__half> canonical_value_;
    DeviceBuffer<__half> argmax_scratch_values_;
    DeviceBuffer<std::int32_t> argmax_scratch_indices_;
    const __half* encoder_output_ = nullptr;
    std::size_t encoder_frames_ = 0;
    std::size_t feature_count_ = 0;
    std::size_t cache_length_ = 0;
    std::size_t current_cache_index_ = 0;
    bool policy_active_ = false;
    std::size_t canonical_length_ = 0;
    std::int64_t canonical_last_token_ = -1;
    bool canonical_valid_ = false;
};

}  // namespace streamglass
