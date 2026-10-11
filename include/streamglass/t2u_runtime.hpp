#pragma once

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/tokenizer_assets.hpp"
#include "streamglass/trt_engine.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace streamglass {

enum class T2uAction : std::int32_t {
    read = 0,
    write = 1,
    finished = 2,
};

struct T2uResult {
    T2uAction action = T2uAction::read;
    std::size_t character_count = 0;
    std::size_t total_unit_count = 0;
    std::size_t unit_offset = 0;
    std::size_t unit_count = 0;
    std::size_t next_duration_start_index = 0;
};

class T2uRuntime final {
public:
    static constexpr std::size_t kFeatureDimension = 1'024;
    static constexpr std::size_t kMinimumTextLength = 2;
    static constexpr std::size_t kMaximumTextLength = 128;
    static constexpr std::size_t kMinimumCharacterCount = 2;
    static constexpr std::size_t kMaximumCharacterCount = 512;
    static constexpr std::size_t kMaximumUnitCount = 1'024;

    T2uRuntime(
        const std::filesystem::path& frontend_engine_path,
        const std::filesystem::path& decoder_engine_path,
        const TokenizerAssets& tokenizer);

    T2uRuntime(const T2uRuntime&) = delete;
    T2uRuntime& operator=(const T2uRuntime&) = delete;

    [[nodiscard]] T2uResult run(
        std::span<const std::int64_t> token_indices,
        const __half* decoder_features,
        std::size_t decoder_feature_count,
        std::size_t duration_start_index,
        bool source_finished,
        std::size_t minimum_unit_chunk = 50,
        std::uint64_t text_version = 0);

    // Drops the memoized frontend/decoder products. Must be called whenever
    // the utterance state resets.
    void invalidate_memo() noexcept {
        memo_valid_ = false;
        memo_text_version_ = 0;
    }

    [[nodiscard]] const std::int64_t* unit_ids_device(
        const T2uResult& result) const noexcept {
        return unit_ids_int64_.data() + result.unit_offset;
    }

    void copy_durations(
        std::span<std::int64_t> destination,
        std::size_t character_count) const;
    void copy_unit_ids(
        std::span<std::int32_t> destination,
        std::size_t unit_count) const;
    void copy_character_features(
        std::span<__half> destination,
        std::size_t character_count) const;
    void copy_expanded_features(
        std::span<__half> destination,
        std::size_t unit_count) const;

    [[nodiscard]] std::span<const std::int64_t> last_character_indices()
        const noexcept {
        return {character_indices_.data(), last_character_count_};
    }
    [[nodiscard]] std::span<const std::int64_t> last_character_lengths()
        const noexcept {
        return {character_lengths_.data(), last_text_length_};
    }

private:
    void validate_engines();

    const TokenizerAssets& tokenizer_;
    TensorRtEngine frontend_engine_;
    TensorRtEngine decoder_engine_;
    CudaStream stream_;
    MappedHostBuffer<std::int64_t> character_indices_;
    MappedHostBuffer<std::int64_t> character_lengths_;
    MappedHostBuffer<std::int64_t> total_unit_count_;
    MappedHostBuffer<std::int64_t> output_range_;
    MappedHostBuffer<std::int32_t> output_state_;
    DeviceBuffer<__half> character_features_;
    DeviceBuffer<std::int64_t> durations_;
    DeviceBuffer<std::int64_t> duration_offsets_;
    DeviceBuffer<__half> expanded_features_;
    DeviceBuffer<std::int32_t> unit_ids_int32_;
    DeviceBuffer<std::int64_t> unit_ids_int64_;
    std::size_t last_text_length_ = 0;
    std::size_t last_character_count_ = 0;
    std::size_t memo_total_units_ = 0;
    std::uint64_t memo_text_version_ = 0;
    bool memo_valid_ = false;
};

}  // namespace streamglass
