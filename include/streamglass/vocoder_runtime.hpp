#pragma once

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/trt_engine.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace streamglass {

struct VocoderResult {
    std::size_t unit_count = 0;
    std::size_t sample_count = 0;
};

class VocoderRuntime final {
public:
    static constexpr std::size_t kSampleRate = 16'000;
    static constexpr std::size_t kSamplesPerUnit = 320;
    static constexpr std::size_t kMinimumUnitCount = 1;
    static constexpr std::size_t kMaximumUnitCount = 256;
    static constexpr std::size_t kMaximumSampleCount =
        kMaximumUnitCount * kSamplesPerUnit;

    explicit VocoderRuntime(const std::filesystem::path& engine_path);

    VocoderRuntime(const VocoderRuntime&) = delete;
    VocoderRuntime& operator=(const VocoderRuntime&) = delete;

    [[nodiscard]] VocoderResult run(
        const std::int64_t* unit_ids,
        std::size_t unit_count,
        std::int64_t language_id,
        std::int64_t speaker_id);

    [[nodiscard]] const __half* waveform_device() const noexcept {
        return waveform_.data();
    }

    [[nodiscard]] std::span<const float> materialize_for_playback(
        const VocoderResult& result);
    [[nodiscard]] std::span<const float> materialize_for_playback(
        const VocoderResult& result,
        MappedHostBuffer<float>& destination);
    void copy_waveform(
        const VocoderResult& result, std::span<__half> destination) const;

private:
    void validate_engine();

    TensorRtEngine engine_;
    CudaStream stream_;
    MappedHostBuffer<std::int64_t> language_id_;
    MappedHostBuffer<std::int64_t> speaker_id_;
    MappedHostBuffer<float> playback_waveform_;
    DeviceBuffer<__half> waveform_;
    std::size_t last_sample_count_ = 0;
};

}  // namespace streamglass
