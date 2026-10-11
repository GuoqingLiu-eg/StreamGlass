#pragma once

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/trt_engine.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>

namespace streamglass {

class SileroVadTrt final {
public:
    static constexpr std::size_t kSampleRate = 16'000;
    static constexpr std::size_t kChunkSamples = 512;
    static constexpr std::size_t kContextSamples = 64;
    static constexpr std::size_t kInputSamples =
        kContextSamples + kChunkSamples;
    static constexpr std::size_t kStateElements = 2 * 1 * 128;

    explicit SileroVadTrt(const std::filesystem::path& engine_path);

    SileroVadTrt(const SileroVadTrt&) = delete;
    SileroVadTrt& operator=(const SileroVadTrt&) = delete;

    void reset();
    [[nodiscard]] float run(std::span<const float> chunk);

private:
    void validate_engine();

    TensorRtEngine engine_;
    CudaStream stream_;
    MappedHostBuffer<float> input_;
    MappedHostBuffer<float> probability_;
    DeviceBuffer<float> state_current_;
    DeviceBuffer<float> state_next_;
    std::array<float, kContextSamples> context_{};
};

}  // namespace streamglass
