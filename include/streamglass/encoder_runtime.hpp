#pragma once

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/trt_engine.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace streamglass {

struct EncoderResult {
    std::size_t tensor_frames = 0;
    std::size_t valid_frames = 0;
};

class EncoderRuntime final {
public:
    static constexpr std::size_t kFeatureDimension = 80;
    static constexpr std::size_t kEncoderDimension = 1'024;
    static constexpr std::size_t kMinimumFbankFrames = 222;
    static constexpr std::size_t kMaximumFbankFrames = 2'000;
    static constexpr std::size_t kMaximumEncoderFrames = 256;

    explicit EncoderRuntime(const std::filesystem::path& engine_path);

    EncoderRuntime(const EncoderRuntime&) = delete;
    EncoderRuntime& operator=(const EncoderRuntime&) = delete;

    [[nodiscard]] std::span<float> fbank_input() noexcept {
        return host_fbank_.span();
    }

    [[nodiscard]] EncoderResult run(
        std::size_t valid_fbank_frames,
        std::size_t padded_fbank_frames);

    [[nodiscard]] const __half* output_device() const noexcept {
        return device_encoder_output_.data();
    }

    void copy_output(
        std::span<__half> destination, std::size_t tensor_frames) const;

private:
    TensorRtEngine engine_;
    CudaStream stream_;
    MappedHostBuffer<float> host_fbank_;
    MappedHostBuffer<std::int64_t> input_length_;
    MappedHostBuffer<std::int64_t> output_length_;
    DeviceBuffer<__half> device_fbank_;
    DeviceBuffer<__half> device_encoder_output_;
};

}  // namespace streamglass
