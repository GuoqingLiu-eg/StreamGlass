#include "streamglass/encoder_runtime.hpp"

#include "streamglass/cuda_kernels.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace streamglass {
namespace {

nvinfer1::Dims make_dims(
    const std::initializer_list<std::int64_t> dimensions) {
    nvinfer1::Dims result{};
    result.nbDims = static_cast<std::int32_t>(dimensions.size());
    std::size_t axis = 0;
    for (const auto value : dimensions) {
        result.d[axis] = value;
        ++axis;
    }
    return result;
}

void set_shape(
    nvinfer1::IExecutionContext& context,
    const char* name,
    const nvinfer1::Dims& shape) {
    if (!context.setInputShape(name, shape)) {
        throw std::runtime_error(
            "TensorRT rejected input shape for " + std::string(name) +
            ": " + dims_to_string(shape));
    }
}

void bind(
    nvinfer1::IExecutionContext& context, const char* name, void* pointer) {
    if (!context.setTensorAddress(name, pointer)) {
        throw std::runtime_error(
            "TensorRT rejected tensor address for " + std::string(name));
    }
}

}  // namespace

EncoderRuntime::EncoderRuntime(const std::filesystem::path& engine_path)
    : engine_("encoder", engine_path),
      stream_(),
      host_fbank_(kMaximumFbankFrames * kFeatureDimension),
      input_length_(1),
      output_length_(1),
      device_fbank_(kMaximumFbankFrames * kFeatureDimension),
      device_encoder_output_(kMaximumEncoderFrames * kEncoderDimension) {
    const auto& trt_engine = engine_.engine();
    if (trt_engine.getNbIOTensors() != 4 ||
        trt_engine.getTensorDataType("fbank") != nvinfer1::DataType::kHALF ||
        trt_engine.getTensorDataType("fbank_seq_lens") !=
            nvinfer1::DataType::kINT64 ||
        trt_engine.getTensorDataType("encoder_output") !=
            nvinfer1::DataType::kHALF ||
        trt_engine.getTensorDataType("encoder_seq_lens") !=
            nvinfer1::DataType::kINT64) {
        throw std::runtime_error("Encoder TensorRT I/O contract changed");
    }
}

EncoderResult EncoderRuntime::run(
    const std::size_t valid_fbank_frames,
    const std::size_t padded_fbank_frames) {
    if (valid_fbank_frames < kMinimumFbankFrames ||
        valid_fbank_frames > padded_fbank_frames ||
        padded_fbank_frames > kMaximumFbankFrames ||
        padded_fbank_frames % 2 != 0) {
        throw std::invalid_argument("Encoder fbank lengths are outside profile");
    }

    std::fill(
        host_fbank_.data() + valid_fbank_frames * kFeatureDimension,
        host_fbank_.data() + padded_fbank_frames * kFeatureDimension,
        0.0F);
    input_length_.data()[0] = static_cast<std::int64_t>(valid_fbank_frames);

    const auto element_count = padded_fbank_frames * kFeatureDimension;
    launch_float_to_half(
        host_fbank_.device_data(),
        device_fbank_.data(),
        element_count,
        stream_.get());

    auto& context = engine_.context();
    set_shape(
        context,
        "fbank",
        make_dims(
            {1,
             static_cast<std::int64_t>(padded_fbank_frames),
             static_cast<std::int64_t>(kFeatureDimension)}));
    set_shape(context, "fbank_seq_lens", make_dims({1}));

    bind(context, "fbank", device_fbank_.data());
    bind(context, "fbank_seq_lens", input_length_.device_data());
    bind(context, "encoder_output", device_encoder_output_.data());
    bind(context, "encoder_seq_lens", output_length_.device_data());

    const auto output_shape = context.getTensorShape("encoder_output");
    if (output_shape.nbDims != 3 || output_shape.d[0] != 1 ||
        output_shape.d[1] <= 0 ||
        output_shape.d[1] >
            static_cast<std::int64_t>(kMaximumEncoderFrames) ||
        output_shape.d[2] != static_cast<std::int64_t>(kEncoderDimension)) {
        throw std::runtime_error(
            "Encoder returned unresolved or invalid output shape " +
            dims_to_string(output_shape));
    }
    if (!context.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT encoder enqueueV3 failed");
    }
    stream_.synchronize();

    const auto valid_output = output_length_.data()[0];
    if (valid_output <= 0 || valid_output > output_shape.d[1]) {
        throw std::runtime_error("Encoder returned an invalid sequence length");
    }
    return EncoderResult{
        .tensor_frames = static_cast<std::size_t>(output_shape.d[1]),
        .valid_frames = static_cast<std::size_t>(valid_output),
    };
}

void EncoderRuntime::copy_output(
    const std::span<__half> destination,
    const std::size_t tensor_frames) const {
    const auto element_count = tensor_frames * kEncoderDimension;
    if (element_count > device_encoder_output_.size() ||
        destination.size() < element_count) {
        throw std::length_error("Encoder output destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            device_encoder_output_.data(),
            element_count * sizeof(__half),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy encoder output");
}

}  // namespace streamglass
