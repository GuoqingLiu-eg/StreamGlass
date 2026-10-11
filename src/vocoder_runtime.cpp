#include "streamglass/vocoder_runtime.hpp"

#include "streamglass/cuda_kernels.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

VocoderRuntime::VocoderRuntime(const std::filesystem::path& engine_path)
    : engine_("vocoder", engine_path),
      stream_(),
      language_id_(1),
      speaker_id_(1),
      playback_waveform_(kMaximumSampleCount),
      waveform_(kMaximumSampleCount) {
    validate_engine();
}

void VocoderRuntime::validate_engine() {
    const auto& engine = engine_.engine();
    if (engine.getNbIOTensors() != 4 ||
        engine.getTensorDataType("unit_ids") != nvinfer1::DataType::kINT64 ||
        engine.getTensorDataType("lang_ids") != nvinfer1::DataType::kINT64 ||
        engine.getTensorDataType("speaker_ids") !=
            nvinfer1::DataType::kINT64 ||
        engine.getTensorDataType("waveform") != nvinfer1::DataType::kHALF) {
        throw std::runtime_error("Vocoder TensorRT I/O contract changed");
    }
}

VocoderResult VocoderRuntime::run(
    const std::int64_t* unit_ids,
    const std::size_t unit_count,
    const std::int64_t language_id,
    const std::int64_t speaker_id) {
    if (unit_ids == nullptr || unit_count < kMinimumUnitCount ||
        unit_count > kMaximumUnitCount) {
        throw std::invalid_argument("Vocoder unit input is outside profile");
    }
    language_id_.data()[0] = language_id;
    speaker_id_.data()[0] = speaker_id;

    auto& context = engine_.context();
    set_shape(
        context,
        "unit_ids",
        make_dims({1, static_cast<std::int64_t>(unit_count)}));
    set_shape(context, "lang_ids", make_dims({1, 1}));
    set_shape(context, "speaker_ids", make_dims({1, 1}));
    bind(context, "unit_ids", const_cast<std::int64_t*>(unit_ids));
    bind(context, "lang_ids", language_id_.device_data());
    bind(context, "speaker_ids", speaker_id_.device_data());
    bind(context, "waveform", waveform_.data());

    const auto waveform_shape = context.getTensorShape("waveform");
    const auto expected_samples = unit_count * kSamplesPerUnit;
    if (waveform_shape.nbDims != 3 || waveform_shape.d[0] != 1 ||
        waveform_shape.d[1] != 1 ||
        waveform_shape.d[2] != static_cast<std::int64_t>(expected_samples)) {
        throw std::runtime_error(
            "Vocoder returned an unexpected waveform shape " +
            dims_to_string(waveform_shape));
    }
    if (!context.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT vocoder enqueueV3 failed");
    }
    stream_.synchronize();
    last_sample_count_ = expected_samples;
    return VocoderResult{
        .unit_count = unit_count,
        .sample_count = expected_samples,
    };
}

std::span<const float> VocoderRuntime::materialize_for_playback(
    const VocoderResult& result) {
    return materialize_for_playback(result, playback_waveform_);
}

std::span<const float> VocoderRuntime::materialize_for_playback(
    const VocoderResult& result,
    MappedHostBuffer<float>& destination) {
    if (result.sample_count == 0 || result.sample_count != last_sample_count_) {
        throw std::invalid_argument("Vocoder result is not the latest waveform");
    }
    if (destination.size() < result.sample_count) {
        throw std::length_error("Mapped playback destination is too small");
    }
    launch_half_to_float(
        waveform_.data(),
        destination.device_data(),
        result.sample_count,
        stream_.get());
    stream_.synchronize();
    return {destination.data(), result.sample_count};
}

void VocoderRuntime::copy_waveform(
    const VocoderResult& result,
    const std::span<__half> destination) const {
    if (result.sample_count == 0 || result.sample_count != last_sample_count_ ||
        destination.size() < result.sample_count) {
        throw std::length_error("Vocoder waveform destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            waveform_.data(),
            result.sample_count * sizeof(__half),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy vocoder waveform");
}

}  // namespace streamglass
