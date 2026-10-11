#include "streamglass/silero_vad_trt.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace streamglass {
namespace {

bool shape_is(
    const nvinfer1::Dims& shape,
    const std::initializer_list<std::int64_t> expected) {
    if (shape.nbDims != static_cast<std::int32_t>(expected.size())) {
        return false;
    }
    std::size_t axis = 0;
    for (const auto value : expected) {
        if (shape.d[axis] != value) {
            return false;
        }
        ++axis;
    }
    return true;
}

void bind(
    nvinfer1::IExecutionContext& context, const char* name, void* pointer) {
    if (!context.setTensorAddress(name, pointer)) {
        throw std::runtime_error(
            "TensorRT rejected Silero tensor address for " + std::string(name));
    }
}

}  // namespace

SileroVadTrt::SileroVadTrt(const std::filesystem::path& engine_path)
    : engine_("silero-vad-experiment", engine_path),
      stream_(),
      input_(kInputSamples),
      probability_(1),
      state_current_(kStateElements),
      state_next_(kStateElements) {
    validate_engine();
    reset();
}

void SileroVadTrt::validate_engine() {
    const auto& engine = engine_.engine();
    if (engine.getNbIOTensors() != 4 ||
        engine.getTensorIOMode("input") != nvinfer1::TensorIOMode::kINPUT ||
        engine.getTensorIOMode("state") != nvinfer1::TensorIOMode::kINPUT ||
        engine.getTensorIOMode("output") != nvinfer1::TensorIOMode::kOUTPUT ||
        engine.getTensorIOMode("stateN") != nvinfer1::TensorIOMode::kOUTPUT ||
        engine.getTensorDataType("input") != nvinfer1::DataType::kFLOAT ||
        engine.getTensorDataType("state") != nvinfer1::DataType::kFLOAT ||
        engine.getTensorDataType("output") != nvinfer1::DataType::kFLOAT ||
        engine.getTensorDataType("stateN") != nvinfer1::DataType::kFLOAT ||
        !shape_is(engine.getTensorShape("input"), {1, 576}) ||
        !shape_is(engine.getTensorShape("state"), {2, 1, 128}) ||
        !shape_is(engine.getTensorShape("output"), {1, 1}) ||
        !shape_is(engine.getTensorShape("stateN"), {2, 1, 128})) {
        throw std::runtime_error(
            "Silero VAD TensorRT I/O contract changed or is not static 16 kHz");
    }
}

void SileroVadTrt::reset() {
    std::fill(context_.begin(), context_.end(), 0.0F);
    std::fill(input_.span().begin(), input_.span().end(), 0.0F);
    probability_.data()[0] = 0.0F;
    check_cuda(
        cudaMemsetAsync(
            state_current_.data(), 0, state_current_.bytes(), stream_.get()),
        "cudaMemsetAsync Silero current state");
    check_cuda(
        cudaMemsetAsync(
            state_next_.data(), 0, state_next_.bytes(), stream_.get()),
        "cudaMemsetAsync Silero next state");
    stream_.synchronize();
}

float SileroVadTrt::run(const std::span<const float> chunk) {
    if (chunk.size() != kChunkSamples) {
        throw std::invalid_argument("Silero VAD requires exactly 512 samples");
    }
    std::copy(context_.begin(), context_.end(), input_.data());
    std::copy(chunk.begin(), chunk.end(), input_.data() + kContextSamples);

    auto& execution = engine_.context();
    bind(execution, "input", input_.device_data());
    bind(execution, "state", state_current_.data());
    bind(execution, "output", probability_.device_data());
    bind(execution, "stateN", state_next_.data());
    if (!execution.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT Silero VAD enqueueV3 failed");
    }
    stream_.synchronize();
    std::swap(state_current_, state_next_);
    std::copy(
        chunk.end() - static_cast<std::ptrdiff_t>(kContextSamples),
        chunk.end(),
        context_.begin());

    const auto result = probability_.data()[0];
    if (!std::isfinite(result) || result < 0.0F || result > 1.0F) {
        throw std::runtime_error("Silero VAD returned an invalid probability");
    }
    return result;
}

}  // namespace streamglass
