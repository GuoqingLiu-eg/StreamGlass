#include "streamglass/trt_engine.hpp"

#include <NvInferPlugin.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace streamglass {
namespace {

TensorRtLogger& process_logger() {
    static TensorRtLogger logger(nvinfer1::ILogger::Severity::kWARNING);
    return logger;
}

std::vector<std::byte> read_binary_file(const std::filesystem::path& path) {
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error(
            "Cannot stat TensorRT engine " + path.string() + ": " +
            error.message());
    }
    if (file_size == 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path.string());
    }

    std::vector<std::byte> bytes(file_size);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Cannot read TensorRT engine: " + path.string());
    }
    return bytes;
}

}  // namespace

void TensorRtLogger::log(const Severity severity, const char* message) noexcept {
    if (severity > threshold_) {
        return;
    }
    std::cerr << "[TensorRT] " << message << '\n';
}

TensorRtEngine::TensorRtEngine(
    std::string name,
    const std::filesystem::path& path)
    : name_(std::move(name)), path_(path), serialized_(read_binary_file(path)) {
    auto& logger = process_logger();
    if (!initLibNvInferPlugins(&logger, "")) {
        throw std::runtime_error("TensorRT plugin initialization failed");
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
        throw std::runtime_error("TensorRT runtime creation failed");
    }
    engine_.reset(runtime_->deserializeCudaEngine(
        serialized_.data(), serialized_.size()));
    if (!engine_) {
        throw std::runtime_error(
            "TensorRT deserialization failed: " + path.string());
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error(
            "TensorRT execution context creation failed: " + path.string());
    }

    const auto profile_count = engine_->getNbOptimizationProfiles();
    const auto tensor_count = engine_->getNbIOTensors();
    tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::int32_t index = 0; index < tensor_count; ++index) {
        const char* raw_name = engine_->getIOTensorName(index);
        if (raw_name == nullptr) {
            throw std::runtime_error("TensorRT returned an unnamed I/O tensor");
        }
        TensorInfo info{
            .name = raw_name,
            .mode = engine_->getTensorIOMode(raw_name),
            .dtype = engine_->getTensorDataType(raw_name),
            .location = engine_->getTensorLocation(raw_name),
            .engine_shape = engine_->getTensorShape(raw_name),
            .profiles = {},
        };
        if (info.mode == nvinfer1::TensorIOMode::kINPUT) {
            info.profiles.reserve(static_cast<std::size_t>(profile_count));
            for (std::int32_t profile = 0; profile < profile_count; ++profile) {
                info.profiles.push_back(TensorProfile{
                    .minimum = engine_->getProfileShape(
                        raw_name, profile, nvinfer1::OptProfileSelector::kMIN),
                    .optimum = engine_->getProfileShape(
                        raw_name, profile, nvinfer1::OptProfileSelector::kOPT),
                    .maximum = engine_->getProfileShape(
                        raw_name, profile, nvinfer1::OptProfileSelector::kMAX),
                });
            }
        }
        tensors_.push_back(std::move(info));
    }

    // The serialized plan is no longer needed after deserialization. Keeping it
    // would duplicate several gigabytes of decoder weights in host memory.
    serialized_.clear();
    serialized_.shrink_to_fit();
}

std::string dims_to_string(const nvinfer1::Dims& dims) {
    std::ostringstream output;
    output << '[';
    for (std::int32_t axis = 0; axis < dims.nbDims; ++axis) {
        if (axis != 0) {
            output << ',';
        }
        output << dims.d[axis];
    }
    output << ']';
    return output.str();
}

std::string data_type_to_string(const nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT8:
            return "int8";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        case nvinfer1::DataType::kUINT8:
            return "uint8";
        case nvinfer1::DataType::kFP8:
            return "fp8";
        case nvinfer1::DataType::kBF16:
            return "bfloat16";
        case nvinfer1::DataType::kINT64:
            return "int64";
        case nvinfer1::DataType::kINT4:
            return "int4";
        case nvinfer1::DataType::kFP4:
            return "fp4";
        case nvinfer1::DataType::kE8M0:
            return "e8m0";
    }
    return "unknown";
}

std::string io_mode_to_string(const nvinfer1::TensorIOMode mode) {
    switch (mode) {
        case nvinfer1::TensorIOMode::kINPUT:
            return "input";
        case nvinfer1::TensorIOMode::kOUTPUT:
            return "output";
        case nvinfer1::TensorIOMode::kNONE:
            return "none";
    }
    return "unknown";
}

std::string location_to_string(const nvinfer1::TensorLocation location) {
    switch (location) {
        case nvinfer1::TensorLocation::kDEVICE:
            return "device";
        case nvinfer1::TensorLocation::kHOST:
            return "host";
    }
    return "unknown";
}

}  // namespace streamglass
