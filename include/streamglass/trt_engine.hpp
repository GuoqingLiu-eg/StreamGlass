#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace streamglass {

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    explicit TensorRtLogger(Severity threshold = Severity::kWARNING) noexcept
        : threshold_(threshold) {}

    void log(Severity severity, const char* message) noexcept override;

private:
    Severity threshold_;
};

template <typename T>
struct TensorRtDelete {
    void operator()(T* pointer) const noexcept { delete pointer; }
};

struct TensorProfile {
    nvinfer1::Dims minimum{};
    nvinfer1::Dims optimum{};
    nvinfer1::Dims maximum{};
};

struct TensorInfo {
    std::string name;
    nvinfer1::TensorIOMode mode{};
    nvinfer1::DataType dtype{};
    nvinfer1::TensorLocation location{};
    nvinfer1::Dims engine_shape{};
    std::vector<TensorProfile> profiles;
};

class TensorRtEngine final {
public:
    TensorRtEngine(
        std::string name,
        const std::filesystem::path& path);

    TensorRtEngine(const TensorRtEngine&) = delete;
    TensorRtEngine& operator=(const TensorRtEngine&) = delete;
    TensorRtEngine(TensorRtEngine&&) = delete;
    TensorRtEngine& operator=(TensorRtEngine&&) = delete;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }
    [[nodiscard]] std::span<const TensorInfo> tensors() const noexcept {
        return tensors_;
    }
    [[nodiscard]] nvinfer1::ICudaEngine& engine() noexcept { return *engine_; }
    [[nodiscard]] nvinfer1::IExecutionContext& context() noexcept {
        return *context_;
    }

private:
    std::string name_;
    std::filesystem::path path_;
    std::vector<std::byte> serialized_;
    std::unique_ptr<nvinfer1::IRuntime, TensorRtDelete<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDelete<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<
        nvinfer1::IExecutionContext,
        TensorRtDelete<nvinfer1::IExecutionContext>>
        context_;
    std::vector<TensorInfo> tensors_;
};

[[nodiscard]] std::string dims_to_string(const nvinfer1::Dims& dims);
[[nodiscard]] std::string data_type_to_string(nvinfer1::DataType dtype);
[[nodiscard]] std::string io_mode_to_string(nvinfer1::TensorIOMode mode);
[[nodiscard]] std::string location_to_string(nvinfer1::TensorLocation location);

}  // namespace streamglass
