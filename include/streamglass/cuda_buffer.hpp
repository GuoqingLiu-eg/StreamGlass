#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace streamglass {

inline void check_cuda(
    const cudaError_t error, const std::string_view operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

class CudaStream final {
public:
    CudaStream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const {
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class DeviceBuffer final {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(const std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = std::exchange(other.pointer_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(const std::size_t count) {
        if (pointer_ != nullptr) {
            throw std::logic_error("CUDA device buffer is already allocated");
        }
        if (count == 0) {
            return;
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
            "cudaMalloc");
        count_ = count;
    }

    [[nodiscard]] T* data() noexcept { return pointer_; }
    [[nodiscard]] const T* data() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }

private:
    void release() noexcept {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
            pointer_ = nullptr;
            count_ = 0;
        }
    }

    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
class MappedHostBuffer final {
public:
    explicit MappedHostBuffer(const std::size_t count) : count_(count) {
        if (count == 0) {
            return;
        }
        check_cuda(
            cudaHostAlloc(
                reinterpret_cast<void**>(&host_),
                count * sizeof(T),
                cudaHostAllocMapped),
            "cudaHostAlloc(mapped)");
        check_cuda(
            cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&device_), host_, 0),
            "cudaHostGetDevicePointer");
    }

    ~MappedHostBuffer() {
        if (host_ != nullptr) {
            static_cast<void>(cudaFreeHost(host_));
        }
    }

    MappedHostBuffer(const MappedHostBuffer&) = delete;
    MappedHostBuffer& operator=(const MappedHostBuffer&) = delete;
    MappedHostBuffer(MappedHostBuffer&&) = delete;
    MappedHostBuffer& operator=(MappedHostBuffer&&) = delete;

    [[nodiscard]] T* data() noexcept { return host_; }
    [[nodiscard]] const T* data() const noexcept { return host_; }
    [[nodiscard]] T* device_data() noexcept { return device_; }
    [[nodiscard]] const T* device_data() const noexcept { return device_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::span<T> span() noexcept { return {host_, count_}; }

private:
    T* host_ = nullptr;
    T* device_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace streamglass
