#include "streamglass/cuda_buffer.hpp"
#include "streamglass/cuda_kernels.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_invalid_argument(Callable&& callable, const char* message) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    constexpr std::array<std::int64_t, 3> durations{3, 5, 7};
    streamglass::DeviceBuffer<std::int64_t> device_durations(durations.size());
    streamglass::DeviceBuffer<std::int64_t> offsets(durations.size() + 1U);
    streamglass::DeviceBuffer<std::int64_t> total(1);
    streamglass::DeviceBuffer<std::int64_t> output_range(2);
    streamglass::DeviceBuffer<std::int32_t> output_state(2);
    streamglass::CudaStream stream;

    streamglass::check_cuda(
        cudaMemcpy(
            device_durations.data(),
            durations.data(),
            device_durations.bytes(),
            cudaMemcpyHostToDevice),
        "cudaMemcpy test durations");
    streamglass::launch_duration_prefix_sum(
        device_durations.data(),
        durations.size(),
        offsets.data(),
        total.data(),
        stream.get());

    // The official final policy permits a cursor exactly at the end. This is
    // the state reached when a provisional trailing character disappears.
    streamglass::launch_t2u_policy_slice(
        offsets.data(),
        durations.size(),
        durations.size(),
        true,
        50,
        output_range.data(),
        output_state.data(),
        stream.get());
    stream.synchronize();

    std::array<std::int64_t, 2> range{};
    std::array<std::int32_t, 2> state{};
    std::int64_t total_units = 0;
    streamglass::check_cuda(
        cudaMemcpy(
            range.data(),
            output_range.data(),
            output_range.bytes(),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy test output range");
    streamglass::check_cuda(
        cudaMemcpy(
            state.data(),
            output_state.data(),
            output_state.bytes(),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy test output state");
    streamglass::check_cuda(
        cudaMemcpy(
            &total_units,
            total.data(),
            sizeof(total_units),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy test total units");

    require(total_units == 15, "duration prefix total changed");
    require(range[0] == total_units, "terminal slice offset is not at the end");
    require(range[1] == 0, "terminal slice emitted duplicate units");
    require(state[0] == 2, "terminal slice did not return FINISHED");
    require(
        state[1] == static_cast<std::int32_t>(durations.size()),
        "terminal slice changed the completed cursor");

    require_invalid_argument(
        [&] {
            streamglass::launch_t2u_policy_slice(
                offsets.data(),
                durations.size(),
                durations.size(),
                false,
                50,
                output_range.data(),
                output_state.data(),
                stream.get());
        },
        "non-final terminal cursor was accepted");
    require_invalid_argument(
        [&] {
            streamglass::launch_t2u_policy_slice(
                offsets.data(),
                durations.size(),
                durations.size() + 1U,
                true,
                50,
                output_range.data(),
                output_state.data(),
                stream.get());
        },
        "out-of-range terminal cursor was accepted");

    std::cout << "T2U terminal cursor policy tests passed\n";
    return 0;
}
