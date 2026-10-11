#include "streamglass/cuda_buffer.hpp"
#include "streamglass/decoder_runtime.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

class BinaryReader final {
public:
    explicit BinaryReader(const std::filesystem::path& path)
        : input_(path, std::ios::binary) {
        if (!input_) {
            throw std::runtime_error("Cannot open decoder fixture: " + path.string());
        }
    }

    template <typename T>
    [[nodiscard]] T scalar() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(value));
        return value;
    }

    template <typename T>
    [[nodiscard]] std::vector<T> vector(const std::size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::vector<T> result(count);
        bytes(result.data(), count * sizeof(T));
        return result;
    }

    void bytes(void* destination, const std::size_t count) {
        if (count == 0) {
            return;
        }
        input_.read(
            static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (input_.gcount() != static_cast<std::streamsize>(count)) {
            throw std::runtime_error("Decoder fixture ended unexpectedly");
        }
    }

    void require_end() {
        char byte = 0;
        if (input_.read(&byte, 1)) {
            throw std::runtime_error("Decoder fixture has trailing bytes");
        }
    }

private:
    std::ifstream input_;
};

struct FeatureMetrics {
    double cosine = 0.0;
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    bool finite = true;
};

FeatureMetrics compare_features(
    const std::span<const __half> actual,
    const std::span<const __half> expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::invalid_argument("Decoder feature comparison shape mismatch");
    }
    long double dot = 0.0L;
    long double actual_norm = 0.0L;
    long double expected_norm = 0.0L;
    long double absolute_sum = 0.0L;
    double maximum = 0.0;
    bool finite = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto left = static_cast<double>(__half2float(actual[index]));
        const auto right = static_cast<double>(__half2float(expected[index]));
        finite = finite && std::isfinite(left) && std::isfinite(right);
        const auto difference = std::abs(left - right);
        maximum = std::max(maximum, difference);
        absolute_sum += difference;
        dot += left * right;
        actual_norm += left * left;
        expected_norm += right * right;
    }
    const auto denominator = std::sqrt(actual_norm * expected_norm);
    return FeatureMetrics{
        .cosine = denominator == 0.0L
            ? 0.0
            : static_cast<double>(dot / denominator),
        .maximum_absolute_error = maximum,
        .mean_absolute_error =
            static_cast<double>(absolute_sum / actual.size()),
        .finite = finite,
    };
}

struct CallReport {
    std::uint32_t call_index = 0;
    std::uint32_t policy_index = 0;
    std::int64_t selected = 0;
    std::uint32_t expected_selected = 0;
    float decision = 0.0F;
    float expected_decision = 0.0F;
    std::size_t cache_length = 0;
    std::uint32_t expected_cache_length = 0;
    bool source_finished = false;
    bool decision_probability_used = true;
    FeatureMetrics features;
    bool passed = false;
};

void write_report(
    const std::filesystem::path& path,
    const std::span<const CallReport> calls,
    const bool passed) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write decoder report: " + path.string());
    }
    output << std::setprecision(10);
    output << "{\n  \"schema_version\": 1,\n";
    output << "  \"component\": \"native_cpp_decoder_official_parity\",\n";
    output << "  \"call_count\": " << calls.size() << ",\n";
    output << "  \"passed\": " << (passed ? "true" : "false") << ",\n";
    output << "  \"calls\": [\n";
    for (std::size_t index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        output << "    {\"call_index\": " << call.call_index
               << ", \"policy_index\": " << call.policy_index
               << ", \"selected_index\": " << call.selected
               << ", \"expected_index\": " << call.expected_selected
               << ", \"decision_probability\": " << call.decision
               << ", \"expected_probability\": " << call.expected_decision
               << ", \"cache_length\": " << call.cache_length
               << ", \"expected_cache_length\": "
               << call.expected_cache_length
               << ", \"source_finished\": "
               << (call.source_finished ? "true" : "false")
               << ", \"decision_probability_used_by_policy\": "
               << (call.decision_probability_used ? "true" : "false")
               << ", \"feature_cosine\": " << call.features.cosine
               << ", \"feature_max_abs\": "
               << call.features.maximum_absolute_error
               << ", \"feature_mean_abs\": "
               << call.features.mean_absolute_error
               << ", \"passed\": " << (call.passed ? "true" : "false")
               << "}" << (index + 1U == calls.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

std::size_t checked_elements(
    const std::size_t first,
    const std::size_t second,
    const std::size_t limit) {
    if (first > limit || second > limit ||
        (second != 0 && first > limit / second)) {
        throw std::runtime_error("Decoder fixture dimensions exceed limits");
    }
    return first * second;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr
                << "usage: streamglass-decoder-replay <prefill.engine> "
                   "<step.engine> <fixture.bin> <report.json>\n";
            return 2;
        }

        BinaryReader reader(argv[3]);
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        constexpr std::array<char, 8> expected_magic{
            'S', 'G', 'D', 'E', 'C', '0', '0', '1'};
        if (magic != expected_magic) {
            throw std::runtime_error("Decoder fixture magic/version mismatch");
        }
        const auto policy_count = reader.scalar<std::uint32_t>();
        const auto declared_call_count = reader.scalar<std::uint32_t>();
        if (policy_count == 0 || policy_count > 10'000 ||
            declared_call_count == 0 || declared_call_count > 100'000) {
            throw std::runtime_error("Decoder fixture count is invalid");
        }

        streamglass::DecoderRuntime runtime(argv[1], argv[2]);
        std::vector<CallReport> reports;
        reports.reserve(declared_call_count);
        bool all_passed = true;

        for (std::uint32_t policy = 0; policy < policy_count; ++policy) {
            const auto policy_index = reader.scalar<std::uint32_t>();
            const auto call_count = reader.scalar<std::uint32_t>();
            const auto encoder_length = reader.scalar<std::uint32_t>();
            static_cast<void>(reader.scalar<std::uint32_t>());
            if (call_count == 0 || encoder_length == 0 ||
                encoder_length > streamglass::DecoderRuntime::kMaximumEncoderFrames) {
                throw std::runtime_error("Decoder fixture policy shape is invalid");
            }
            const auto encoder_elements = checked_elements(
                encoder_length,
                streamglass::DecoderRuntime::kDecoderDimension,
                streamglass::DecoderRuntime::kMaximumEncoderFrames *
                    streamglass::DecoderRuntime::kDecoderDimension);
            const auto host_encoder = reader.vector<__half>(encoder_elements);
            streamglass::DeviceBuffer<__half> device_encoder(encoder_elements);
            streamglass::check_cuda(
                cudaMemcpy(
                    device_encoder.data(),
                    host_encoder.data(),
                    device_encoder.bytes(),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy decoder replay encoder");

            for (std::uint32_t local_call = 0; local_call < call_count;
                 ++local_call) {
                const auto call_index = reader.scalar<std::uint32_t>();
                const auto kind = reader.scalar<std::uint32_t>();
                const auto target_length = reader.scalar<std::uint32_t>();
                const auto position = reader.scalar<std::uint32_t>();
                const auto expected_selected = reader.scalar<std::uint32_t>();
                const auto decoder_length = reader.scalar<std::uint32_t>();
                const auto expected_cache_length = reader.scalar<std::uint32_t>();
                const auto source_finished_raw = reader.scalar<std::uint32_t>();
                const auto expected_decision = reader.scalar<float>();
                if (target_length == 0 ||
                    target_length >
                        streamglass::DecoderRuntime::kMaximumTargetLength ||
                    decoder_length != target_length) {
                    throw std::runtime_error("Decoder fixture call shape is invalid");
                }
                if (source_finished_raw > 1U) {
                    throw std::runtime_error("Decoder source_finished flag is invalid");
                }
                const auto source_finished = source_finished_raw == 1U;
                const auto target = reader.vector<std::int64_t>(target_length);
                const auto feature_elements = checked_elements(
                    decoder_length,
                    streamglass::DecoderRuntime::kDecoderDimension,
                    streamglass::DecoderRuntime::kMaximumTargetLength *
                        streamglass::DecoderRuntime::kDecoderDimension);
                const auto expected_features =
                    reader.vector<__half>(feature_elements);

                streamglass::DecoderResult result;
                if (kind == 0U) {
                    if (local_call != 0U || position != 0U) {
                        throw std::runtime_error("Decoder prefill is out of order");
                    }
                    result = runtime.begin_policy(
                        target, device_encoder.data(), encoder_length);
                } else if (kind == 1U) {
                    if (local_call == 0U || target_length != 1U ||
                        position != runtime.cache_length()) {
                        throw std::runtime_error("Decoder step state is inconsistent");
                    }
                    result = runtime.step(target[0]);
                } else {
                    throw std::runtime_error("Decoder fixture call kind is invalid");
                }

                std::vector<__half> actual_features(feature_elements);
                runtime.copy_result_features(result, actual_features);
                const auto metrics =
                    compare_features(actual_features, expected_features);
                const auto probability_error = std::abs(
                    result.decision_probability - expected_decision);
                const auto decision_probability_used = !source_finished;
                const auto passed =
                    result.selected_index ==
                        static_cast<std::int64_t>(expected_selected) &&
                    (!decision_probability_used || probability_error <= 0.02F) &&
                    result.cache_length == expected_cache_length &&
                    result.feature_count == decoder_length && metrics.finite &&
                    metrics.cosine >= 0.999;
                all_passed = all_passed && passed;
                reports.push_back(CallReport{
                    .call_index = call_index,
                    .policy_index = policy_index,
                    .selected = result.selected_index,
                    .expected_selected = expected_selected,
                    .decision = result.decision_probability,
                    .expected_decision = expected_decision,
                    .cache_length = result.cache_length,
                    .expected_cache_length = expected_cache_length,
                    .source_finished = source_finished,
                    .decision_probability_used = decision_probability_used,
                    .features = metrics,
                    .passed = passed,
                });
                std::cout << "call=" << call_index
                          << " policy=" << policy_index
                          << " token=" << result.selected_index
                          << " decision=" << result.decision_probability
                          << " cosine=" << metrics.cosine
                          << " passed=" << (passed ? "true" : "false")
                          << '\n';
            }
        }

        reader.require_end();
        if (reports.size() != declared_call_count) {
            throw std::runtime_error("Decoder fixture call count mismatch");
        }
        write_report(argv[4], reports, all_passed);
        std::cout << "decoder_calls=" << reports.size()
                  << " passed=" << (all_passed ? "true" : "false") << '\n';
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
