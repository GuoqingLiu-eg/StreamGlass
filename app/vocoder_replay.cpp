#include "streamglass/cuda_buffer.hpp"
#include "streamglass/vocoder_runtime.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
            throw std::runtime_error("Cannot open vocoder fixture: " + path.string());
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
            throw std::runtime_error("Vocoder fixture ended unexpectedly");
        }
    }

    void require_end() {
        char byte = 0;
        if (input_.read(&byte, 1)) {
            throw std::runtime_error("Vocoder fixture has trailing bytes");
        }
    }

private:
    std::ifstream input_;
};

struct Metrics {
    double cosine = 0.0;
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double reference_rms = 0.0;
    double candidate_rms = 0.0;
    bool finite = true;
};

Metrics compare_waveforms(
    const std::span<const __half> actual,
    const std::span<const __half> expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::invalid_argument("Vocoder waveform comparison shape mismatch");
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
    return Metrics{
        .cosine = denominator == 0.0L
            ? 1.0
            : static_cast<double>(dot / denominator),
        .maximum_absolute_error = maximum,
        .mean_absolute_error =
            static_cast<double>(absolute_sum / actual.size()),
        .reference_rms =
            std::sqrt(static_cast<double>(expected_norm / actual.size())),
        .candidate_rms =
            std::sqrt(static_cast<double>(actual_norm / actual.size())),
        .finite = finite,
    };
}

struct CaseReport {
    std::size_t unit_count = 0;
    std::size_t sample_count = 0;
    Metrics metrics;
    bool playback_conversion_equal = false;
    bool passed = false;
};

void write_report(
    const std::filesystem::path& path,
    const std::span<const CaseReport> cases,
    const bool passed) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write vocoder report: " + path.string());
    }
    output << std::setprecision(10);
    output << "{\n  \"schema_version\": 1,\n";
    output << "  \"component\": \"native_cpp_vocoder_official_parity\",\n";
    output << "  \"case_count\": " << cases.size() << ",\n";
    output << "  \"passed\": " << (passed ? "true" : "false") << ",\n";
    output << "  \"cases\": [\n";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& item = cases[index];
        output << "    {\"unit_count\": " << item.unit_count
               << ", \"sample_count\": " << item.sample_count
               << ", \"cosine\": " << item.metrics.cosine
               << ", \"max_abs\": "
               << item.metrics.maximum_absolute_error
               << ", \"mean_abs\": "
               << item.metrics.mean_absolute_error
               << ", \"reference_rms\": " << item.metrics.reference_rms
               << ", \"candidate_rms\": " << item.metrics.candidate_rms
               << ", \"finite\": "
               << (item.metrics.finite ? "true" : "false")
               << ", \"playback_conversion_equal\": "
               << (item.playback_conversion_equal ? "true" : "false")
               << ", \"passed\": " << (item.passed ? "true" : "false")
               << "}" << (index + 1U == cases.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr
                << "usage: streamglass-vocoder-replay <vocoder.engine> "
                   "<fixture.bin> <report.json>\n";
            return 2;
        }

        BinaryReader reader(argv[2]);
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        constexpr std::array<char, 8> expected_magic{
            'S', 'G', 'V', 'O', 'C', '0', '0', '1'};
        if (magic != expected_magic) {
            throw std::runtime_error("Vocoder fixture magic/version mismatch");
        }
        const auto case_count = reader.scalar<std::uint32_t>();
        static_cast<void>(reader.scalar<std::uint32_t>());
        if (case_count == 0 || case_count > 1'000) {
            throw std::runtime_error("Vocoder fixture case count is invalid");
        }

        streamglass::VocoderRuntime runtime(argv[1]);
        std::vector<CaseReport> reports;
        reports.reserve(case_count);
        bool all_passed = true;

        for (std::uint32_t index = 0; index < case_count; ++index) {
            const auto unit_count = reader.scalar<std::uint32_t>();
            const auto sample_count = reader.scalar<std::uint32_t>();
            const auto language_id = reader.scalar<std::int64_t>();
            const auto speaker_id = reader.scalar<std::int64_t>();
            if (unit_count < streamglass::VocoderRuntime::kMinimumUnitCount ||
                unit_count > streamglass::VocoderRuntime::kMaximumUnitCount ||
                sample_count !=
                    unit_count * streamglass::VocoderRuntime::kSamplesPerUnit) {
                throw std::runtime_error("Vocoder fixture shape is invalid");
            }
            const auto units = reader.vector<std::int64_t>(unit_count);
            const auto expected_waveform = reader.vector<__half>(sample_count);
            streamglass::DeviceBuffer<std::int64_t> device_units(unit_count);
            streamglass::check_cuda(
                cudaMemcpy(
                    device_units.data(),
                    units.data(),
                    device_units.bytes(),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy vocoder replay units");

            const auto result = runtime.run(
                device_units.data(), unit_count, language_id, speaker_id);
            std::vector<__half> actual_waveform(sample_count);
            runtime.copy_waveform(result, actual_waveform);
            const auto metrics =
                compare_waveforms(actual_waveform, expected_waveform);
            const auto playback = runtime.materialize_for_playback(result);
            auto playback_equal = playback.size() == expected_waveform.size();
            for (std::size_t sample = 0;
                 playback_equal && sample < playback.size(); ++sample) {
                playback_equal =
                    playback[sample] == __half2float(actual_waveform[sample]);
            }
            const auto passed = result.sample_count == sample_count &&
                                metrics.finite && metrics.cosine >= 0.999 &&
                                playback_equal;
            all_passed = all_passed && passed;
            reports.push_back(CaseReport{
                .unit_count = unit_count,
                .sample_count = result.sample_count,
                .metrics = metrics,
                .playback_conversion_equal = playback_equal,
                .passed = passed,
            });
            std::cout << "units=" << unit_count
                      << " samples=" << result.sample_count
                      << " cosine=" << metrics.cosine
                      << " passed=" << (passed ? "true" : "false") << '\n';
        }

        reader.require_end();
        write_report(argv[3], reports, all_passed);
        std::cout << "vocoder_cases=" << reports.size()
                  << " passed=" << (all_passed ? "true" : "false") << '\n';
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
