#include "streamglass/silero_vad_trt.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
            throw std::runtime_error("Cannot open Silero fixture: " + path.string());
        }
    }

    template <typename T>
    [[nodiscard]] T scalar() {
        static_assert(std::is_trivially_copyable_v<T>);
        T result{};
        bytes(&result, sizeof(result));
        return result;
    }

    template <typename T>
    [[nodiscard]] std::vector<T> vector(const std::size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::vector<T> result(count);
        bytes(result.data(), result.size() * sizeof(T));
        return result;
    }

    void bytes(void* destination, const std::size_t count) {
        input_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (input_.gcount() != static_cast<std::streamsize>(count)) {
            throw std::runtime_error("Silero fixture ended unexpectedly");
        }
    }

    void require_end() {
        char trailing = 0;
        if (input_.read(&trailing, 1)) {
            throw std::runtime_error("Silero fixture has trailing bytes");
        }
    }

private:
    std::ifstream input_;
};

struct FixtureChunk {
    std::array<float, streamglass::SileroVadTrt::kChunkSamples> samples{};
    float expected = 0.0F;
};

struct Report {
    std::size_t chunks = 0;
    std::size_t repetitions = 0;
    std::size_t classification_mismatches = 0;
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double load_ms = 0.0;
    double latency_mean_ms = 0.0;
    double latency_p50_ms = 0.0;
    double latency_p95_ms = 0.0;
    double latency_p99_ms = 0.0;
    double latency_max_ms = 0.0;
    bool finite = true;
    bool passed = false;
};

std::vector<FixtureChunk> read_fixture(const std::filesystem::path& path) {
    BinaryReader reader(path);
    std::array<char, 8> magic{};
    reader.bytes(magic.data(), magic.size());
    constexpr std::array<char, 8> expected_magic{
        'S', 'G', 'V', 'A', 'D', '0', '0', '1'};
    if (magic != expected_magic) {
        throw std::runtime_error("Silero fixture magic/version mismatch");
    }
    const auto sample_rate = reader.scalar<std::uint32_t>();
    const auto chunk_samples = reader.scalar<std::uint32_t>();
    const auto chunk_count = reader.scalar<std::uint32_t>();
    static_cast<void>(reader.scalar<std::uint32_t>());
    if (sample_rate != streamglass::SileroVadTrt::kSampleRate ||
        chunk_samples != streamglass::SileroVadTrt::kChunkSamples ||
        chunk_count == 0 || chunk_count > 1'000'000) {
        throw std::runtime_error("Silero fixture header is invalid");
    }
    std::vector<FixtureChunk> chunks;
    chunks.reserve(chunk_count);
    for (std::uint32_t index = 0; index < chunk_count; ++index) {
        FixtureChunk chunk;
        const auto samples = reader.vector<float>(chunk_samples);
        std::copy(samples.begin(), samples.end(), chunk.samples.begin());
        chunk.expected = reader.scalar<float>();
        chunks.push_back(chunk);
    }
    reader.require_end();
    return chunks;
}

double percentile(const std::span<const double> sorted, const double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const auto scaled = fraction * static_cast<double>(sorted.size() - 1U);
    const auto index = static_cast<std::size_t>(std::ceil(scaled));
    return sorted[index];
}

void write_report(const std::filesystem::path& path, const Report& report) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write Silero report: " + path.string());
    }
    output << std::setprecision(10)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"component\": \"silero_vad_v6.2.1_tensorrt_experiment\",\n"
           << "  \"chunks\": " << report.chunks << ",\n"
           << "  \"repetitions\": " << report.repetitions << ",\n"
           << "  \"classification_mismatches\": "
           << report.classification_mismatches << ",\n"
           << "  \"max_abs\": " << report.maximum_absolute_error << ",\n"
           << "  \"mean_abs\": " << report.mean_absolute_error << ",\n"
           << "  \"load_ms\": " << report.load_ms << ",\n"
           << "  \"latency_mean_ms\": " << report.latency_mean_ms << ",\n"
           << "  \"latency_p50_ms\": " << report.latency_p50_ms << ",\n"
           << "  \"latency_p95_ms\": " << report.latency_p95_ms << ",\n"
           << "  \"latency_p99_ms\": " << report.latency_p99_ms << ",\n"
           << "  \"latency_max_ms\": " << report.latency_max_ms << ",\n"
           << "  \"finite\": " << (report.finite ? "true" : "false") << ",\n"
           << "  \"passed\": " << (report.passed ? "true" : "false") << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "usage: streamglass-silero-vad-probe <engine> "
                         "<fixture> <report.json> [repetitions]\n";
            return 2;
        }
        const auto repetitions = argc == 5 ? std::stoul(argv[4]) : 100UL;
        if (repetitions == 0 || repetitions > 100'000) {
            throw std::invalid_argument("Silero probe repetitions are invalid");
        }
        const auto chunks = read_fixture(argv[2]);
        const auto load_begin = std::chrono::steady_clock::now();
        streamglass::SileroVadTrt runtime(argv[1]);
        const auto load_end = std::chrono::steady_clock::now();

        runtime.reset();
        long double absolute_sum = 0.0L;
        double maximum_absolute_error = 0.0;
        std::size_t classification_mismatches = 0;
        bool finite = true;
        for (const auto& chunk : chunks) {
            const auto actual = runtime.run(chunk.samples);
            const auto difference =
                std::abs(static_cast<double>(actual - chunk.expected));
            absolute_sum += difference;
            maximum_absolute_error = std::max(maximum_absolute_error, difference);
            finite = finite && std::isfinite(actual);
            classification_mismatches +=
                (actual >= 0.5F) != (chunk.expected >= 0.5F) ? 1U : 0U;
        }

        std::vector<double> latencies;
        latencies.reserve(chunks.size() * repetitions);
        long double latency_sum = 0.0L;
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            runtime.reset();
            for (const auto& chunk : chunks) {
                const auto begin = std::chrono::steady_clock::now();
                static_cast<void>(runtime.run(chunk.samples));
                const auto end = std::chrono::steady_clock::now();
                const auto elapsed =
                    std::chrono::duration<double, std::milli>(end - begin).count();
                latencies.push_back(elapsed);
                latency_sum += elapsed;
            }
        }
        std::sort(latencies.begin(), latencies.end());
        const auto mean_absolute_error =
            static_cast<double>(absolute_sum / chunks.size());
        Report report{
            .chunks = chunks.size(),
            .repetitions = repetitions,
            .classification_mismatches = classification_mismatches,
            .maximum_absolute_error = maximum_absolute_error,
            .mean_absolute_error = mean_absolute_error,
            .load_ms = std::chrono::duration<double, std::milli>(load_end - load_begin)
                           .count(),
            .latency_mean_ms =
                static_cast<double>(latency_sum / latencies.size()),
            .latency_p50_ms = percentile(latencies, 0.50),
            .latency_p95_ms = percentile(latencies, 0.95),
            .latency_p99_ms = percentile(latencies, 0.99),
            .latency_max_ms = latencies.back(),
            .finite = finite,
            .passed = finite && classification_mismatches == 0 &&
                      maximum_absolute_error <= 0.02 &&
                      mean_absolute_error <= 0.005,
        };
        write_report(argv[3], report);
        std::cout << std::fixed << std::setprecision(6)
                  << "chunks=" << report.chunks
                  << " max_abs=" << report.maximum_absolute_error
                  << " mean_abs=" << report.mean_absolute_error
                  << " mismatches=" << report.classification_mismatches
                  << " p50_ms=" << report.latency_p50_ms
                  << " p95_ms=" << report.latency_p95_ms
                  << " p99_ms=" << report.latency_p99_ms
                  << " passed=" << (report.passed ? "true" : "false") << '\n';
        return report.passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
