#include "streamglass/cuda_buffer.hpp"
#include "streamglass/t2u_runtime.hpp"
#include "streamglass/tokenizer_assets.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
            throw std::runtime_error("Cannot open T2U fixture: " + path.string());
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
            throw std::runtime_error("T2U fixture ended unexpectedly");
        }
    }

    void require_end() {
        char byte = 0;
        if (input_.read(&byte, 1)) {
            throw std::runtime_error("T2U fixture has trailing bytes");
        }
    }

private:
    std::ifstream input_;
};

const char* action_name(const streamglass::T2uAction action) {
    switch (action) {
        case streamglass::T2uAction::read:
            return "read";
        case streamglass::T2uAction::write:
            return "write";
        case streamglass::T2uAction::finished:
            return "finished";
    }
    return "invalid";
}

struct CallReport {
    std::uint32_t call_index = 0;
    streamglass::T2uAction action = streamglass::T2uAction::read;
    std::uint32_t expected_action = 0;
    std::size_t character_count = 0;
    std::size_t total_unit_count = 0;
    std::size_t unit_offset = 0;
    std::size_t unit_count = 0;
    std::size_t next_duration_start_index = 0;
    bool characters_equal = false;
    bool character_lengths_equal = false;
    bool durations_equal = false;
    bool expansion_equal = false;
    bool full_units_equal = false;
    bool policy_units_equal = false;
    bool passed = false;
};

void write_report(
    const std::filesystem::path& path,
    const std::span<const CallReport> calls,
    const bool passed) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write T2U report: " + path.string());
    }
    output << "{\n  \"schema_version\": 1,\n";
    output << "  \"component\": \"native_cpp_t2u_official_parity\",\n";
    output << "  \"call_count\": " << calls.size() << ",\n";
    output << "  \"passed\": " << (passed ? "true" : "false") << ",\n";
    output << "  \"calls\": [\n";
    for (std::size_t index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        output << "    {\"call_index\": " << call.call_index
               << ", \"action\": \"" << action_name(call.action)
               << "\", \"expected_action\": " << call.expected_action
               << ", \"character_count\": " << call.character_count
               << ", \"total_unit_count\": " << call.total_unit_count
               << ", \"unit_offset\": " << call.unit_offset
               << ", \"unit_count\": " << call.unit_count
               << ", \"next_duration_start_index\": "
               << call.next_duration_start_index
               << ", \"characters_equal\": "
               << (call.characters_equal ? "true" : "false")
               << ", \"character_lengths_equal\": "
               << (call.character_lengths_equal ? "true" : "false")
               << ", \"durations_equal\": "
               << (call.durations_equal ? "true" : "false")
               << ", \"expansion_equal\": "
               << (call.expansion_equal ? "true" : "false")
               << ", \"full_units_equal\": "
               << (call.full_units_equal ? "true" : "false")
               << ", \"policy_units_equal\": "
               << (call.policy_units_equal ? "true" : "false")
               << ", \"passed\": " << (call.passed ? "true" : "false")
               << "}" << (index + 1U == calls.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

template <typename T>
bool equal_span(const std::span<const T> left, const std::span<const T> right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            std::cerr
                << "usage: streamglass-t2u-replay <frontend.engine> "
                   "<decoder.engine> <runtime-assets> <fixture.bin> "
                   "<report.json>\n";
            return 2;
        }

        BinaryReader reader(argv[4]);
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        constexpr std::array<char, 8> expected_magic{
            'S', 'G', 'T', '2', 'U', '0', '0', '1'};
        if (magic != expected_magic) {
            throw std::runtime_error("T2U fixture magic/version mismatch");
        }
        const auto call_count = reader.scalar<std::uint32_t>();
        static_cast<void>(reader.scalar<std::uint32_t>());
        if (call_count == 0 || call_count > 10'000) {
            throw std::runtime_error("T2U fixture call count is invalid");
        }

        streamglass::TokenizerAssets tokenizer(argv[3]);
        streamglass::T2uRuntime runtime(argv[1], argv[2], tokenizer);
        std::vector<CallReport> reports;
        reports.reserve(call_count);
        std::size_t duration_cursor = 0;
        bool all_passed = true;

        for (std::uint32_t call_number = 0; call_number < call_count;
             ++call_number) {
            const auto call_index = reader.scalar<std::uint32_t>();
            const auto source_finished_raw = reader.scalar<std::uint32_t>();
            const auto text_length = reader.scalar<std::uint32_t>();
            const auto character_count = reader.scalar<std::uint32_t>();
            const auto expected_cursor_before = reader.scalar<std::uint32_t>();
            const auto expected_action = reader.scalar<std::uint32_t>();
            const auto total_unit_count = reader.scalar<std::uint32_t>();
            const auto expected_unit_offset = reader.scalar<std::uint32_t>();
            const auto expected_policy_unit_count = reader.scalar<std::uint32_t>();
            const auto expected_cursor_after = reader.scalar<std::uint32_t>();
            static_cast<void>(reader.scalar<std::uint32_t>());
            static_cast<void>(reader.scalar<std::uint32_t>());
            if (source_finished_raw > 1U || text_length < 2U ||
                text_length > streamglass::T2uRuntime::kMaximumTextLength ||
                character_count < 2U ||
                character_count >
                    streamglass::T2uRuntime::kMaximumCharacterCount ||
                total_unit_count == 0U ||
                total_unit_count > streamglass::T2uRuntime::kMaximumUnitCount ||
                expected_cursor_before != duration_cursor ||
                expected_action > 2U ||
                expected_unit_offset + expected_policy_unit_count >
                    total_unit_count) {
                throw std::runtime_error("T2U fixture call metadata is invalid");
            }

            const auto tokens = reader.vector<std::int64_t>(text_length);
            const auto feature_elements =
                static_cast<std::size_t>(text_length) *
                streamglass::T2uRuntime::kFeatureDimension;
            const auto host_features = reader.vector<__half>(feature_elements);
            const auto expected_characters =
                reader.vector<std::int64_t>(character_count);
            const auto expected_character_lengths =
                reader.vector<std::int64_t>(text_length);
            const auto expected_durations =
                reader.vector<std::int64_t>(character_count);
            const auto expected_full_units =
                reader.vector<std::int32_t>(total_unit_count);
            const auto expected_policy_units =
                reader.vector<std::int32_t>(expected_policy_unit_count);

            streamglass::DeviceBuffer<__half> device_features(feature_elements);
            streamglass::check_cuda(
                cudaMemcpy(
                    device_features.data(),
                    host_features.data(),
                    device_features.bytes(),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy T2U replay features");
            const auto result = runtime.run(
                tokens,
                device_features.data(),
                text_length,
                duration_cursor,
                source_finished_raw == 1U,
                50);

            std::vector<std::int64_t> actual_durations(character_count);
            runtime.copy_durations(actual_durations, character_count);
            std::vector<std::int32_t> actual_units(total_unit_count);
            runtime.copy_unit_ids(actual_units, total_unit_count);
            std::vector<__half> actual_character_features(
                static_cast<std::size_t>(character_count) *
                streamglass::T2uRuntime::kFeatureDimension);
            runtime.copy_character_features(
                actual_character_features, character_count);
            std::vector<__half> actual_expanded_features(
                static_cast<std::size_t>(total_unit_count) *
                streamglass::T2uRuntime::kFeatureDimension);
            runtime.copy_expanded_features(
                actual_expanded_features, total_unit_count);
            std::vector<__half> expected_expanded_features;
            expected_expanded_features.reserve(actual_expanded_features.size());
            for (std::size_t character = 0; character < character_count;
                 ++character) {
                const auto begin = actual_character_features.begin() +
                    static_cast<std::ptrdiff_t>(
                        character * streamglass::T2uRuntime::kFeatureDimension);
                const auto end = begin + static_cast<std::ptrdiff_t>(
                    streamglass::T2uRuntime::kFeatureDimension);
                for (std::int64_t repeat = 0;
                     repeat < actual_durations[character]; ++repeat) {
                    expected_expanded_features.insert(
                        expected_expanded_features.end(), begin, end);
                }
            }
            const auto expansion_equal =
                expected_expanded_features.size() ==
                    actual_expanded_features.size() &&
                std::memcmp(
                    expected_expanded_features.data(),
                    actual_expanded_features.data(),
                    actual_expanded_features.size() * sizeof(__half)) == 0;
            const auto characters_equal = equal_span<std::int64_t>(
                runtime.last_character_indices(), expected_characters);
            const auto character_lengths_equal = equal_span<std::int64_t>(
                runtime.last_character_lengths(), expected_character_lengths);
            const auto durations_equal = actual_durations == expected_durations;
            const auto full_units_equal = actual_units == expected_full_units;
            const auto policy_units_equal = equal_span<std::int32_t>(
                std::span<const std::int32_t>(actual_units).subspan(
                    result.unit_offset, result.unit_count),
                expected_policy_units);
            const auto passed =
                static_cast<std::uint32_t>(result.action) == expected_action &&
                result.character_count == character_count &&
                result.total_unit_count == total_unit_count &&
                result.unit_offset == expected_unit_offset &&
                result.unit_count == expected_policy_unit_count &&
                result.next_duration_start_index == expected_cursor_after &&
                characters_equal && character_lengths_equal && durations_equal &&
                expansion_equal && full_units_equal && policy_units_equal;
            all_passed = all_passed && passed;
            duration_cursor = result.next_duration_start_index;
            reports.push_back(CallReport{
                .call_index = call_index,
                .action = result.action,
                .expected_action = expected_action,
                .character_count = result.character_count,
                .total_unit_count = result.total_unit_count,
                .unit_offset = result.unit_offset,
                .unit_count = result.unit_count,
                .next_duration_start_index =
                    result.next_duration_start_index,
                .characters_equal = characters_equal,
                .character_lengths_equal = character_lengths_equal,
                .durations_equal = durations_equal,
                .expansion_equal = expansion_equal,
                .full_units_equal = full_units_equal,
                .policy_units_equal = policy_units_equal,
                .passed = passed,
            });
            std::cout << "call=" << call_index
                      << " action=" << action_name(result.action)
                      << " chars=" << result.character_count
                      << " units=" << result.total_unit_count
                      << " emit=" << result.unit_count
                      << " passed=" << (passed ? "true" : "false") << '\n';
        }

        reader.require_end();
        write_report(argv[5], reports, all_passed);
        std::cout << "t2u_calls=" << reports.size()
                  << " passed=" << (all_passed ? "true" : "false") << '\n';
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
