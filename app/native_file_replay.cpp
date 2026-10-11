#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/decoder_runtime.hpp"
#include "streamglass/encoder_runtime.hpp"
#include "streamglass/online_fbank.hpp"
#include "streamglass/streaming_text_policy.hpp"
#include "streamglass/t2u_runtime.hpp"
#include "streamglass/tokenizer_assets.hpp"
#include "streamglass/vocoder_runtime.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSegmentSamples = 5'120;
constexpr std::size_t kMinimumStartingFbankFrames = 192;

class BinaryReader final {
public:
    explicit BinaryReader(const std::filesystem::path& path)
        : input_(path, std::ios::binary) {
        if (!input_) {
            throw std::runtime_error(
                "Cannot open native pipeline fixture: " + path.string());
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
            throw std::runtime_error("Native pipeline fixture ended unexpectedly");
        }
    }

    void require_end() {
        char byte = 0;
        if (input_.read(&byte, 1)) {
            throw std::runtime_error("Native pipeline fixture has trailing bytes");
        }
    }

private:
    std::ifstream input_;
};

struct ExpectedPolicy {
    std::size_t fbank_valid = 0;
    std::size_t fbank_padded = 0;
    std::size_t encoder_frames = 0;
    bool source_finished = false;
    std::uint32_t text_action = 0;
    bool text_finished = false;
    bool has_t2u = false;
    std::uint32_t t2u_action = 0;
    bool t2u_finished = false;
    std::size_t duration_cursor = 0;
    std::vector<std::int64_t> target_after;
    std::vector<std::int64_t> selected;
    std::vector<float> probabilities;
    std::vector<std::int32_t> unit_ids;
};

std::vector<ExpectedPolicy> read_fixture(const std::filesystem::path& path) {
    BinaryReader reader(path);
    std::array<char, 8> magic{};
    reader.bytes(magic.data(), magic.size());
    constexpr std::array<char, 8> expected_magic{
        'S', 'G', 'P', 'I', 'P', '0', '0', '1'};
    if (magic != expected_magic) {
        throw std::runtime_error("Native pipeline fixture magic/version mismatch");
    }
    const auto policy_count = reader.scalar<std::uint32_t>();
    static_cast<void>(reader.scalar<std::uint32_t>());
    if (policy_count == 0 || policy_count > 1'000) {
        throw std::runtime_error("Native pipeline policy count is invalid");
    }

    std::vector<ExpectedPolicy> result;
    result.reserve(policy_count);
    for (std::uint32_t index = 0; index < policy_count; ++index) {
        const auto fbank_valid = reader.scalar<std::uint32_t>();
        const auto fbank_padded = reader.scalar<std::uint32_t>();
        const auto encoder_frames = reader.scalar<std::uint32_t>();
        const auto source_finished = reader.scalar<std::uint32_t>();
        const auto text_action = reader.scalar<std::uint32_t>();
        const auto text_finished = reader.scalar<std::uint32_t>();
        const auto target_count = reader.scalar<std::uint32_t>();
        const auto selected_count = reader.scalar<std::uint32_t>();
        const auto has_t2u = reader.scalar<std::uint32_t>();
        const auto t2u_action = reader.scalar<std::uint32_t>();
        const auto t2u_finished = reader.scalar<std::uint32_t>();
        const auto duration_cursor = reader.scalar<std::uint32_t>();
        const auto unit_count = reader.scalar<std::uint32_t>();
        if (source_finished > 1U || text_action > 1U || text_finished > 1U ||
            has_t2u > 1U || t2u_action > 2U || t2u_finished > 1U) {
            throw std::runtime_error("Native pipeline fixture flags are invalid");
        }
        result.push_back(ExpectedPolicy{
            .fbank_valid = fbank_valid,
            .fbank_padded = fbank_padded,
            .encoder_frames = encoder_frames,
            .source_finished = source_finished == 1U,
            .text_action = text_action,
            .text_finished = text_finished == 1U,
            .has_t2u = has_t2u == 1U,
            .t2u_action = t2u_action,
            .t2u_finished = t2u_finished == 1U,
            .duration_cursor = duration_cursor,
            .target_after = reader.vector<std::int64_t>(target_count),
            .selected = reader.vector<std::int64_t>(selected_count),
            .probabilities = reader.vector<float>(selected_count),
            .unit_ids = reader.vector<std::int32_t>(unit_count),
        });
    }
    reader.require_end();
    return result;
}

std::vector<float> read_waveform(const std::filesystem::path& path) {
    unsigned int channels = 0;
    unsigned int sample_rate = 0;
    drwav_uint64 frame_count = 0;
    float* interleaved = drwav_open_file_and_read_pcm_frames_f32(
        path.c_str(), &channels, &sample_rate, &frame_count, nullptr);
    if (interleaved == nullptr) {
        throw std::runtime_error("Cannot decode WAV: " + path.string());
    }
    if (channels != 1 ||
        sample_rate != streamglass::OfficialOnlineFbank::kSampleRate) {
        drwav_free(interleaved, nullptr);
        throw std::runtime_error("WAV must be mono 16kHz PCM");
    }
    if (frame_count > std::numeric_limits<std::size_t>::max()) {
        drwav_free(interleaved, nullptr);
        throw std::length_error("WAV is too large for this process");
    }
    std::vector<float> waveform(
        interleaved, interleaved + static_cast<std::size_t>(frame_count));
    drwav_free(interleaved, nullptr);
    return waveform;
}

void write_waveform(
    const std::filesystem::path& path, const std::span<const float> waveform) {
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = streamglass::VocoderRuntime::kSampleRate;
    format.bitsPerSample = 32;
    drwav writer{};
    if (!drwav_init_file_write(&writer, path.c_str(), &format, nullptr)) {
        throw std::runtime_error("Cannot create output WAV: " + path.string());
    }
    const auto written =
        drwav_write_pcm_frames(&writer, waveform.size(), waveform.data());
    drwav_uninit(&writer);
    if (written != waveform.size()) {
        throw std::runtime_error("Output WAV write was incomplete");
    }
}

template <typename Function>
double timed_milliseconds(Function&& function) {
    const auto begin = std::chrono::steady_clock::now();
    std::forward<Function>(function)();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <typename T>
bool equal_span(const std::span<const T> left, const std::span<const T> right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

template <typename T>
std::size_t matching_elements(
    const std::span<const T> actual, const std::span<const T> expected) {
    const auto common = std::min(actual.size(), expected.size());
    return static_cast<std::size_t>(std::count_if(
        actual.begin(),
        actual.begin() + static_cast<std::ptrdiff_t>(common),
        [expected, index = std::size_t{0}](const T& value) mutable {
            return value == expected[index++];
        }));
}

template <typename T>
std::string first_difference(
    const std::span<const T> actual, const std::span<const T> expected) {
    const auto common = std::min(actual.size(), expected.size());
    for (std::size_t index = 0; index < common; ++index) {
        if (actual[index] != expected[index]) {
            return "index=" + std::to_string(index) +
                   " actual=" + std::to_string(actual[index]) +
                   " expected=" + std::to_string(expected[index]);
        }
    }
    if (actual.size() != expected.size()) {
        return "size actual=" + std::to_string(actual.size()) +
               " expected=" + std::to_string(expected.size());
    }
    return "none";
}

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const auto character : value) {
        switch (character) {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result.push_back(character);
            break;
        }
    }
    return result;
}

std::string environment_or(
    const char* name, const std::string& fallback) {
    const auto* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : value;
}

struct PolicyReport {
    std::size_t index = 0;
    std::size_t fbank_frames = 0;
    std::size_t encoder_frames = 0;
    std::uint32_t text_action = 0;
    bool text_finished = false;
    std::uint32_t t2u_action = 0;
    bool t2u_finished = false;
    std::size_t emitted_tokens = 0;
    std::size_t emitted_units = 0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    double t2u_ms = 0.0;
    double vocoder_ms = 0.0;
    double probability_max_abs = 0.0;
    std::size_t matching_units = 0;
    std::size_t reference_units = 0;
    bool semantic_passed = false;
    bool strict_unit_passed = false;
};

void write_report(
    const std::filesystem::path& path,
    const std::span<const PolicyReport> policies,
    const std::string& translation,
    const std::string& first_semantic_divergence,
    const std::string& first_strict_unit_divergence,
    const std::size_t input_samples,
    const std::size_t output_samples,
    const std::size_t matching_units,
    const std::size_t reference_units,
    const bool policy_semantic_passed,
    const bool strict_unit_parity_passed) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write pipeline report: " + path.string());
    }
    output << std::setprecision(10);
    output << "{\n  \"schema_version\": 1,\n";
    output << "  \"component\": \"native_cpp_s0_file_pipeline\",\n";
    output << "  \"passed\": "
           << (policy_semantic_passed ? "true" : "false") << ",\n";
    output << "  \"policy_semantic_passed\": "
           << (policy_semantic_passed ? "true" : "false") << ",\n";
    output << "  \"strict_unit_parity_passed\": "
           << (strict_unit_parity_passed ? "true" : "false") << ",\n";
    output << "  \"matching_units\": " << matching_units << ",\n";
    output << "  \"reference_units\": " << reference_units << ",\n";
    output << "  \"first_semantic_divergence\": \""
           << json_escape(first_semantic_divergence) << "\",\n";
    output << "  \"first_strict_unit_divergence\": \""
           << json_escape(first_strict_unit_divergence) << "\",\n";
    output << "  \"translation\": \"" << json_escape(translation)
           << "\",\n";
    output << "  \"input_samples\": " << input_samples << ",\n";
    output << "  \"output_samples\": " << output_samples << ",\n";
    output << "  \"policies\": [\n";
    for (std::size_t index = 0; index < policies.size(); ++index) {
        const auto& item = policies[index];
        output << "    {\"index\": " << item.index
               << ", \"fbank_frames\": " << item.fbank_frames
               << ", \"encoder_frames\": " << item.encoder_frames
               << ", \"text_action\": " << item.text_action
               << ", \"text_finished\": "
               << (item.text_finished ? "true" : "false")
               << ", \"t2u_action\": " << item.t2u_action
               << ", \"t2u_finished\": "
               << (item.t2u_finished ? "true" : "false")
               << ", \"emitted_tokens\": " << item.emitted_tokens
               << ", \"emitted_units\": " << item.emitted_units
               << ", \"encoder_ms\": " << item.encoder_ms
               << ", \"decoder_ms\": " << item.decoder_ms
               << ", \"t2u_ms\": " << item.t2u_ms
               << ", \"vocoder_ms\": " << item.vocoder_ms
               << ", \"probability_max_abs\": "
               << item.probability_max_abs
               << ", \"matching_units\": " << item.matching_units
               << ", \"reference_units\": " << item.reference_units
               << ", \"semantic_passed\": "
               << (item.semantic_passed ? "true" : "false")
               << ", \"strict_unit_passed\": "
               << (item.strict_unit_passed ? "true" : "false")
               << "}" << (index + 1U == policies.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 8) {
            std::cerr
                << "usage: streamglass-native-file-replay <input.wav> "
                   "<engine-dir> <runtime-assets> <target-language> "
                   "<fixture.bin> <output.wav> <report.json>\n";
            return 2;
        }

        const std::filesystem::path input_path(argv[1]);
        const std::filesystem::path engine_dir(argv[2]);
        const std::filesystem::path runtime_assets(argv[3]);
        const std::string target_language(argv[4]);
        const std::filesystem::path fixture_path(argv[5]);
        const std::filesystem::path output_path(argv[6]);
        const std::filesystem::path report_path(argv[7]);

        const auto expected = read_fixture(fixture_path);
        const auto waveform = read_waveform(input_path);
        const streamglass::TokenizerAssets tokenizer(runtime_assets);
        const auto& conditioning = tokenizer.language(target_language);
        streamglass::OfficialOnlineFbank frontend;
        streamglass::EncoderRuntime encoder(
            engine_dir / environment_or(
                "STREAMGLASS_ENCODER_ENGINE_NAME",
                "encoder_cumulative_fp16.engine"));
        streamglass::DecoderRuntime decoder(
            engine_dir / environment_or(
                "STREAMGLASS_DECODER_PREFILL_ENGINE_NAME",
                "decoder_prefill_mixed.engine"),
            engine_dir / environment_or(
                "STREAMGLASS_DECODER_STEP_ENGINE_NAME",
                "decoder_step_mixed.engine"));
        streamglass::StreamingTextPolicy text_policy(
            decoder, tokenizer, target_language);
        text_policy.set_cached_mode(
            environment_or("STREAMGLASS_DECODER_CACHED", "0") ==
            std::string("1"));
        streamglass::T2uRuntime t2u(
            engine_dir / environment_or(
                "STREAMGLASS_T2U_FRONTEND_ENGINE_NAME",
                "t2u_frontend_fp16.engine"),
            engine_dir / "t2u_decoder_fp32.engine",
            tokenizer);
        streamglass::VocoderRuntime vocoder(
            engine_dir / "vocoder_fp16.engine");

        streamglass::DeviceBuffer<__half> retained_text_features(
            streamglass::T2uRuntime::kMaximumTextLength *
            streamglass::T2uRuntime::kFeatureDimension);
        std::vector<std::int64_t> retained_text_tokens;
        retained_text_tokens.reserve(streamglass::T2uRuntime::kMaximumTextLength);
        std::size_t retained_feature_count = 0;
        bool retained_text_finished = false;
        bool t2u_active = false;
        std::size_t duration_cursor = 0;
        std::vector<float> output_waveform;
        std::vector<PolicyReport> reports;
        reports.reserve(expected.size());
        std::string first_semantic_divergence;
        std::string first_strict_unit_divergence;
        bool all_semantics_passed = true;
        bool all_units_passed = true;
        std::size_t total_matching_units = 0;
        std::size_t total_reference_units = 0;

        auto fbank_input = encoder.fbank_input();
        std::size_t valid_fbank_frames = 0;
        std::size_t audio_offset = 0;
        std::size_t policy_index = 0;
        while (audio_offset < waveform.size()) {
            const auto sample_count =
                std::min(kSegmentSamples, waveform.size() - audio_offset);
            const auto source_finished =
                audio_offset + sample_count == waveform.size();
            const auto output_offset =
                valid_fbank_frames * streamglass::OfficialOnlineFbank::kFeatureDimension;
            const auto emitted_frames = frontend.extract(
                std::span(waveform).subspan(audio_offset, sample_count),
                fbank_input.subspan(output_offset));
            valid_fbank_frames += emitted_frames;
            audio_offset += sample_count;

            if (valid_fbank_frames < kMinimumStartingFbankFrames &&
                !source_finished) {
                continue;
            }
            if (valid_fbank_frames < streamglass::EncoderRuntime::kMinimumFbankFrames) {
                throw std::runtime_error(
                    "Final utterance is shorter than the exported encoder profile");
            }
            if (policy_index >= expected.size()) {
                throw std::runtime_error("Native pipeline produced extra policies");
            }

            const auto padded_fbank_frames =
                (valid_fbank_frames + 1U) & ~std::size_t{1U};
            streamglass::EncoderResult encoder_result{};
            const auto encoder_ms = timed_milliseconds([&] {
                encoder_result =
                    encoder.run(valid_fbank_frames, padded_fbank_frames);
            });

            streamglass::TextPolicyResult text_result{};
            const auto decoder_ms = timed_milliseconds([&] {
                text_result = text_policy.run(
                    encoder.output_device(),
                    encoder_result.valid_frames,
                    source_finished);
            });

            if (text_result.action == streamglass::TextPolicyAction::write) {
                if (text_result.t2u_feature_count >
                    streamglass::T2uRuntime::kMaximumTextLength) {
                    throw std::length_error("T2U text feature sequence is too long");
                }
                const auto feature_bytes =
                    text_result.t2u_feature_count *
                    streamglass::T2uRuntime::kFeatureDimension * sizeof(__half);
                streamglass::check_cuda(
                    cudaMemcpy(
                        retained_text_features.data(),
                        text_result.t2u_features_device,
                        feature_bytes,
                        cudaMemcpyDeviceToDevice),
                    "cudaMemcpy retained T2U decoder features");
                retained_text_tokens.assign(
                    text_result.t2u_token_indices.begin(),
                    text_result.t2u_token_indices.end());
                retained_feature_count = text_result.t2u_feature_count;
                retained_text_finished = text_result.finished;
                t2u_active = text_policy.committed_tokens().size() >= 2U;
            }

            bool actual_has_t2u = false;
            streamglass::T2uResult t2u_result{};
            double t2u_ms = 0.0;
            double vocoder_ms = 0.0;
            std::vector<std::int32_t> policy_units;
            bool actual_t2u_finished = false;
            if (t2u_active) {
                actual_has_t2u = true;
                t2u_ms = timed_milliseconds([&] {
                    t2u_result = t2u.run(
                        retained_text_tokens,
                        retained_text_features.data(),
                        retained_feature_count,
                        duration_cursor,
                        retained_text_finished,
                        50);
                });
                duration_cursor = t2u_result.next_duration_start_index;
                actual_t2u_finished =
                    retained_text_finished &&
                    t2u_result.action != streamglass::T2uAction::read;

                if (t2u_result.unit_count > 0) {
                    std::vector<std::int32_t> all_units(
                        t2u_result.total_unit_count);
                    t2u.copy_unit_ids(all_units, all_units.size());
                    policy_units.assign(
                        all_units.begin() +
                            static_cast<std::ptrdiff_t>(t2u_result.unit_offset),
                        all_units.begin() + static_cast<std::ptrdiff_t>(
                            t2u_result.unit_offset + t2u_result.unit_count));
                    if (t2u_result.unit_count >
                        streamglass::VocoderRuntime::kMaximumUnitCount) {
                        throw std::length_error(
                            "Official unit WRITE exceeds vocoder TensorRT profile");
                    }
                    vocoder_ms = timed_milliseconds([&] {
                        const auto vocoder_result = vocoder.run(
                            t2u.unit_ids_device(t2u_result),
                            t2u_result.unit_count,
                            conditioning.vocoder_language,
                            conditioning.vocoder_speaker);
                        const auto playback =
                            vocoder.materialize_for_playback(vocoder_result);
                        output_waveform.insert(
                            output_waveform.end(), playback.begin(), playback.end());
                    });
                }
            }

            const auto& reference = expected[policy_index];
            std::vector<std::int64_t> selected;
            selected.reserve(text_result.decoder_calls.size());
            double probability_max_abs = 0.0;
            for (std::size_t call = 0; call < text_result.decoder_calls.size();
                 ++call) {
                selected.push_back(
                    text_result.decoder_calls[call].selected_index);
                if (call < reference.probabilities.size()) {
                    probability_max_abs = std::max(
                        probability_max_abs,
                        std::abs(
                            static_cast<double>(
                                text_result.decoder_calls[call].probability) -
                            static_cast<double>(reference.probabilities[call])));
                }
            }
            const auto text_action =
                static_cast<std::uint32_t>(text_result.action);
            const auto t2u_action = actual_has_t2u
                ? static_cast<std::uint32_t>(t2u_result.action)
                : 0U;
            const auto semantic_passed =
                valid_fbank_frames == reference.fbank_valid &&
                padded_fbank_frames == reference.fbank_padded &&
                encoder_result.valid_frames == reference.encoder_frames &&
                source_finished == reference.source_finished &&
                text_action == reference.text_action &&
                text_result.finished == reference.text_finished &&
                equal_span<std::int64_t>(
                    text_policy.committed_tokens(), reference.target_after) &&
                selected == reference.selected &&
                actual_has_t2u == reference.has_t2u &&
                (!actual_has_t2u ||
                 (t2u_action == reference.t2u_action &&
                  actual_t2u_finished == reference.t2u_finished &&
                  duration_cursor == reference.duration_cursor &&
                  policy_units.size() == reference.unit_ids.size()));
            const auto strict_unit_passed =
                !actual_has_t2u || policy_units == reference.unit_ids;
            const auto policy_matching_units = actual_has_t2u
                ? matching_elements<std::int32_t>(
                      policy_units, reference.unit_ids)
                : 0U;
            all_semantics_passed = all_semantics_passed && semantic_passed;
            all_units_passed = all_units_passed && strict_unit_passed;
            total_matching_units += policy_matching_units;
            total_reference_units += reference.unit_ids.size();
            if (!semantic_passed && first_semantic_divergence.empty()) {
                first_semantic_divergence =
                    "policy " + std::to_string(policy_index);
            }
            if (!strict_unit_passed && first_strict_unit_divergence.empty()) {
                first_strict_unit_divergence =
                    "policy " + std::to_string(policy_index) + " " +
                    first_difference<std::int32_t>(
                        policy_units, reference.unit_ids);
            }
            if (!semantic_passed || !strict_unit_passed) {
                std::cerr
                    << "divergence policy=" << policy_index
                    << " fbank="
                    << (valid_fbank_frames == reference.fbank_valid)
                    << " encoder="
                    << (encoder_result.valid_frames == reference.encoder_frames)
                    << " source_finished="
                    << (source_finished == reference.source_finished)
                    << " text_action="
                    << (text_action == reference.text_action)
                    << " text_finished="
                    << (text_result.finished == reference.text_finished)
                    << " target="
                    << equal_span<std::int64_t>(
                           text_policy.committed_tokens(),
                           reference.target_after)
                    << " selected=" << (selected == reference.selected)
                    << " has_t2u="
                    << (actual_has_t2u == reference.has_t2u)
                    << " t2u_action="
                    << (t2u_action == reference.t2u_action)
                    << " t2u_finished="
                    << (actual_t2u_finished == reference.t2u_finished)
                    << " cursor="
                    << (duration_cursor == reference.duration_cursor)
                    << " units=" << (policy_units == reference.unit_ids)
                    << '\n';
                if (selected != reference.selected) {
                    std::cerr << "selected "
                              << first_difference<std::int64_t>(
                                     selected, reference.selected)
                              << '\n';
                }
                if (!strict_unit_passed) {
                    std::cerr << "units "
                              << first_difference<std::int32_t>(
                                     policy_units, reference.unit_ids)
                              << '\n';
                }
            }
            reports.push_back(PolicyReport{
                .index = policy_index,
                .fbank_frames = valid_fbank_frames,
                .encoder_frames = encoder_result.valid_frames,
                .text_action = text_action,
                .text_finished = text_result.finished,
                .t2u_action = t2u_action,
                .t2u_finished = actual_t2u_finished,
                .emitted_tokens = text_result.emitted_tokens.size(),
                .emitted_units = policy_units.size(),
                .encoder_ms = encoder_ms,
                .decoder_ms = decoder_ms,
                .t2u_ms = t2u_ms,
                .vocoder_ms = vocoder_ms,
                .probability_max_abs = probability_max_abs,
                .matching_units = policy_matching_units,
                .reference_units = reference.unit_ids.size(),
                .semantic_passed = semantic_passed,
                .strict_unit_passed = strict_unit_passed,
            });
            std::cout << "policy=" << policy_index
                      << " fbank=" << valid_fbank_frames
                      << " encoder=" << encoder_result.valid_frames
                      << " text_action=" << text_action
                      << " tokens=" << text_result.emitted_tokens.size()
                      << " t2u_action=" << t2u_action
                      << " units=" << policy_units.size()
                      << " latency_ms="
                      << encoder_ms + decoder_ms + t2u_ms + vocoder_ms
                      << " semantic_passed="
                      << (semantic_passed ? "true" : "false")
                      << " strict_unit_passed="
                      << (strict_unit_passed ? "true" : "false")
                      << '\n';
            ++policy_index;
        }

        if (policy_index != expected.size()) {
            all_semantics_passed = false;
            if (first_semantic_divergence.empty()) {
                first_semantic_divergence = "policy count";
            }
        }
        const auto translation = tokenizer.detokenize(
            text_policy.committed_tokens());
        write_waveform(output_path, output_waveform);
        write_report(
            report_path,
            reports,
            translation,
            first_semantic_divergence,
            first_strict_unit_divergence,
            waveform.size(),
            output_waveform.size(),
            total_matching_units,
            total_reference_units,
            all_semantics_passed,
            all_units_passed);
        const auto strict_required = environment_or(
            "STREAMGLASS_REQUIRE_STRICT_UNIT_PARITY", "0") == "1";
        std::cout << "translation=\"" << translation << "\"\n"
                  << "output_samples=" << output_waveform.size() << '\n'
                  << "policies=" << policy_index
                  << " semantic_passed="
                  << (all_semantics_passed ? "true" : "false")
                  << " strict_unit_parity_passed="
                  << (all_units_passed ? "true" : "false")
                  << " units=" << total_matching_units << '/'
                  << total_reference_units << '\n';
        return all_semantics_passed && (!strict_required || all_units_passed)
            ? 0
            : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
