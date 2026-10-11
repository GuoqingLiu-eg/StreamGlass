#pragma once

#include "streamglass/cuda_buffer.hpp"
#include "streamglass/decoder_runtime.hpp"
#include "streamglass/encoder_runtime.hpp"
#include "streamglass/online_fbank.hpp"
#include "streamglass/streaming_text_policy.hpp"
#include "streamglass/t2u_runtime.hpp"
#include "streamglass/tokenizer_assets.hpp"
#include "streamglass/vocoder_runtime.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace streamglass {

struct StreamingPipelinePaths {
    std::filesystem::path encoder;
    std::filesystem::path decoder_prefill;
    std::filesystem::path decoder_step;
    std::filesystem::path t2u_frontend;
    std::filesystem::path t2u_decoder;
    std::filesystem::path vocoder;
    std::filesystem::path runtime_assets;
};

struct StreamingPipelineOptions {
    // Cross-policy decoder KV cache (hybrid probe mode). Semantic deviation
    // gated by reference/cached A/B; reference behaviour is the default.
    bool cached_decoder = false;
};

struct PipelineAudioSegment {
    std::span<const float> samples;
    bool source_finished = false;
};

struct StreamingPipelineStep {
    std::uint64_t serial = 0;
    bool policy_executed = false;
    bool source_finished = false;
    std::size_t input_samples = 0;
    std::size_t fbank_frames = 0;
    std::size_t encoder_frames = 0;
    TextPolicyAction text_action = TextPolicyAction::read;
    bool text_finished = false;
    std::size_t emitted_token_count = 0;
    std::span<const std::int64_t> committed_tokens;
    std::span<const DecoderCallDecision> decoder_calls;
    bool t2u_active = false;
    T2uAction t2u_action = T2uAction::read;
    bool t2u_finished = false;
    std::size_t duration_cursor = 0;
    std::size_t output_unit_count = 0;
    double frontend_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    double t2u_ms = 0.0;

    [[nodiscard]] double inference_ms() const noexcept {
        return frontend_ms + encoder_ms + decoder_ms + t2u_ms;
    }
};

struct StreamingSynthesisResult {
    std::size_t sample_count = 0;
    double vocoder_ms = 0.0;
};

class StreamingPipeline final {
public:
    static constexpr std::size_t kSegmentSamples = 5'120;
    static constexpr std::size_t kMinimumStartingFbankFrames = 192;
    static constexpr std::size_t kMinimumUnitChunk = 50;

    StreamingPipeline(
        const StreamingPipelinePaths& paths,
        std::string target_language,
        const StreamingPipelineOptions& options = {});

    StreamingPipeline(const StreamingPipeline&) = delete;
    StreamingPipeline& operator=(const StreamingPipeline&) = delete;

    void reset();
    void set_target_language(std::string_view target_language);

    [[nodiscard]] StreamingPipelineStep process_audio(
        std::span<const float> samples,
        bool source_finished);

    [[nodiscard]] StreamingPipelineStep process_audio_batch(
        std::span<const PipelineAudioSegment> segments);

    [[nodiscard]] StreamingSynthesisResult synthesize(
        const StreamingPipelineStep& step,
        MappedHostBuffer<float>& destination);

    [[nodiscard]] std::string translation() const;
    [[nodiscard]] std::size_t total_input_samples() const noexcept {
        return total_input_samples_;
    }

private:
    TokenizerAssets tokenizer_;
    LanguageConditioning conditioning_;
    OfficialOnlineFbank frontend_;
    EncoderRuntime encoder_;
    DecoderRuntime decoder_;
    StreamingTextPolicy text_policy_;
    T2uRuntime t2u_;
    VocoderRuntime vocoder_;
    DeviceBuffer<__half> retained_text_features_;
    std::vector<std::int64_t> retained_text_tokens_;
    std::size_t retained_feature_count_ = 0;
    bool retained_text_finished_ = false;
    std::uint64_t retained_text_version_ = 0;
    bool t2u_active_ = false;
    std::size_t duration_cursor_ = 0;
    std::size_t valid_fbank_frames_ = 0;
    std::size_t total_input_samples_ = 0;
    std::uint64_t serial_ = 0;
    std::uint64_t synthesis_serial_ = 0;
    T2uResult last_t2u_result_{};
    bool source_finished_ = false;
};

}  // namespace streamglass
