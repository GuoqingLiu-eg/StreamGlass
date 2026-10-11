#include "streamglass/streaming_pipeline.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace streamglass {
namespace {

template <typename Function>
double timed_milliseconds(Function&& function) {
    const auto begin = std::chrono::steady_clock::now();
    std::forward<Function>(function)();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

StreamingPipeline::StreamingPipeline(
    const StreamingPipelinePaths& paths,
    std::string target_language,
    const StreamingPipelineOptions& options)
    : tokenizer_(paths.runtime_assets),
      conditioning_(tokenizer_.language(target_language)),
      frontend_(),
      encoder_(paths.encoder),
      decoder_(paths.decoder_prefill, paths.decoder_step),
      text_policy_(decoder_, tokenizer_, target_language),
      t2u_(paths.t2u_frontend, paths.t2u_decoder, tokenizer_),
      vocoder_(paths.vocoder),
      retained_text_features_(
          T2uRuntime::kMaximumTextLength * T2uRuntime::kFeatureDimension) {
    retained_text_tokens_.reserve(T2uRuntime::kMaximumTextLength);
    text_policy_.set_cached_mode(options.cached_decoder);
}

void StreamingPipeline::reset() {
    frontend_.reset();
    text_policy_.reset();
    retained_text_tokens_.clear();
    retained_feature_count_ = 0;
    retained_text_finished_ = false;
    retained_text_version_ = 0;
    t2u_.invalidate_memo();
    t2u_active_ = false;
    duration_cursor_ = 0;
    valid_fbank_frames_ = 0;
    total_input_samples_ = 0;
    synthesis_serial_ = 0;
    last_t2u_result_ = {};
    source_finished_ = false;
}

void StreamingPipeline::set_target_language(
    const std::string_view target_language) {
    const auto next_conditioning = tokenizer_.language(target_language);
    reset();
    conditioning_ = next_conditioning;
    text_policy_.set_target_language(target_language);
}

StreamingPipelineStep StreamingPipeline::process_audio(
    const std::span<const float> samples,
    const bool source_finished) {
    const std::array segments{
        PipelineAudioSegment{.samples = samples, .source_finished = source_finished}};
    return process_audio_batch(segments);
}

StreamingPipelineStep StreamingPipeline::process_audio_batch(
    const std::span<const PipelineAudioSegment> segments) {
    if (source_finished_) {
        throw std::logic_error("Streaming pipeline requires reset after EOS");
    }
    if (segments.empty() || segments.size() > 4U) {
        throw std::invalid_argument("Audio batch must contain one to four segments");
    }
    auto fbank_input = encoder_.fbank_input();
    double frontend_ms = 0.0;
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const auto& segment = segments[index];
        if (segment.samples.size() > kSegmentSamples ||
            (!segment.source_finished &&
             segment.samples.size() != kSegmentSamples) ||
            (segment.samples.empty() && !segment.source_finished) ||
            (segment.source_finished && index + 1U != segments.size())) {
            throw std::invalid_argument(
                "Audio batch contains an invalid 320ms segment boundary");
        }
        if (total_input_samples_ + segment.samples.size() >
            EncoderRuntime::kMaximumFbankFrames * 160U) {
            throw std::length_error(
                "Utterance exceeds cumulative encoder profile");
        }
        ++serial_;
        total_input_samples_ += segment.samples.size();
        const auto output_offset =
            valid_fbank_frames_ * OfficialOnlineFbank::kFeatureDimension;
        std::size_t emitted_frames = 0;
        frontend_ms += timed_milliseconds([&] {
            emitted_frames = frontend_.extract(
                segment.samples, fbank_input.subspan(output_offset));
        });
        valid_fbank_frames_ += emitted_frames;
    }
    source_finished_ = segments.back().source_finished;

    StreamingPipelineStep step{};
    step.serial = serial_;
    step.source_finished = source_finished_;
    step.input_samples = total_input_samples_;
    step.fbank_frames = valid_fbank_frames_;
    step.frontend_ms = frontend_ms;
    if (valid_fbank_frames_ < kMinimumStartingFbankFrames && !source_finished_) {
        return step;
    }
    if (valid_fbank_frames_ < EncoderRuntime::kMinimumFbankFrames) {
        throw std::runtime_error(
            "Final utterance is shorter than the current TensorRT encoder profile");
    }

    const auto padded_fbank_frames =
        (valid_fbank_frames_ + 1U) & ~std::size_t{1U};
    EncoderResult encoder_result{};
    step.encoder_ms = timed_milliseconds([&] {
        encoder_result = encoder_.run(
            valid_fbank_frames_, padded_fbank_frames);
    });

    TextPolicyResult text_result{};
    step.decoder_ms = timed_milliseconds([&] {
        text_result = text_policy_.run(
            encoder_.output_device(),
            encoder_result.valid_frames,
            source_finished_);
    });

    if (text_result.action == TextPolicyAction::write) {
        if (text_result.t2u_feature_count > T2uRuntime::kMaximumTextLength) {
            throw std::length_error("T2U text feature sequence is too long");
        }
        const auto feature_bytes =
            text_result.t2u_feature_count * T2uRuntime::kFeatureDimension *
            sizeof(__half);
        check_cuda(
            cudaMemcpy(
                retained_text_features_.data(),
                text_result.t2u_features_device,
                feature_bytes,
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy retained T2U decoder features");
        retained_text_tokens_.assign(
            text_result.t2u_token_indices.begin(),
            text_result.t2u_token_indices.end());
        retained_feature_count_ = text_result.t2u_feature_count;
        retained_text_finished_ = text_result.finished;
        ++retained_text_version_;
        t2u_active_ = text_policy_.committed_tokens().size() >= 2U;
    }

    last_t2u_result_ = {};
    bool t2u_finished = false;
    if (t2u_active_) {
        step.t2u_ms = timed_milliseconds([&] {
            last_t2u_result_ = t2u_.run(
                retained_text_tokens_,
                retained_text_features_.data(),
                retained_feature_count_,
                duration_cursor_,
                retained_text_finished_,
                kMinimumUnitChunk,
                retained_text_version_);
        });
        duration_cursor_ = last_t2u_result_.next_duration_start_index;
        t2u_finished = retained_text_finished_ &&
                       last_t2u_result_.action != T2uAction::read;
        if (last_t2u_result_.unit_count > VocoderRuntime::kMaximumUnitCount) {
            throw std::length_error(
                "Official unit WRITE exceeds vocoder TensorRT profile");
        }
        if (last_t2u_result_.unit_count > 0) {
            synthesis_serial_ = serial_;
        }
    }

    step.policy_executed = true;
    step.encoder_frames = encoder_result.valid_frames;
    step.text_action = text_result.action;
    step.text_finished = text_result.finished;
    step.emitted_token_count = text_result.emitted_tokens.size();
    step.committed_tokens = text_policy_.committed_tokens();
    step.decoder_calls = text_result.decoder_calls;
    step.t2u_active = t2u_active_;
    step.t2u_action = last_t2u_result_.action;
    step.t2u_finished = t2u_finished;
    step.duration_cursor = duration_cursor_;
    step.output_unit_count = last_t2u_result_.unit_count;
    return step;
}

StreamingSynthesisResult StreamingPipeline::synthesize(
    const StreamingPipelineStep& step,
    MappedHostBuffer<float>& destination) {
    if (step.serial != synthesis_serial_ || step.output_unit_count == 0 ||
        step.output_unit_count != last_t2u_result_.unit_count) {
        throw std::invalid_argument("Pipeline step has no current vocoder output");
    }

    VocoderResult result{};
    const auto vocoder_ms = timed_milliseconds([&] {
        result = vocoder_.run(
            t2u_.unit_ids_device(last_t2u_result_),
            last_t2u_result_.unit_count,
            conditioning_.vocoder_language,
            conditioning_.vocoder_speaker);
        static_cast<void>(
            vocoder_.materialize_for_playback(result, destination));
    });
    return StreamingSynthesisResult{
        .sample_count = result.sample_count,
        .vocoder_ms = vocoder_ms,
    };
}

std::string StreamingPipeline::translation() const {
    return tokenizer_.detokenize(text_policy_.committed_tokens());
}

}  // namespace streamglass
