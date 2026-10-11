#include "streamglass/utterance_segmenter.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace streamglass {

namespace {
constexpr std::uint32_t kWindowMs = 32;
}  // namespace

UtteranceSegmenter::UtteranceSegmenter(
    IVoiceActivityDetector& vad, SegmenterConfig config, Sink sink)
    : vad_(vad), config_(config), sink_(std::move(sink)) {
    if (!sink_) {
        throw std::invalid_argument("Segmenter requires a sink");
    }
    window_.reserve(kWindowSamples);
    candidate_.reserve(
        static_cast<std::size_t>(config_.start_speech_ms + kWindowMs - 1U) /
        kWindowMs * kWindowSamples);
    assembly_.reserve(kChunkSamples);
    preroll_.reserve(
        static_cast<std::size_t>(config_.preroll_ms) * kSampleRate / 1000U);
}

void UtteranceSegmenter::push(const std::span<const float> samples) {
    for (const auto sample : samples) {
        window_.push_back(sample);
        if (window_.size() == kWindowSamples) {
            process_window();
            window_.clear();
        }
    }
}

void UtteranceSegmenter::process_window() {
    const auto probability = vad_.probability(window_);
    const auto threshold = active_
                               ? config_.speech_threshold
                               : config_.start_speech_threshold;
    const bool speech = probability > threshold;

    if (!active_) {
        if (!speech) {
            if (config_.preroll_ms > 0) {
                const auto capacity =
                    static_cast<std::size_t>(config_.preroll_ms) *
                    kSampleRate / 1000U;
                preroll_.insert(
                    preroll_.end(), candidate_.begin(), candidate_.end());
                preroll_.insert(
                    preroll_.end(), window_.begin(), window_.end());
                if (preroll_.size() > capacity) {
                    preroll_.erase(
                        preroll_.begin(),
                        preroll_.begin() + static_cast<std::ptrdiff_t>(
                            preroll_.size() - capacity));
                }
            }
            candidate_.clear();
            candidate_speech_ms_ = 0;
            return;
        }

        candidate_.insert(candidate_.end(), window_.begin(), window_.end());
        candidate_speech_ms_ += kWindowMs;
        const auto start_limit =
            std::max(kWindowMs, config_.start_speech_ms);
        if (candidate_speech_ms_ < start_limit) {
            return;
        }

        ++utterance_id_;
        active_ = true;
        speech_ms_ = candidate_speech_ms_;
        silence_ms_ = 0;
        utterance_ms_ = 0;
        assembly_.clear();
        if (!preroll_.empty()) {
            assembly_.insert(
                assembly_.end(), preroll_.begin(), preroll_.end());
            utterance_ms_ += static_cast<std::uint32_t>(
                preroll_.size() * 1000U / kSampleRate);
            preroll_.clear();
        }
        assembly_.insert(
            assembly_.end(), candidate_.begin(), candidate_.end());
        utterance_ms_ += candidate_speech_ms_;
        candidate_.clear();
        candidate_speech_ms_ = 0;
    } else {
        assembly_.insert(assembly_.end(), window_.begin(), window_.end());
        utterance_ms_ += kWindowMs;
        if (speech) {
            speech_ms_ += kWindowMs;
            silence_ms_ = 0;
        } else {
            silence_ms_ += kWindowMs;
        }
    }

    while (assembly_.size() >= kChunkSamples) {
        emit(false);
    }

    auto silence_limit = config_.silence_limit_ms;
    if (speech_ms_ >= config_.speech_soft_limit_ms) {
        silence_limit /= 2U;
    }
    if (silence_ms_ >= silence_limit ||
        utterance_ms_ >= config_.hard_cap_ms) {
        finish_utterance();
    }
}

void UtteranceSegmenter::emit(const bool finished) {
    const auto count = finished
                           ? assembly_.size()
                           : std::min(assembly_.size(), kChunkSamples);
    const SegmentEvent event{
        .utterance_id = utterance_id_,
        .samples = std::span<const float>(assembly_.data(), count),
        .finished = finished,
    };
    sink_(event);
    assembly_.erase(
        assembly_.begin(), assembly_.begin() +
            static_cast<std::ptrdiff_t>(count));
}

void UtteranceSegmenter::finish_utterance() {
    // The final flush may be shorter than 5120 samples (or empty); the
    // downstream pipeline handles the residual exactly like the official
    // agent does on finished=True.
    emit(true);
    active_ = false;
    vad_.reset();
}

void UtteranceSegmenter::reset() {
    window_.clear();
    preroll_.clear();
    candidate_.clear();
    assembly_.clear();
    speech_ms_ = 0;
    candidate_speech_ms_ = 0;
    silence_ms_ = 0;
    utterance_ms_ = 0;
    active_ = false;
    vad_.reset();
}

}  // namespace streamglass
