#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace streamglass {

// Voice activity probability per 512-sample (32 ms) window at 16 kHz.
class IVoiceActivityDetector {
public:
    virtual ~IVoiceActivityDetector() = default;
    [[nodiscard]] virtual float probability(
        std::span<const float> window) = 0;
    virtual void reset() = 0;
};

struct SegmenterConfig {
    float start_speech_threshold = 0.15F;        // IDLE/candidate threshold
    float speech_threshold = 0.15F;              // official default
    std::uint32_t start_speech_ms = 0;           // consecutive speech to start
    std::uint32_t silence_limit_ms = 700;        // official default
    std::uint32_t speech_soft_limit_ms = 12'000; // halves the silence limit
    std::uint32_t hard_cap_ms = 20'000;          // forced finished (deploy)
    std::uint32_t preroll_ms = 0;                // 0 = official parity
};

struct SegmentEvent {
    std::uint64_t utterance_id = 0;
    std::span<const float> samples;  // 5120 except a final partial flush
    bool finished = false;
};

// Official SileroVADAgent semantics (silero_vad.py) plus the deployment
// hard cap and an optional pre-roll enhancement:
// IDLE: windows run through VAD; audio is dropped (pre-roll ring aside).
// ACTIVE: every sample is forwarded in 5120-sample segments; silence
//   windows accumulate towards the (soft-limit halved) end threshold.
// finished=true flushes the partial remainder; the next utterance gets a
// fresh id. The hard cap injects finished without dropping any samples.
class UtteranceSegmenter final {
public:
    static constexpr std::size_t kSampleRate = 16'000;
    static constexpr std::size_t kWindowSamples = 512;
    static constexpr std::size_t kChunkSamples = 5'120;

    using Sink = std::function<void(const SegmentEvent&)>;

    UtteranceSegmenter(
        IVoiceActivityDetector& vad, SegmenterConfig config, Sink sink);

    UtteranceSegmenter(const UtteranceSegmenter&) = delete;
    UtteranceSegmenter& operator=(const UtteranceSegmenter&) = delete;

    void push(std::span<const float> samples);
    void reset();

    [[nodiscard]] std::uint64_t utterance_id() const noexcept {
        return utterance_id_;
    }
    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    void process_window();
    void emit(bool finished);
    void finish_utterance();

    IVoiceActivityDetector& vad_;
    SegmenterConfig config_;
    Sink sink_;
    std::vector<float> window_;
    std::vector<float> preroll_;
    std::vector<float> candidate_;
    std::vector<float> assembly_;
    std::uint64_t utterance_id_ = 0;
    std::uint32_t speech_ms_ = 0;
    std::uint32_t candidate_speech_ms_ = 0;
    std::uint32_t silence_ms_ = 0;
    std::uint32_t utterance_ms_ = 0;
    bool active_ = false;
};

}  // namespace streamglass
