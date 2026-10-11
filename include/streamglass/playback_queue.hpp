#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <span>
#include <vector>

namespace streamglass {

enum class PlaybackPushStatus : std::int32_t {
    accepted = 0,
    rejected_order = 1,
    rejected_overflow = 2,
    rejected_timeout = 3,
};

struct PlaybackStats {
    std::uint64_t chunks_accepted = 0;
    std::uint64_t gap_events = 0;
    std::uint64_t overflow_rejects = 0;
    std::uint64_t order_rejects = 0;
    double jitter_target_ms = 0.0;
};

// Playback shaping queue (engineering framework §4).
//
// - Chunks are accepted strictly ordered by (utterance_id, write_id).
// - A per-utterance prebuffer gate implements the adaptive jitter buffer
//   (AIMD: gap +40 ms, 300 s clean -10 ms, clamped to [80, 250] ms).
// - Adjacent chunks of one utterance join with an 8 ms equal-power
//   crossfade; recovery after a gap gets a 5 ms fade-in and the final
//   tail a 5 ms fade-out. Shaping happens on the producer side; the
//   consumer only copies samples and decides starvation at boundaries.
// - Overflow is an explicit rejection, never a silent drop.
class PlaybackQueue final {
public:
    static constexpr std::size_t kSampleRate = 16'000;
    static constexpr std::size_t kCrossfadeSamples = 128;  // 8 ms
    static constexpr std::size_t kFadeSamples = 80;        // 5 ms
    static constexpr double kJitterInitialMs = 120.0;
    static constexpr double kJitterMinimumMs = 80.0;
    static constexpr double kJitterMaximumMs = 250.0;
    static constexpr double kJitterIncreaseMs = 40.0;
    static constexpr double kJitterDecreaseMs = 10.0;
    static constexpr std::size_t kCleanSamplesForDecrease =
        300U * kSampleRate;
    static constexpr std::size_t kCapacitySamples = 8U * kSampleRate;

    PlaybackQueue() = default;

    [[nodiscard]] PlaybackPushStatus push(
        std::uint64_t utterance_id,
        std::uint32_t write_id,
        std::span<const float> samples,
        bool is_final);

    // Waits for the consumer to free capacity without consuming write_id on
    // failure. Normal pushes remain non-blocking; the realtime runtime uses
    // this bounded path only when playback has fallen behind inference.
    [[nodiscard]] PlaybackPushStatus push_wait(
        std::uint64_t utterance_id,
        std::uint32_t write_id,
        std::span<const float> samples,
        bool is_final,
        std::chrono::milliseconds timeout);

    // Fills the whole span (silence where no audio is due) and returns the
    // number of real audio samples written.
    std::size_t pull(std::span<float> output);

    void reset() noexcept;

    [[nodiscard]] PlaybackStats stats() const;
    [[nodiscard]] double buffered_ms() const;

private:
    struct Chunk {
        std::uint64_t utterance_id = 0;
        std::vector<float> samples;
        bool is_final = false;
    };

    void finalize_pending_tail_locked(std::uint64_t utterance_id);
    void gap_locked() noexcept;
    void clean_locked(std::size_t sample_count) noexcept;
    [[nodiscard]] std::size_t front_utterance_samples_locked(
        std::uint64_t utterance_id) const noexcept;
    [[nodiscard]] PlaybackPushStatus push_locked(
        std::uint64_t utterance_id,
        std::uint32_t write_id,
        std::span<const float> samples,
        bool is_final);

    mutable std::mutex mutex_;
    std::condition_variable space_available_;
    std::deque<Chunk> queue_;
    std::vector<float> pending_tail_;
    std::size_t cursor_ = 0;
    std::size_t buffered_samples_ = 0;
    std::uint64_t producer_utterance_ = 0;
    std::uint32_t producer_next_write_ = 0;
    bool producer_active_ = false;
    bool producer_after_gap_ = false;
    std::uint64_t consumer_utterance_ = 0;
    bool gate_open_ = false;
    bool consumer_active_ = false;
    bool in_gap_ = false;
    double jitter_target_ms_ = kJitterInitialMs;
    std::size_t clean_samples_ = 0;
    PlaybackStats stats_{.jitter_target_ms = kJitterInitialMs};
};

}  // namespace streamglass
