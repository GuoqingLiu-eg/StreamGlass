#include "streamglass/playback_queue.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace streamglass {
namespace {

void apply_fade_in(std::span<float> samples) {
    const auto count = std::min(samples.size(), PlaybackQueue::kFadeSamples);
    for (std::size_t index = 0; index < count; ++index) {
        samples[index] *= static_cast<float>(index) /
                          static_cast<float>(PlaybackQueue::kFadeSamples);
    }
}

void apply_fade_out_tail(std::span<float> samples) {
    const auto count = std::min(samples.size(), PlaybackQueue::kFadeSamples);
    const auto offset = samples.size() - count;
    for (std::size_t index = 0; index < count; ++index) {
        samples[offset + index] *=
            static_cast<float>(count - index) / static_cast<float>(count);
    }
}

}  // namespace

PlaybackPushStatus PlaybackQueue::push(
    const std::uint64_t utterance_id,
    const std::uint32_t write_id,
    const std::span<const float> samples,
    const bool is_final) {
    const std::scoped_lock lock(mutex_);
    return push_locked(utterance_id, write_id, samples, is_final);
}

PlaybackPushStatus PlaybackQueue::push_wait(
    const std::uint64_t utterance_id,
    const std::uint32_t write_id,
    const std::span<const float> samples,
    const bool is_final,
    const std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);

    if (producer_active_ && utterance_id < producer_utterance_) {
        ++stats_.order_rejects;
        return PlaybackPushStatus::rejected_order;
    }
    if (producer_active_ && utterance_id == producer_utterance_ &&
        write_id != producer_next_write_) {
        ++stats_.order_rejects;
        return PlaybackPushStatus::rejected_order;
    }
    if (samples.size() > kCapacitySamples) {
        ++stats_.overflow_rejects;
        return PlaybackPushStatus::rejected_overflow;
    }

    const auto has_capacity = [&] {
        return buffered_samples_ + samples.size() <= kCapacitySamples;
    };
    if (!has_capacity() &&
        !space_available_.wait_for(lock, timeout, has_capacity)) {
        ++stats_.overflow_rejects;
        return PlaybackPushStatus::rejected_timeout;
    }

    return push_locked(utterance_id, write_id, samples, is_final);
}

PlaybackPushStatus PlaybackQueue::push_locked(
    const std::uint64_t utterance_id,
    const std::uint32_t write_id,
    const std::span<const float> samples,
    const bool is_final) {

    if (producer_active_ && utterance_id < producer_utterance_) {
        ++stats_.order_rejects;
        return PlaybackPushStatus::rejected_order;
    }
    if (producer_active_ && utterance_id == producer_utterance_ &&
        write_id != producer_next_write_) {
        ++stats_.order_rejects;
        return PlaybackPushStatus::rejected_order;
    }
    if (buffered_samples_ + samples.size() > kCapacitySamples) {
        ++stats_.overflow_rejects;
        return PlaybackPushStatus::rejected_overflow;
    }

    if (producer_active_ && utterance_id != producer_utterance_) {
        // The previous utterance never delivered is_final; close it audibly
        // instead of dropping its withheld tail.
        finalize_pending_tail_locked(producer_utterance_);
        producer_active_ = false;
    }
    if (!producer_active_) {
        producer_utterance_ = utterance_id;
        producer_active_ = true;
        producer_after_gap_ = false;
        pending_tail_.clear();
    }
    producer_next_write_ = write_id + 1U;

    Chunk chunk;
    chunk.utterance_id = utterance_id;
    chunk.is_final = is_final;
    chunk.samples.reserve(samples.size() + pending_tail_.size());

    std::size_t head = 0;
    if (!pending_tail_.empty()) {
        // Equal-power crossfade of the withheld previous tail with this head.
        const auto blend =
            std::min({pending_tail_.size(), samples.size(), kCrossfadeSamples});
        for (std::size_t index = 0; index < blend; ++index) {
            const auto theta =
                static_cast<float>(index) /
                static_cast<float>(blend) *
                (std::numbers::pi_v<float> / 2.0F);
            chunk.samples.push_back(
                pending_tail_[index] * std::cos(theta) +
                samples[index] * std::sin(theta));
        }
        head = blend;
        pending_tail_.clear();
    }

    const auto keep = (is_final || samples.size() <= head + kCrossfadeSamples)
                          ? std::size_t{0}
                          : kCrossfadeSamples;
    for (std::size_t index = head; index < samples.size() - keep; ++index) {
        chunk.samples.push_back(samples[index]);
    }
    if (keep > 0) {
        pending_tail_.assign(samples.end() - static_cast<std::ptrdiff_t>(keep),
                             samples.end());
    }
    if (producer_after_gap_) {
        apply_fade_in(chunk.samples);
        producer_after_gap_ = false;
        in_gap_ = false;
    }
    if (is_final) {
        apply_fade_out_tail(chunk.samples);
        producer_active_ = false;
    }

    buffered_samples_ += chunk.samples.size();
    queue_.push_back(std::move(chunk));
    ++stats_.chunks_accepted;
    return PlaybackPushStatus::accepted;
}

void PlaybackQueue::finalize_pending_tail_locked(
    const std::uint64_t utterance_id) {
    if (pending_tail_.empty()) {
        return;
    }
    Chunk chunk;
    chunk.utterance_id = utterance_id;
    chunk.is_final = true;
    chunk.samples = std::move(pending_tail_);
    pending_tail_.clear();
    apply_fade_out_tail(chunk.samples);
    buffered_samples_ += chunk.samples.size();
    queue_.push_back(std::move(chunk));
}

std::size_t PlaybackQueue::pull(const std::span<float> output) {
    std::unique_lock lock(mutex_);
    std::size_t written = 0;

    while (written < output.size()) {
        if (!gate_open_) {
            if (queue_.empty()) {
                break;
            }
            const auto head_utterance = queue_.front().utterance_id;
            const auto ready = front_utterance_samples_locked(head_utterance);
            const auto target = static_cast<std::size_t>(
                jitter_target_ms_ * static_cast<double>(kSampleRate) / 1000.0);
            const auto has_final = std::any_of(
                queue_.begin(), queue_.end(), [&](const Chunk& chunk) {
                    return chunk.utterance_id == head_utterance &&
                           chunk.is_final;
                });
            if (ready < target && !has_final) {
                break;
            }
            gate_open_ = true;
            consumer_active_ = true;
            consumer_utterance_ = head_utterance;
        }

        if (queue_.empty() ||
            queue_.front().utterance_id != consumer_utterance_) {
            if (consumer_active_) {
                // Mid-utterance starvation: the producer-side shaping keeps
                // chunk boundaries clean, so this stop is always "decent".
                if (!in_gap_) {
                    in_gap_ = true;
                    producer_after_gap_ = true;
                    gap_locked();
                }
                if (!queue_.empty()) {
                    // Next utterance arrived without a final marker for the
                    // current one; hand over through the gate.
                    consumer_active_ = false;
                    gate_open_ = false;
                    continue;
                }
            }
            break;
        }

        auto& front = queue_.front();
        const auto available = front.samples.size() - cursor_;
        const auto count = std::min(available, output.size() - written);
        std::copy_n(front.samples.begin() +
                        static_cast<std::ptrdiff_t>(cursor_),
                    count, output.begin() +
                        static_cast<std::ptrdiff_t>(written));
        cursor_ += count;
        written += count;
        buffered_samples_ -= count;
        clean_locked(count);

        if (cursor_ == front.samples.size()) {
            const auto was_final = front.is_final;
            queue_.pop_front();
            cursor_ = 0;
            if (was_final) {
                consumer_active_ = false;
                gate_open_ = false;
                in_gap_ = false;
            }
        }
    }

    std::fill(output.begin() + static_cast<std::ptrdiff_t>(written),
              output.end(), 0.0F);
    lock.unlock();
    if (written > 0) {
        space_available_.notify_one();
    }
    return written;
}

void PlaybackQueue::gap_locked() noexcept {
    ++stats_.gap_events;
    jitter_target_ms_ =
        std::min(jitter_target_ms_ + kJitterIncreaseMs, kJitterMaximumMs);
    stats_.jitter_target_ms = jitter_target_ms_;
    clean_samples_ = 0;
}

void PlaybackQueue::clean_locked(const std::size_t sample_count) noexcept {
    clean_samples_ += sample_count;
    if (clean_samples_ >= kCleanSamplesForDecrease) {
        jitter_target_ms_ =
            std::max(jitter_target_ms_ - kJitterDecreaseMs, kJitterMinimumMs);
        stats_.jitter_target_ms = jitter_target_ms_;
        clean_samples_ = 0;
    }
}

std::size_t PlaybackQueue::front_utterance_samples_locked(
    const std::uint64_t utterance_id) const noexcept {
    std::size_t total = 0;
    for (const auto& chunk : queue_) {
        if (chunk.utterance_id != utterance_id) {
            break;
        }
        total += chunk.samples.size() -
                 (&chunk == &queue_.front() ? cursor_ : std::size_t{0});
    }
    return total;
}

void PlaybackQueue::reset() noexcept {
    {
        const std::scoped_lock lock(mutex_);
        queue_.clear();
        pending_tail_.clear();
        cursor_ = 0;
        buffered_samples_ = 0;
        producer_active_ = false;
        producer_after_gap_ = false;
        consumer_active_ = false;
        gate_open_ = false;
        in_gap_ = false;
        jitter_target_ms_ = kJitterInitialMs;
        clean_samples_ = 0;
        stats_ = PlaybackStats{.jitter_target_ms = kJitterInitialMs};
    }
    space_available_.notify_all();
}

PlaybackStats PlaybackQueue::stats() const {
    const std::scoped_lock lock(mutex_);
    return stats_;
}

double PlaybackQueue::buffered_ms() const {
    const std::scoped_lock lock(mutex_);
    return static_cast<double>(buffered_samples_) * 1000.0 /
           static_cast<double>(kSampleRate);
}

}  // namespace streamglass
