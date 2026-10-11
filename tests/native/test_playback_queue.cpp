#include "streamglass/playback_queue.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> sine(
    const std::size_t count, const std::size_t phase_offset) {
    std::vector<float> samples(count);
    for (std::size_t index = 0; index < count; ++index) {
        samples[index] = 0.5F * std::sin(
            2.0F * std::numbers::pi_v<float> * 110.0F *
            static_cast<float>(index + phase_offset) / 16'000.0F);
    }
    return samples;
}

}  // namespace

int main() {
    using streamglass::PlaybackPushStatus;
    using streamglass::PlaybackQueue;

    {   // Ordering rules.
        PlaybackQueue queue;
        require(queue.push(1, 0, sine(256, 0), false) ==
                PlaybackPushStatus::accepted, "first chunk accepted");
        require(queue.push(1, 2, sine(256, 0), false) ==
                PlaybackPushStatus::rejected_order, "write gap rejected");
        require(queue.push(0, 0, sine(256, 0), false) ==
                PlaybackPushStatus::rejected_order, "old utterance rejected");
        require(queue.push(1, 1, sine(256, 0), true) ==
                PlaybackPushStatus::accepted, "in-order accepted");
    }

    {   // Prebuffer gate: below target -> silence, final -> immediate open.
        PlaybackQueue queue;
        static_cast<void>(queue.push(1, 0, sine(800, 0), false));  // 50 ms
        std::vector<float> out(320);
        require(queue.pull(out) == 0, "gate holds below jitter target");
        static_cast<void>(queue.push(1, 1, sine(800, 800), true));
        require(queue.pull(out) > 0, "final opens the gate");
    }

    {   // Crossfade continuity across a chunk boundary.
        PlaybackQueue queue;
        const auto first = sine(4'000, 0);
        const auto second = sine(4'000, 4'000);
        static_cast<void>(queue.push(7, 0, first, false));
        static_cast<void>(queue.push(7, 1, second, true));
        std::vector<float> out(first.size() + second.size());
        const auto real = queue.pull(out);
        require(real ==
                first.size() + second.size() - PlaybackQueue::kCrossfadeSamples,
                "crossfade overlaps exactly one blend region");
        float max_step = 0.0F;
        for (std::size_t index = 1; index < real; ++index) {
            max_step = std::max(
                max_step, std::abs(out[index] - out[index - 1]));
        }
        require(max_step < 0.15F, "no click at the chunk boundary");
        require(std::abs(out[real - 1]) < 0.02F, "final tail fades out");
    }

    {   // Starvation at boundary -> one gap event, AIMD raise, fade-in resume.
        PlaybackQueue queue;
        static_cast<void>(queue.push(9, 0, sine(4'000, 0), false));
        std::vector<float> out(8'000);
        static_cast<void>(queue.pull(out));           // drains + starves
        const auto stats = queue.stats();
        require(stats.gap_events == 1, "starvation counted once");
        require(stats.jitter_target_ms > 150.0, "jitter target raised");
        static_cast<void>(queue.push(9, 1, sine(4'000, 4'000), true));
        std::vector<float> resume(4'000);
        require(queue.pull(resume) > 0, "resume after gap");
        require(std::abs(resume[0]) < 0.02F, "fade-in after gap");
    }

    {   // Overflow is an explicit rejection.
        PlaybackQueue queue;
        const auto big = sine(PlaybackQueue::kCapacitySamples - 100, 0);
        require(queue.push(3, 0, big, false) ==
                PlaybackPushStatus::accepted, "large chunk fits");
        require(queue.push(3, 1, sine(4'000, 0), false) ==
                PlaybackPushStatus::rejected_overflow, "overflow rejected");
    }

    {   // Bounded backpressure resumes with the same write id after drain.
        PlaybackQueue queue;
        const auto big = sine(PlaybackQueue::kCapacitySamples - 100, 0);
        const auto next = sine(4'000, big.size());
        require(queue.push(4, 0, big, false) ==
                PlaybackPushStatus::accepted, "backpressure fixture accepted");

        PlaybackPushStatus waited = PlaybackPushStatus::rejected_timeout;
        std::thread producer([&] {
            waited = queue.push_wait(
                4, 1, next, false, std::chrono::milliseconds(500));
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::vector<float> drained(8'000);
        require(queue.pull(drained) == drained.size(),
                "consumer frees queue capacity");
        producer.join();
        require(waited == PlaybackPushStatus::accepted,
                "producer resumes after capacity is available");
        require(queue.push(4, 2, sine(256, 0), true) ==
                PlaybackPushStatus::accepted,
                "write sequence continues after backpressure");
    }

    {   // A timeout does not consume the producer's expected write id.
        PlaybackQueue queue;
        const auto big = sine(PlaybackQueue::kCapacitySamples - 100, 0);
        const auto next = sine(4'000, big.size());
        require(queue.push(5, 0, big, false) ==
                PlaybackPushStatus::accepted, "timeout fixture accepted");
        require(queue.push_wait(
                    5, 1, next, false, std::chrono::milliseconds(5)) ==
                PlaybackPushStatus::rejected_timeout,
                "bounded wait reports timeout");
        std::vector<float> drained(8'000);
        static_cast<void>(queue.pull(drained));
        require(queue.push(5, 1, next, true) ==
                PlaybackPushStatus::accepted,
                "same write id remains valid after timeout");
    }

    std::cout << "playback queue tests passed\n";
    return 0;
}
