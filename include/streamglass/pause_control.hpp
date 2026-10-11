#pragma once

#include <atomic>
#include <cstdint>

namespace streamglass {

// Cross-thread control state. Event-source threads only update atomics; the
// realtime capture thread owns all model, VAD, and playback resets.
class PauseControl final {
public:
    [[nodiscard]] bool set_paused(const bool paused) noexcept {
        const auto previous =
            paused_.exchange(paused, std::memory_order_acq_rel);
        if (previous == paused) {
            return false;
        }
        revision_.fetch_add(1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool toggle() noexcept {
        auto observed = paused_.load(std::memory_order_acquire);
        while (!paused_.compare_exchange_weak(
            observed, !observed,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        }
        revision_.fetch_add(1, std::memory_order_release);
        return !observed;
    }

    [[nodiscard]] bool paused() const noexcept {
        return paused_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::uint64_t revision() const noexcept {
        return revision_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> paused_{false};
    std::atomic<std::uint64_t> revision_{0};
};

}  // namespace streamglass
