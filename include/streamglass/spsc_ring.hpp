#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace streamglass {

template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity > 0, "SPSC capacity must be non-zero");
    static_assert(
        std::is_nothrow_copy_assignable_v<T>,
        "Hot-path queue payloads must be nothrow copy assignable");

public:
    [[nodiscard]] bool try_push(const T& value) noexcept {
        const auto write = write_.load(std::memory_order_relaxed);
        const auto read = read_.load(std::memory_order_acquire);
        if (write - read == Capacity) {
            return false;
        }
        slots_[write % Capacity] = value;
        write_.store(write + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_push(T&& value) noexcept {
        const auto write = write_.load(std::memory_order_relaxed);
        const auto read = read_.load(std::memory_order_acquire);
        if (write - read == Capacity) {
            return false;
        }
        slots_[write % Capacity] = std::move(value);
        write_.store(write + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(T& value) noexcept {
        const auto read = read_.load(std::memory_order_relaxed);
        const auto write = write_.load(std::memory_order_acquire);
        if (read == write) {
            return false;
        }
        value = slots_[read % Capacity];
        read_.store(read + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        const auto write = write_.load(std::memory_order_acquire);
        const auto read = read_.load(std::memory_order_acquire);
        return write - read;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool full() const noexcept { return size() == Capacity; }

    static constexpr std::size_t capacity() noexcept { return Capacity; }

private:
    alignas(64) std::atomic<std::size_t> write_{0};
    alignas(64) std::atomic<std::size_t> read_{0};
    std::array<T, Capacity> slots_{};
};

}  // namespace streamglass

