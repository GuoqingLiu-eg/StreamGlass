#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include "streamglass/spsc_ring.hpp"

namespace streamglass {

struct BufferHandle {
    std::uint32_t slot = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return slot != std::numeric_limits<std::uint32_t>::max();
    }

    friend bool operator==(const BufferHandle&, const BufferHandle&) = default;
};

template <typename Payload, std::size_t Capacity>
class SpscBufferPool {
    static_assert(Capacity > 0, "Buffer pool capacity must be non-zero");
    static_assert(
        Capacity <= std::numeric_limits<std::uint32_t>::max(),
        "Buffer pool is too large for BufferHandle");

    struct Slot {
        Payload payload{};
        std::atomic<std::uint32_t> generation{1};
        std::atomic<bool> leased{false};
    };

public:
    SpscBufferPool() {
        for (std::size_t index = 0; index < Capacity; ++index) {
            const auto slot = static_cast<std::uint32_t>(index);
            if (!free_.try_push(slot)) {
                throw std::logic_error("Failed to initialize buffer pool");
            }
        }
    }

    SpscBufferPool(const SpscBufferPool&) = delete;
    SpscBufferPool& operator=(const SpscBufferPool&) = delete;

    [[nodiscard]] std::optional<BufferHandle> try_acquire() noexcept {
        std::uint32_t index = 0;
        if (!free_.try_pop(index)) {
            return std::nullopt;
        }
        auto& slot = slots_[index];
        slot.leased.store(true, std::memory_order_release);
        return BufferHandle{
            index, slot.generation.load(std::memory_order_acquire)};
    }

    [[nodiscard]] Payload* get(const BufferHandle handle) noexcept {
        if (!is_live(handle)) {
            return nullptr;
        }
        return &slots_[handle.slot].payload;
    }

    [[nodiscard]] const Payload* get(const BufferHandle handle) const noexcept {
        if (!is_live(handle)) {
            return nullptr;
        }
        return &slots_[handle.slot].payload;
    }

    [[nodiscard]] bool release(const BufferHandle handle) noexcept {
        if (!is_live(handle)) {
            return false;
        }
        auto& slot = slots_[handle.slot];
        slot.leased.store(false, std::memory_order_release);
        slot.generation.fetch_add(1, std::memory_order_acq_rel);
        return free_.try_push(handle.slot);
    }

    [[nodiscard]] std::size_t available() const noexcept { return free_.size(); }

private:
    [[nodiscard]] bool is_live(const BufferHandle handle) const noexcept {
        if (handle.slot >= Capacity) {
            return false;
        }
        const auto& slot = slots_[handle.slot];
        return slot.leased.load(std::memory_order_acquire) &&
               slot.generation.load(std::memory_order_acquire) ==
                   handle.generation;
    }

    std::array<Slot, Capacity> slots_{};
    SpscRing<std::uint32_t, Capacity> free_{};
};

}  // namespace streamglass

