#pragma once

#include <cstdint>
#include <memory>

namespace streamglass {

class PauseControl;

// Reads HFP headset-button commands from Linux's read-only HCI monitor
// channel. The monitor runs in its own control thread and only updates
// PauseControl; realtime model and audio state remain capture-thread owned.
class HfpButtonMonitor final {
public:
    HfpButtonMonitor(
        PauseControl& pause_control,
        std::uint32_t retry_suppression_ms);
    ~HfpButtonMonitor();

    HfpButtonMonitor(const HfpButtonMonitor&) = delete;
    HfpButtonMonitor& operator=(const HfpButtonMonitor&) = delete;

    void start();
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace streamglass
