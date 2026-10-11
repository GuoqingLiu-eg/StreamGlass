#pragma once

#include <memory>
#include <string>

namespace streamglass {

class PauseControl;

// Registers a local MPRIS player with BlueZ. Standard AVRCP Play, Pause, and
// PlayPause commands are converted into PauseControl state updates.
class BluezMediaController final {
public:
    BluezMediaController(
        PauseControl& pause_control,
        std::string controller_address,
        std::string device_address);
    ~BluezMediaController();

    BluezMediaController(const BluezMediaController&) = delete;
    BluezMediaController& operator=(const BluezMediaController&) = delete;

    void start();
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace streamglass
