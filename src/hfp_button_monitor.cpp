#include "streamglass/hfp_button_monitor.hpp"

#include "streamglass/hfp_button_packet.hpp"
#include "streamglass/pause_control.hpp"

#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <span>
#include <thread>

namespace streamglass {

class HfpButtonMonitor::Impl final {
public:
    Impl(
        PauseControl& pause_control,
        const std::uint32_t retry_suppression_ms)
        : pause_control_(pause_control),
          retry_suppression_(
              std::chrono::milliseconds(retry_suppression_ms)) {}

    ~Impl() { stop(); }

    void start() {
        if (worker_.joinable()) {
            return;
        }
        stop_requested_.store(false, std::memory_order_release);
        worker_ = std::thread([this] { run(); });
    }

    void stop() noexcept {
        stop_requested_.store(true, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    [[nodiscard]] bool wait_before_reopen() const noexcept {
        constexpr auto kRetrySlice = std::chrono::milliseconds(100);
        constexpr auto kRetryDelay = std::chrono::seconds(1);
        auto waited = std::chrono::milliseconds::zero();
        while (waited < kRetryDelay) {
            if (stop_requested_.load(std::memory_order_acquire)) {
                return false;
            }
            std::this_thread::sleep_for(kRetrySlice);
            waited += kRetrySlice;
        }
        return !stop_requested_.load(std::memory_order_acquire);
    }

    [[nodiscard]] int open_monitor_socket() const noexcept {
        const int socket_fd = ::socket(
            AF_BLUETOOTH,
            SOCK_RAW | SOCK_CLOEXEC | SOCK_NONBLOCK,
            BTPROTO_HCI);
        if (socket_fd < 0) {
            std::cerr << "[control] HFP button monitor unavailable: "
                      << std::strerror(errno)
                      << " (streamglass-native-realtime needs "
                         "cap_net_raw=ep)\n";
            return -1;
        }

        sockaddr_hci address{};
        address.hci_family = AF_BLUETOOTH;
        address.hci_dev = HCI_DEV_NONE;
        address.hci_channel = HCI_CHANNEL_MONITOR;
        if (::bind(
                socket_fd,
                reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) < 0) {
            std::cerr << "[control] HFP button monitor bind failed: "
                      << std::strerror(errno) << "\n";
            ::close(socket_fd);
            return -1;
        }

        return socket_fd;
    }

    void run() noexcept {
        auto suppress_until = std::chrono::steady_clock::time_point::min();
        while (!stop_requested_.load(std::memory_order_acquire)) {
            const int socket_fd = open_monitor_socket();
            if (socket_fd < 0) {
                if (!wait_before_reopen()) {
                    break;
                }
                continue;
            }

            std::cerr << "[control] HFP single-tap monitor active"
                      << " retry_suppression_ms="
                      << retry_suppression_.count() << "\n";

            std::array<std::uint8_t, 8192> packet{};
            bool reopen = false;
            while (!stop_requested_.load(std::memory_order_acquire)) {
                pollfd descriptor{
                    .fd = socket_fd,
                    .events = POLLIN,
                    .revents = 0,
                };
                const int ready = ::poll(&descriptor, 1, 200);
                if (ready < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    std::cerr
                        << "[control] HFP button monitor poll failed: "
                        << std::strerror(errno) << "\n";
                    reopen = true;
                    break;
                }
                if (ready == 0) {
                    continue;
                }
                if ((descriptor.revents &
                     (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                    std::cerr
                        << "[control] HFP button monitor transport changed; "
                           "reopening\n";
                    reopen = true;
                    break;
                }
                if ((descriptor.revents & POLLIN) == 0) {
                    continue;
                }

                const auto received =
                    ::recv(socket_fd, packet.data(), packet.size(), 0);
                if (received <= 0) {
                    if (received < 0 &&
                        errno != EAGAIN && errno != EWOULDBLOCK &&
                        errno != EINTR) {
                        std::cerr
                            << "[control] HFP button monitor receive failed: "
                            << std::strerror(errno) << "\n";
                        reopen = true;
                        break;
                    }
                    continue;
                }
                const auto parsed = parse_hfp_button_packet(
                    std::span<const std::uint8_t>(
                        packet.data(),
                        static_cast<std::size_t>(received)));
                if (!parsed.has_value()) {
                    continue;
                }

                const auto now = std::chrono::steady_clock::now();
                if (now < suppress_until) {
                    std::cerr
                        << "[control] HFP single-tap retry suppressed"
                        << " controller=" << parsed->controller_index
                        << " handle=" << parsed->connection_handle
                        << "\n";
                    continue;
                }
                suppress_until = now + retry_suppression_;
                const bool paused = pause_control_.toggle();
                std::cerr << "[control] HFP single-tap requested "
                          << (paused ? "pause" : "resume")
                          << " controller=" << parsed->controller_index
                          << " handle=" << parsed->connection_handle
                          << "\n";
            }
            ::close(socket_fd);
            if (reopen && !wait_before_reopen()) {
                break;
            }
        }
    }

    PauseControl& pause_control_;
    std::chrono::milliseconds retry_suppression_;
    std::atomic<bool> stop_requested_{false};
    std::thread worker_;
};

HfpButtonMonitor::HfpButtonMonitor(
    PauseControl& pause_control,
    const std::uint32_t retry_suppression_ms)
    : impl_(std::make_unique<Impl>(
          pause_control, retry_suppression_ms)) {}

HfpButtonMonitor::~HfpButtonMonitor() = default;

void HfpButtonMonitor::start() {
    impl_->start();
}

void HfpButtonMonitor::stop() noexcept {
    impl_->stop();
}

}  // namespace streamglass
