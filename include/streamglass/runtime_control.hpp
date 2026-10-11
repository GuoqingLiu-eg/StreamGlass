#pragma once

#include "streamglass/pause_control.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace streamglass {

struct LanguageChangeRequest {
    std::string language;
    std::uint64_t revision = 0;
};

// Local control plane for the realtime process. The socket thread only
// changes atomics or copies command data; model and audio state remain owned
// by the capture thread.
class RuntimeControl final {
public:
    RuntimeControl(
        PauseControl& pause_control,
        std::string initial_language,
        std::filesystem::path socket_path);
    ~RuntimeControl();

    RuntimeControl(const RuntimeControl&) = delete;
    RuntimeControl& operator=(const RuntimeControl&) = delete;

    void start();
    void stop() noexcept;

    [[nodiscard]] std::optional<LanguageChangeRequest>
    language_request_after(std::uint64_t revision) const;
    void mark_language_applied(
        std::string_view language,
        std::uint64_t revision);
    void mark_language_failed(
        std::string_view message,
        std::uint64_t revision);
    void record_translation(
        std::string_view translation,
        std::uint64_t utterance_id,
        bool is_final,
        std::uint64_t processed_utterances);

private:
    void serve() noexcept;
    [[nodiscard]] std::string execute(std::string_view command);
    [[nodiscard]] std::string status_json() const;

    PauseControl& pause_control_;
    std::filesystem::path socket_path_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    mutable std::mutex state_mutex_;
    std::string requested_language_;
    std::string current_language_;
    std::string latest_translation_;
    std::string last_error_;
    std::uint64_t language_revision_ = 0;
    std::uint64_t applied_language_revision_ = 0;
    std::uint64_t processed_utterances_ = 0;
    std::uint64_t translation_revision_ = 0;
    std::uint64_t translation_utterance_ = 0;
    bool translation_final_ = true;
};

[[nodiscard]] std::filesystem::path default_runtime_control_socket();

}  // namespace streamglass
