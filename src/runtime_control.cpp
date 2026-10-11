#include "streamglass/runtime_control.hpp"

#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace streamglass {
namespace {

constexpr std::size_t kMaximumCommandBytes = 256;

std::string trim(std::string_view value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

std::string json_escape(const std::string_view value) {
    std::string output;
    output.reserve(value.size() + 8U);
    for (const auto character : value) {
        switch (character) {
        case '\\':
            output += "\\\\";
            break;
        case '"':
            output += "\\\"";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(character) < 0x20U) {
                output += '?';
            } else {
                output += character;
            }
            break;
        }
    }
    return output;
}

bool supported_language(const std::string_view language) {
    static const std::unordered_set<std::string> languages{
        "cmn", "eng", "fra", "deu", "spa", "ita",
        "jpn", "kor", "rus", "por", "arb", "vie",
    };
    return languages.contains(std::string(language));
}

std::string error_json(const std::string_view message) {
    return "{\"ok\":false,\"error\":\"" + json_escape(message) + "\"}\n";
}

void write_response(const int client, const std::string_view response) {
    std::size_t offset = 0;
    while (offset < response.size()) {
        const auto written = ::send(
            client,
            response.data() + offset,
            response.size() - offset,
            MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        offset += static_cast<std::size_t>(written);
    }
}

}  // namespace

std::filesystem::path default_runtime_control_socket() {
    if (const auto* configured =
            std::getenv("STREAMGLASS_CONTROL_SOCKET");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    if (const auto* runtime_directory = std::getenv("XDG_RUNTIME_DIR");
        runtime_directory != nullptr && runtime_directory[0] != '\0') {
        return std::filesystem::path(runtime_directory) /
               "streamglass/control.sock";
    }
    return std::filesystem::path("/tmp") /
           ("streamglass-" + std::to_string(::getuid()) + ".sock");
}

RuntimeControl::RuntimeControl(
    PauseControl& pause_control,
    std::string initial_language,
    std::filesystem::path socket_path)
    : pause_control_(pause_control),
      socket_path_(std::move(socket_path)),
      requested_language_(initial_language),
      current_language_(std::move(initial_language)) {}

RuntimeControl::~RuntimeControl() {
    stop();
}

void RuntimeControl::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    thread_ = std::thread([this] { serve(); });
}

void RuntimeControl::stop() noexcept {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

std::optional<LanguageChangeRequest>
RuntimeControl::language_request_after(const std::uint64_t revision) const {
    std::lock_guard lock(state_mutex_);
    if (language_revision_ <= revision) {
        return std::nullopt;
    }
    return LanguageChangeRequest{
        .language = requested_language_,
        .revision = language_revision_,
    };
}

void RuntimeControl::mark_language_applied(
    const std::string_view language,
    const std::uint64_t revision) {
    std::lock_guard lock(state_mutex_);
    current_language_ = language;
    applied_language_revision_ = revision;
    last_error_.clear();
}

void RuntimeControl::mark_language_failed(
    const std::string_view message,
    const std::uint64_t revision) {
    std::lock_guard lock(state_mutex_);
    applied_language_revision_ = revision;
    last_error_ = message;
}

void RuntimeControl::record_translation(
    const std::string_view translation,
    const std::uint64_t utterance_id,
    const bool is_final,
    const std::uint64_t processed_utterances) {
    std::lock_guard lock(state_mutex_);
    if (latest_translation_ == translation &&
        translation_utterance_ == utterance_id &&
        translation_final_ == is_final &&
        processed_utterances_ == processed_utterances) {
        return;
    }
    latest_translation_ = translation;
    translation_utterance_ = utterance_id;
    translation_final_ = is_final;
    processed_utterances_ = processed_utterances;
    ++translation_revision_;
}

std::string RuntimeControl::status_json() const {
    std::lock_guard lock(state_mutex_);
    const auto pending = language_revision_ > applied_language_revision_;
    return "{\"ok\":true,\"online\":true,\"paused\":" +
           std::string(pause_control_.paused() ? "true" : "false") +
           ",\"pauseRevision\":" +
           std::to_string(pause_control_.revision()) +
           ",\"targetLanguage\":\"" + json_escape(current_language_) +
           "\",\"requestedLanguage\":\"" +
           json_escape(requested_language_) +
           "\",\"languagePending\":" +
           std::string(pending ? "true" : "false") +
           ",\"languageRevision\":" +
           std::to_string(language_revision_) +
           ",\"processedUtterances\":" +
           std::to_string(processed_utterances_) +
           ",\"translationRevision\":" +
           std::to_string(translation_revision_) +
           ",\"translationUtterance\":" +
           std::to_string(translation_utterance_) +
           ",\"translationFinal\":" +
           std::string(translation_final_ ? "true" : "false") +
           ",\"latestTranslation\":\"" +
           json_escape(latest_translation_) +
           "\",\"lastError\":\"" + json_escape(last_error_) + "\"}\n";
}

std::string RuntimeControl::execute(const std::string_view raw_command) {
    const auto command = trim(raw_command);
    if (command == "status") {
        return status_json();
    }
    if (command == "pause") {
        static_cast<void>(pause_control_.set_paused(true));
        return status_json();
    }
    if (command == "resume") {
        static_cast<void>(pause_control_.set_paused(false));
        return status_json();
    }
    if (command == "toggle") {
        static_cast<void>(pause_control_.toggle());
        return status_json();
    }
    constexpr std::string_view prefix = "language ";
    if (command.starts_with(prefix)) {
        const auto language = trim(
            std::string_view(command).substr(prefix.size()));
        if (!supported_language(language)) {
            return error_json("unsupported target language");
        }
        {
            std::lock_guard lock(state_mutex_);
            if (language != requested_language_ ||
                language != current_language_) {
                requested_language_ = language;
                ++language_revision_;
                last_error_.clear();
            }
        }
        return status_json();
    }

    return error_json("unknown control command");
}

void RuntimeControl::serve() noexcept {
    int server = -1;
    try {
        std::filesystem::create_directories(socket_path_.parent_path());
        if (socket_path_.string().size() >= sizeof(sockaddr_un{}.sun_path)) {
            throw std::runtime_error("control socket path is too long");
        }
        std::filesystem::remove(socket_path_);

        server = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (server < 0) {
            throw std::runtime_error(
                "socket failed: " + std::string(std::strerror(errno)));
        }

        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        const auto socket_path = socket_path_.string();
        std::memcpy(
            address.sun_path,
            socket_path.c_str(),
            socket_path.size() + 1U);
        if (::bind(
                server,
                reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) < 0) {
            throw std::runtime_error(
                "bind failed: " + std::string(std::strerror(errno)));
        }
        if (::chmod(socket_path_.c_str(), S_IRUSR | S_IWUSR) < 0) {
            throw std::runtime_error(
                "chmod failed: " + std::string(std::strerror(errno)));
        }
        if (::listen(server, 8) < 0) {
            throw std::runtime_error(
                "listen failed: " + std::string(std::strerror(errno)));
        }
        std::fprintf(
            stderr,
            "[control] listening socket=%s\n",
            socket_path_.c_str());

        pollfd descriptor{
            .fd = server,
            .events = POLLIN,
            .revents = 0,
        };
        while (running_.load(std::memory_order_acquire)) {
            descriptor.revents = 0;
            const auto ready = ::poll(&descriptor, 1, 200);
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(
                    "poll failed: " + std::string(std::strerror(errno)));
            }
            if (ready == 0 || (descriptor.revents & POLLIN) == 0) {
                continue;
            }

            const auto client = ::accept(server, nullptr, nullptr);
            if (client < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(
                    "accept failed: " + std::string(std::strerror(errno)));
            }

            std::array<char, kMaximumCommandBytes + 1U> buffer{};
            const auto received =
                ::recv(client, buffer.data(), kMaximumCommandBytes, 0);
            if (received > 0) {
                const auto response = execute(std::string_view(
                    buffer.data(), static_cast<std::size_t>(received)));
                write_response(client, response);
            } else {
                write_response(client, error_json("empty control command"));
            }
            ::close(client);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[control] ERROR: %s\n", error.what());
    }

    if (server >= 0) {
        ::close(server);
    }
    std::error_code ignored;
    std::filesystem::remove(socket_path_, ignored);
}

}  // namespace streamglass
