// Realtime glasses runner (v1 scaffold).
//
// stdin:  f32le mono 16 kHz capture (pw-record)
// stdout: f32le mono 16 kHz playback (pw-play)
//
// Deliberate v1 simplifications, to be replaced by in-process PipeWire
// streams: audio I/O rides on pipes, VAD is an energy gate behind the
// IVoiceActivityDetector interface (Silero drop-in later), and echo is
// handled by a half-duplex gate (no new utterance may start while
// translated audio is still playing).
#include "streamglass/bluez_media_controller.hpp"
#include "streamglass/hfp_button_monitor.hpp"
#include "streamglass/pause_control.hpp"
#include "streamglass/playback_queue.hpp"
#include "streamglass/runtime_control.hpp"
#include "streamglass/streaming_pipeline.hpp"
#include "streamglass/utterance_segmenter.hpp"

#ifdef STREAMGLASS_HAS_SILERO_VAD
#include "streamglass/silero_vad_trt.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string environment_or(const char* name, const std::string& fallback) {
    const auto* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
}

std::int64_t monotonic_now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void extend_playback_end(
    std::atomic<std::int64_t>& playback_end_ns,
    const std::size_t sample_count) noexcept {
    constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
    constexpr std::int64_t kSampleRate = 16'000LL;
    const auto duration_ns =
        static_cast<std::int64_t>(sample_count) * kNanosecondsPerSecond /
        kSampleRate;

    auto observed = playback_end_ns.load(std::memory_order_relaxed);
    while (true) {
        const auto desired =
            std::max(observed, monotonic_now_ns()) + duration_ns;
        if (playback_end_ns.compare_exchange_weak(
                observed, desired,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

class EnergyVad final : public streamglass::IVoiceActivityDetector {
public:
    float probability(const std::span<const float> window) override {
        double energy = 0.0;
        for (const auto sample : window) {
            energy += static_cast<double>(sample) *
                      static_cast<double>(sample);
        }
        const auto rms = std::sqrt(energy / static_cast<double>(window.size()));
        return static_cast<float>(std::clamp(rms, 0.0, 1.0));
    }
    void reset() override {}
};

#ifdef STREAMGLASS_HAS_SILERO_VAD
class SileroVadAdapter final
    : public streamglass::IVoiceActivityDetector {
public:
    explicit SileroVadAdapter(const std::filesystem::path& engine_path)
        : runtime_(engine_path) {}

    float probability(const std::span<const float> window) override {
        return runtime_.run(window);
    }

    void reset() override { runtime_.reset(); }

private:
    streamglass::SileroVadTrt runtime_;
};
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        // The playback worker can block in fwrite(stdout) while PipeWire
        // consumes at the device clock. cerr is tied to cout by default, so a
        // log write from the capture thread would otherwise try to flush the
        // same stdout FILE lock and deadlock the capture path.
        std::cerr.tie(nullptr);

        if (argc != 4) {
            std::cerr << "usage: streamglass-native-realtime "
                         "<engine-dir> <runtime-assets> <target-language>\n";
            return 2;
        }
        const std::filesystem::path engine_dir(argv[1]);
        const std::filesystem::path runtime_assets(argv[2]);
        const std::string target_language(argv[3]);

        const streamglass::StreamingPipelinePaths paths{
            .encoder = engine_dir / environment_or(
                "STREAMGLASS_ENCODER_ENGINE_NAME",
                "encoder_cumulative_fp16.engine"),
            .decoder_prefill = engine_dir / environment_or(
                "STREAMGLASS_DECODER_PREFILL_ENGINE_NAME",
                "decoder_prefill_mixed.engine"),
            .decoder_step = engine_dir / environment_or(
                "STREAMGLASS_DECODER_STEP_ENGINE_NAME",
                "decoder_step_mixed.engine"),
            .t2u_frontend = engine_dir / "t2u_frontend_fp16.engine",
            .t2u_decoder = engine_dir / "t2u_decoder_fp32.engine",
            .vocoder = engine_dir / "vocoder_fp16.engine",
            .runtime_assets = runtime_assets,
        };
        const streamglass::StreamingPipelineOptions options{
            .cached_decoder =
                environment_or("STREAMGLASS_DECODER_CACHED", "0") == "1",
        };

        std::cerr << "[realtime] loading engines...\n";
        streamglass::StreamingPipeline pipeline(
            paths, target_language, options);
        streamglass::MappedHostBuffer<float> synthesis(
            streamglass::VocoderRuntime::kMaximumSampleCount);
        streamglass::PlaybackQueue playback;

        const bool half_duplex =
            environment_or("STREAMGLASS_HALF_DUPLEX", "1") == "1";
        const auto playback_drain_ms = std::max(
            0.0,
            std::stod(environment_or(
                "STREAMGLASS_PLAYBACK_DRAIN_MS", "500")));
        const auto playback_drain_ns = static_cast<std::int64_t>(
            playback_drain_ms * 1'000'000.0);
        const auto vad_start_rms = static_cast<float>(std::stod(
            environment_or("STREAMGLASS_VAD_START_RMS", "0.030")));
        const auto vad_continue_rms = static_cast<float>(std::stod(
            environment_or("STREAMGLASS_VAD_CONTINUE_RMS", "0.018")));
        const auto vad_start_ms = std::stoi(environment_or(
            "STREAMGLASS_VAD_START_MS", "160"));
        const auto vad_preroll_ms = std::stoi(environment_or(
            "STREAMGLASS_VAD_PREROLL_MS", "160"));
        const auto vad_silence_ms = std::stoi(environment_or(
            "STREAMGLASS_VAD_SILENCE_MS", "620"));
        if (vad_start_rms <= 0.0F || vad_continue_rms <= 0.0F ||
            vad_continue_rms > vad_start_rms) {
            throw std::invalid_argument(
                "VAD RMS values must satisfy 0 < continue <= start");
        }
        if (vad_start_ms < 0 || vad_preroll_ms < 0 ||
            vad_silence_ms < 32) {
            throw std::invalid_argument(
                "VAD durations are invalid (silence must be at least 32 ms)");
        }
        const auto silero_engine = environment_or(
            "STREAMGLASS_SILERO_PRIMARY_ENGINE", "");
        float vad_start_threshold = vad_start_rms;
        float vad_continue_threshold = vad_continue_rms;
        std::string vad_name = "energy-rms";
        std::unique_ptr<streamglass::IVoiceActivityDetector> vad;
        if (!silero_engine.empty()) {
#ifdef STREAMGLASS_HAS_SILERO_VAD
            vad_start_threshold = static_cast<float>(std::stod(
                environment_or(
                    "STREAMGLASS_SILERO_START_THRESHOLD", "0.50")));
            vad_continue_threshold = static_cast<float>(std::stod(
                environment_or(
                    "STREAMGLASS_SILERO_CONTINUE_THRESHOLD", "0.35")));
            if (vad_start_threshold <= 0.0F ||
                vad_start_threshold > 1.0F ||
                vad_continue_threshold <= 0.0F ||
                vad_continue_threshold > vad_start_threshold) {
                throw std::invalid_argument(
                    "Silero thresholds must satisfy "
                    "0 < continue <= start <= 1");
            }
            vad = std::make_unique<SileroVadAdapter>(silero_engine);
            vad_name = "silero-tensorrt";
#else
            throw std::runtime_error(
                "Silero primary VAD was requested, but this binary was "
                "built without Silero TensorRT support");
#endif
        } else {
            vad = std::make_unique<EnergyVad>();
        }
        const auto utterance_hard_cap_ms = std::stoi(environment_or(
            "STREAMGLASS_UTTERANCE_HARD_CAP_MS", "10000"));
        if (utterance_hard_cap_ms < 320) {
            throw std::invalid_argument(
                "STREAMGLASS_UTTERANCE_HARD_CAP_MS must be at least 320");
        }
        const auto playback_enqueue_timeout_ms = std::stoi(environment_or(
            "STREAMGLASS_PLAYBACK_ENQUEUE_TIMEOUT_MS", "5000"));
        if (playback_enqueue_timeout_ms < 0) {
            throw std::invalid_argument(
                "STREAMGLASS_PLAYBACK_ENQUEUE_TIMEOUT_MS must not be negative");
        }

        streamglass::PauseControl pause_control;
        const bool start_paused =
            environment_or("STREAMGLASS_START_PAUSED", "0") == "1";
        if (start_paused) {
            static_cast<void>(pause_control.set_paused(true));
        }
        streamglass::RuntimeControl runtime_control(
            pause_control,
            target_language,
            streamglass::default_runtime_control_socket());

        std::uint32_t write_index = 0;
        std::uint64_t dropped_by_gate = 0;
        std::uint64_t processed_utterances = 0;

        constexpr std::size_t kMinimumUtteranceSamples = 36'000;
        std::size_t utterance_samples = 0;
        auto run_step = [&](const std::span<const float> samples,
                            const bool finished,
                            const std::uint64_t utterance) {
            const auto step = pipeline.process_audio(samples, finished);
            if (step.output_unit_count > 0) {
                const auto result = pipeline.synthesize(step, synthesis);
                const auto status = playback.push_wait(
                    utterance,
                    write_index,
                    std::span<const float>(
                        synthesis.data(), result.sample_count),
                    finished && step.t2u_finished,
                    std::chrono::milliseconds(playback_enqueue_timeout_ms));
                std::cerr << "[playback] push utterance=" << utterance
                          << " samples=" << result.sample_count
                          << " status=" << static_cast<int>(status)
                          << "\n";
                if (status == streamglass::PlaybackPushStatus::accepted) {
                    ++write_index;
                } else {
                    playback.reset();
                    write_index = 0;
                    throw std::runtime_error(
                        "playback enqueue failed with status=" +
                        std::to_string(static_cast<int>(status)));
                }
            }
            return step;
        };
        auto sink = [&](const streamglass::SegmentEvent& event) {
            try {
                if (event.finished &&
                    utterance_samples + event.samples.size() <
                        kMinimumUtteranceSamples) {
                    // Pad short utterances with silence up to the encoder
                    // profile minimum instead of dropping the sentence; the
                    // padding steps may WRITE and must reach playback too.
                    const std::vector<float> silence(5'120, 0.0F);
                    while (utterance_samples + event.samples.size() <
                           kMinimumUtteranceSamples) {
                        static_cast<void>(run_step(
                            silence, false, event.utterance_id));
                        utterance_samples += silence.size();
                    }
                }
                utterance_samples += event.samples.size();
                const auto step = run_step(
                    event.samples, event.finished, event.utterance_id);
                if (step.text_action ==
                        streamglass::TextPolicyAction::write &&
                    !event.finished) {
                    runtime_control.record_translation(
                        pipeline.translation(),
                        event.utterance_id,
                        false,
                        processed_utterances);
                }
                if (event.finished) {
                    const auto translation = pipeline.translation();
                    std::cerr << "[realtime] utterance "
                              << event.utterance_id << ": "
                              << translation << "\n";
                    pipeline.reset();
                    write_index = 0;
                    utterance_samples = 0;
                    ++processed_utterances;
                    runtime_control.record_translation(
                        translation,
                        event.utterance_id,
                        true,
                        processed_utterances);
                }
            } catch (const std::exception& error) {
                // Typically: utterance shorter than the encoder profile.
                std::cerr << "[realtime] utterance "
                          << event.utterance_id
                          << " skipped: " << error.what() << "\n";
                pipeline.reset();
                write_index = 0;
                utterance_samples = 0;
            }
        };

        auto segmenter_config = streamglass::SegmenterConfig{};
        segmenter_config.start_speech_threshold = vad_start_threshold;
        segmenter_config.speech_threshold = vad_continue_threshold;
        segmenter_config.start_speech_ms =
            static_cast<std::uint32_t>(vad_start_ms);
        segmenter_config.preroll_ms =
            static_cast<std::uint32_t>(vad_preroll_ms);
        segmenter_config.silence_limit_ms =
            static_cast<std::uint32_t>(vad_silence_ms);
        segmenter_config.hard_cap_ms =
            static_cast<std::uint32_t>(utterance_hard_cap_ms);
        streamglass::UtteranceSegmenter segmenter(
            *vad, segmenter_config, sink);

        std::unique_ptr<streamglass::HfpButtonMonitor>
            hfp_button_monitor;
        if (environment_or(
                "STREAMGLASS_HFP_BUTTON_CONTROL", "1") == "1") {
            const auto retry_suppression_ms = std::max(
                0,
                std::stoi(environment_or(
                    "STREAMGLASS_HFP_BUTTON_RETRY_MS", "3500")));
            hfp_button_monitor =
                std::make_unique<streamglass::HfpButtonMonitor>(
                    pause_control,
                    static_cast<std::uint32_t>(
                        retry_suppression_ms));
            hfp_button_monitor->start();
        }
        std::unique_ptr<streamglass::BluezMediaController>
            media_controller;
        if (environment_or("STREAMGLASS_AVRCP_CONTROL", "0") == "1") {
            media_controller =
                std::make_unique<streamglass::BluezMediaController>(
                    pause_control,
                    environment_or(
                        "STREAMGLASS_BT_CONTROLLER",
                        "04:7F:0E:11:36:75"),
                    environment_or(
                        "STREAMGLASS_BT_DEVICE",
                        "41:42:84:69:36:F8"));
            media_controller->start();
        }
        runtime_control.start();

        std::atomic<bool> running{true};
        std::atomic<std::int64_t> playback_end_ns{0};
        std::thread playback_thread([&] {
            // Keep pipe writes close to the 32 ms audio callback cadence. Do
            // not prefill the pipe with silence: that creates hidden latency
            // and can indefinitely postpone the first translated samples.
            std::vector<float> buffer(512);
            std::uint64_t samples_written = 0;
            while (running.load(std::memory_order_relaxed)) {
                const auto sample_count =
                    playback.pull(std::span<float>(buffer));
                if (sample_count == 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(2));
                    continue;
                }
                if (std::fwrite(buffer.data(), sizeof(float),
                                sample_count, stdout) != sample_count) {
                    running.store(false, std::memory_order_relaxed);
                    break;
                }
                std::fflush(stdout);
                extend_playback_end(playback_end_ns, sample_count);
                if (samples_written == 0) {
                    std::cerr << "[playback] first PCM samples="
                              << sample_count << "\n";
                }
                samples_written += sample_count;
            }
            std::cerr << "[playback] stdout samples="
                      << samples_written << "\n";
        });

        std::cerr << "[realtime] ready (half_duplex="
                  << (half_duplex ? "on" : "off")
                  << ", vad=" << vad_name
                  << ", vad_start_threshold=" << vad_start_threshold
                  << ", vad_continue_threshold=" << vad_continue_threshold
                  << ", vad_start_ms=" << vad_start_ms
                  << ", vad_preroll_ms=" << vad_preroll_ms
                  << ", vad_silence_ms=" << vad_silence_ms
                  << ", playback_drain_ms=" << playback_drain_ms
                  << ", utterance_hard_cap_ms="
                  << utterance_hard_cap_ms
                  << ")\n";
        std::vector<float> block(512);
        bool playback_gate_active = false;
        bool pause_applied = start_paused;
        std::uint64_t pause_transitions = 0;
        std::uint64_t dropped_while_paused = 0;
        std::uint64_t applied_language_revision = 0;
        if (pause_applied) {
            std::cerr << "[control] restored paused state\n";
        }
        while (running.load(std::memory_order_relaxed)) {
            const auto read = std::fread(
                block.data(), sizeof(float), block.size(), stdin);
            if (read == 0) {
                break;
            }
            if (const auto language_request =
                    runtime_control.language_request_after(
                        applied_language_revision);
                language_request.has_value()) {
                try {
                    // Language changes are utterance boundaries. The capture
                    // thread owns every mutable runtime object, so no model
                    // state is touched by the control socket thread.
                    segmenter.reset();
                    pipeline.set_target_language(
                        language_request->language);
                    playback.reset();
                    write_index = 0;
                    utterance_samples = 0;
                    playback_end_ns.store(0, std::memory_order_relaxed);
                    playback_gate_active = false;
                    applied_language_revision =
                        language_request->revision;
                    runtime_control.mark_language_applied(
                        language_request->language,
                        language_request->revision);
                    std::cerr << "[control] target language="
                              << language_request->language
                              << " revision="
                              << language_request->revision << "\n";
                } catch (const std::exception& error) {
                    applied_language_revision =
                        language_request->revision;
                    runtime_control.mark_language_failed(
                        error.what(), language_request->revision);
                    std::cerr << "[control] language switch failed: "
                              << error.what() << "\n";
                }
                continue;
            }
            const auto pause_requested = pause_control.paused();
            if (pause_requested != pause_applied) {
                // The D-Bus thread only updates PauseControl. All mutable
                // runtime state stays owned by this capture thread.
                segmenter.reset();
                pipeline.reset();
                playback.reset();
                write_index = 0;
                utterance_samples = 0;
                playback_end_ns.store(0, std::memory_order_relaxed);
                playback_gate_active = false;
                pause_applied = pause_requested;
                ++pause_transitions;
                std::cerr << "[control] translation "
                          << (pause_applied ? "paused" : "resumed")
                          << " revision=" << pause_control.revision()
                          << "\n";
                dropped_while_paused += read;
                continue;
            }
            if (pause_applied) {
                dropped_while_paused += read;
                continue;
            }
            const auto samples =
                std::span<const float>(block.data(), read);
            static double meter_energy = 0.0;
            static std::size_t meter_count = 0;
            for (std::size_t index = 0; index < read; ++index) {
                meter_energy += static_cast<double>(block[index]) *
                                static_cast<double>(block[index]);
            }
            meter_count += read;
            if (meter_count >= 16'000) {
                std::cerr << "[meter] rms="
                          << std::sqrt(meter_energy /
                                       static_cast<double>(meter_count))
                          << (segmenter.active() ? " ACTIVE" : "")
                          << "\n";
                meter_energy = 0.0;
                meter_count = 0;
            }
            const auto buffered_ms = playback.buffered_ms();
            const auto media_end_ns =
                playback_end_ns.load(std::memory_order_relaxed);
            const bool device_playback_pending =
                media_end_ns > 0 &&
                monotonic_now_ns() < media_end_ns + playback_drain_ns;
            const bool hold_capture =
                half_duplex && !segmenter.active() &&
                (buffered_ms > 1.0 || device_playback_pending);
            if (hold_capture) {
                if (!playback_gate_active) {
                    std::cerr << "[gate] playback hold buffered_ms="
                              << buffered_ms
                              << " drain_ms=" << playback_drain_ms
                              << "\n";
                    playback_gate_active = true;
                }
                dropped_by_gate += read;
                continue;
            }
            if (playback_gate_active) {
                std::cerr << "[gate] playback released\n";
                playback_gate_active = false;
            }
            segmenter.push(samples);
        }

        running.store(false, std::memory_order_relaxed);
        runtime_control.stop();
        if (hfp_button_monitor != nullptr) {
            hfp_button_monitor->stop();
        }
        if (media_controller != nullptr) {
            media_controller->stop();
        }
        playback_thread.join();
        std::cerr << "[realtime] done utterances=" << processed_utterances
                  << " gate_dropped_samples=" << dropped_by_gate
                  << " paused_dropped_samples=" << dropped_while_paused
                  << " pause_transitions=" << pause_transitions
                  << " gaps=" << playback.stats().gap_events << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
