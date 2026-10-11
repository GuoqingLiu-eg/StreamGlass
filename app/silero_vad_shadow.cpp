#include "streamglass/silero_vad_trt.hpp"
#include "streamglass/utterance_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string environment_or(const char* name, const std::string& fallback) {
    const auto* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
}

class EnergyVad final : public streamglass::IVoiceActivityDetector {
public:
    float probability(const std::span<const float> window) override {
        double energy = 0.0;
        for (const auto sample : window) {
            energy += static_cast<double>(sample) * static_cast<double>(sample);
        }
        last_probability_ = static_cast<float>(
            std::clamp(std::sqrt(energy / static_cast<double>(window.size())),
                       0.0, 1.0));
        return last_probability_;
    }

    void reset() override {}
    [[nodiscard]] float last_probability() const noexcept {
        return last_probability_;
    }

private:
    float last_probability_ = 0.0F;
};

class SileroVadAdapter final : public streamglass::IVoiceActivityDetector {
public:
    explicit SileroVadAdapter(const std::filesystem::path& engine_path)
        : runtime_(engine_path) {}

    float probability(const std::span<const float> window) override {
        last_probability_ = runtime_.run(window);
        return last_probability_;
    }

    void reset() override { runtime_.reset(); }
    [[nodiscard]] float last_probability() const noexcept {
        return last_probability_;
    }

private:
    streamglass::SileroVadTrt runtime_;
    float last_probability_ = 0.0F;
};

struct DetectorStats {
    std::size_t emitted_samples = 0;
    std::uint64_t completed_utterances = 0;
};

streamglass::SegmenterConfig make_config(
    const float start_threshold,
    const float continue_threshold) {
    streamglass::SegmenterConfig config;
    config.start_speech_threshold = start_threshold;
    config.speech_threshold = continue_threshold;
    config.start_speech_ms = static_cast<std::uint32_t>(std::stoul(
        environment_or("STREAMGLASS_VAD_START_MS", "144")));
    config.preroll_ms = static_cast<std::uint32_t>(std::stoul(
        environment_or("STREAMGLASS_VAD_PREROLL_MS", "192")));
    config.silence_limit_ms = static_cast<std::uint32_t>(std::stoul(
        environment_or("STREAMGLASS_VAD_SILENCE_MS", "544")));
    config.hard_cap_ms = static_cast<std::uint32_t>(std::stoul(
        environment_or("STREAMGLASS_UTTERANCE_HARD_CAP_MS", "12800")));
    return config;
}

streamglass::UtteranceSegmenter::Sink make_sink(
    const std::string& detector, DetectorStats& stats) {
    return [detector, &stats](const streamglass::SegmentEvent& event) {
        stats.emitted_samples += event.samples.size();
        if (!event.finished) {
            return;
        }
        ++stats.completed_utterances;
        std::cerr << "[shadow-boundary] detector=" << detector
                  << " event=end utterance=" << event.utterance_id
                  << " samples=" << stats.emitted_samples
                  << " seconds=" << std::fixed << std::setprecision(3)
                  << static_cast<double>(stats.emitted_samples) / 16'000.0
                  << '\n';
        stats.emitted_samples = 0;
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cerr.tie(nullptr);
        if (argc != 2) {
            std::cerr << "usage: streamglass-silero-vad-shadow <engine>\n";
            return 2;
        }

        const auto energy_start = static_cast<float>(std::stod(
            environment_or("STREAMGLASS_VAD_START_RMS", "0.030")));
        const auto energy_continue = static_cast<float>(std::stod(
            environment_or("STREAMGLASS_VAD_CONTINUE_RMS", "0.016")));
        const auto silero_threshold = static_cast<float>(std::stod(
            environment_or("STREAMGLASS_SILERO_THRESHOLD", "0.5")));
        if (energy_start <= 0.0F || energy_continue <= 0.0F ||
            energy_continue > energy_start || silero_threshold <= 0.0F ||
            silero_threshold >= 1.0F) {
            throw std::invalid_argument("Shadow VAD thresholds are invalid");
        }

        EnergyVad energy_vad;
        SileroVadAdapter silero_vad(argv[1]);
        DetectorStats energy_stats;
        DetectorStats silero_stats;
        streamglass::UtteranceSegmenter energy_segmenter(
            energy_vad,
            make_config(energy_start, energy_continue),
            make_sink("energy", energy_stats));
        streamglass::UtteranceSegmenter silero_segmenter(
            silero_vad,
            make_config(silero_threshold, silero_threshold),
            make_sink("silero", silero_stats));

        std::cerr << "[shadow-vad] ready mode=observe-only"
                  << " energy_start=" << energy_start
                  << " energy_continue=" << energy_continue
                  << " silero_threshold=" << silero_threshold << '\n';

        std::vector<float> block(streamglass::SileroVadTrt::kChunkSamples);
        std::uint64_t windows = 0;
        std::uint64_t label_disagreements = 0;
        while (true) {
            const auto read = std::fread(
                block.data(), sizeof(float), block.size(), stdin);
            if (read == 0) {
                break;
            }
            if (read != block.size()) {
                std::cerr << "[shadow-vad] ignored_partial_samples=" << read
                          << '\n';
                break;
            }

            const auto energy_was_active = energy_segmenter.active();
            const auto silero_was_active = silero_segmenter.active();
            energy_segmenter.push(block);
            silero_segmenter.push(block);
            ++windows;

            const auto energy_threshold =
                energy_was_active ? energy_continue : energy_start;
            const auto energy_label =
                energy_vad.last_probability() > energy_threshold;
            const auto silero_label =
                silero_vad.last_probability() > silero_threshold;
            label_disagreements += energy_label != silero_label ? 1U : 0U;

            if (!energy_was_active && energy_segmenter.active()) {
                std::cerr << "[shadow-boundary] detector=energy event=start"
                          << " window=" << windows << '\n';
            }
            if (!silero_was_active && silero_segmenter.active()) {
                std::cerr << "[shadow-boundary] detector=silero event=start"
                          << " window=" << windows << '\n';
            }
            if (windows % 32U == 0U) {
                std::cerr << std::fixed << std::setprecision(5)
                          << "[shadow-vad] windows=" << windows
                          << " rms=" << energy_vad.last_probability()
                          << " silero=" << silero_vad.last_probability()
                          << " energy_active="
                          << (energy_segmenter.active() ? 1 : 0)
                          << " silero_active="
                          << (silero_segmenter.active() ? 1 : 0)
                          << " disagreement_rate="
                          << static_cast<double>(label_disagreements) /
                                 static_cast<double>(windows)
                          << '\n';
            }
        }

        std::cerr << "[shadow-summary] windows=" << windows
                  << " energy_utterances=" << energy_stats.completed_utterances
                  << " silero_utterances=" << silero_stats.completed_utterances
                  << " label_disagreements=" << label_disagreements << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
