#include "streamglass/utterance_segmenter.hpp"

#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class ScriptedVad final : public streamglass::IVoiceActivityDetector {
public:
    std::deque<float> probabilities;
    std::size_t resets = 0;

    float probability(std::span<const float>) override {
        if (probabilities.empty()) {
            return 0.0F;
        }
        const auto value = probabilities.front();
        probabilities.pop_front();
        return value;
    }
    void reset() override { ++resets; }

    void feed(const std::size_t windows, const float value) {
        for (std::size_t index = 0; index < windows; ++index) {
            probabilities.push_back(value);
        }
    }
};

struct Collected {
    std::vector<std::size_t> sizes;
    std::vector<std::uint64_t> ids;
    std::size_t total = 0;
    std::size_t finishes = 0;
};

void drive(streamglass::UtteranceSegmenter& segmenter,
           const std::size_t windows) {
    const std::vector<float> window(512, 0.1F);
    for (std::size_t index = 0; index < windows; ++index) {
        segmenter.push(window);
    }
}

}  // namespace

int main() {
    using streamglass::SegmenterConfig;
    using streamglass::UtteranceSegmenter;

    {   // Silence only: nothing is emitted.
        ScriptedVad vad;
        vad.feed(100, 0.0F);
        Collected got;
        UtteranceSegmenter segmenter(vad, SegmenterConfig{}, [&](const auto& e) {
            got.total += e.samples.size();
        });
        drive(segmenter, 100);
        require(got.total == 0, "silence emits nothing");
    }

    {   // Speech then 700 ms silence: chunks + finished flush, no sample loss.
        ScriptedVad vad;
        vad.feed(60, 0.9F);   // ~1.92 s speech
        vad.feed(22, 0.0F);   // 704 ms silence
        Collected got;
        UtteranceSegmenter segmenter(vad, SegmenterConfig{}, [&](const auto& e) {
            got.sizes.push_back(e.samples.size());
            got.ids.push_back(e.utterance_id);
            got.total += e.samples.size();
            if (e.finished) { ++got.finishes; }
        });
        drive(segmenter, 82);
        require(got.finishes == 1, "one finished flush");
        require(got.total == 82U * 512U, "all forwarded samples conserved");
        require(got.ids.front() == 1 && got.ids.back() == 1, "single id");
        for (std::size_t index = 0; index + 1 < got.sizes.size(); ++index) {
            require(got.sizes[index] == 5'120, "full 5120 chunks before end");
        }
        require(vad.resets == 1, "vad reset on utterance end");
    }

    {   // Soft limit: after 12 s speech only ~350 ms silence ends it.
        ScriptedVad vad;
        vad.feed(380, 0.9F);  // 12.16 s speech
        vad.feed(11, 0.0F);   // 352 ms silence
        std::size_t finishes = 0;
        UtteranceSegmenter segmenter(vad, SegmenterConfig{}, [&](const auto& e) {
            if (e.finished) { ++finishes; }
        });
        drive(segmenter, 391);
        require(finishes == 1, "halved silence limit ends the utterance");
    }

    {   // Hard cap: continuous speech is split, nothing dropped.
        ScriptedVad vad;
        vad.feed(800, 0.9F);  // 25.6 s continuous speech
        vad.feed(22, 0.0F);   // let the second utterance finish
        Collected got;
        UtteranceSegmenter segmenter(vad, SegmenterConfig{}, [&](const auto& e) {
            got.total += e.samples.size();
            got.ids.push_back(e.utterance_id);
            if (e.finished) { ++got.finishes; }
        });
        drive(segmenter, 822);
        require(got.finishes == 2, "hard cap splits into two finishes");
        require(got.ids.back() == 2, "speech continues in a new utterance");
        require(got.total == 822U * 512U, "hard cap drops no samples");
    }

    {   // Pre-roll: enabled config prepends buffered silence.
        ScriptedVad vad;
        vad.feed(20, 0.0F);
        vad.feed(4, 0.9F);
        std::size_t first_size = 0;
        SegmenterConfig config;
        config.preroll_ms = 320;
        UtteranceSegmenter segmenter(vad, config, [&](const auto& e) {
            if (first_size == 0) { first_size = e.samples.size(); }
        });
        drive(segmenter, 24);
        require(first_size == 5'120,
                "pre-roll fills the first chunk to a full 5120");
    }

    {   // A transient below the configured onset duration is discarded.
        ScriptedVad vad;
        vad.feed(4, 0.0F);
        vad.feed(4, 0.9F);   // 128 ms, shorter than the 160 ms onset.
        vad.feed(30, 0.0F);
        Collected got;
        SegmenterConfig config;
        config.start_speech_ms = 160;
        config.preroll_ms = 160;
        UtteranceSegmenter segmenter(vad, config, [&](const auto& e) {
            got.total += e.samples.size();
            if (e.finished) { ++got.finishes; }
        });
        drive(segmenter, 38);
        require(got.total == 0, "short onset is not admitted");
        require(got.finishes == 0, "short onset creates no utterance");
    }

    {   // Confirmed onset keeps both pre-roll and the first speech samples.
        ScriptedVad vad;
        vad.feed(5, 0.0F);
        vad.feed(5, 0.9F);   // 160 ms confirms the utterance.
        vad.feed(22, 0.0F);  // 704 ms ends it.
        Collected got;
        SegmenterConfig config;
        config.start_speech_ms = 160;
        config.preroll_ms = 160;
        UtteranceSegmenter segmenter(vad, config, [&](const auto& e) {
            got.total += e.samples.size();
            if (e.finished) { ++got.finishes; }
        });
        drive(segmenter, 32);
        require(got.finishes == 1, "confirmed onset creates one utterance");
        require(got.total == 32U * 512U,
                "confirmed onset preserves pre-roll and speech onset");
    }

    {   // Lower continuation threshold preserves quiet words after onset.
        ScriptedVad vad;
        vad.feed(5, 0.8F);   // Above the high start threshold.
        vad.feed(20, 0.2F);  // Below start, above continuation threshold.
        vad.feed(22, 0.0F);
        Collected got;
        SegmenterConfig config;
        config.start_speech_threshold = 0.7F;
        config.speech_threshold = 0.1F;
        config.start_speech_ms = 160;
        UtteranceSegmenter segmenter(vad, config, [&](const auto& e) {
            got.total += e.samples.size();
            if (e.finished) { ++got.finishes; }
        });
        drive(segmenter, 47);
        require(got.finishes == 1,
                "quiet continuation remains in one utterance");
        require(got.total == 47U * 512U,
                "quiet continuation samples are preserved");
    }

    std::cout << "utterance segmenter tests passed\n";
    return 0;
}
