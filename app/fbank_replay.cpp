#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "streamglass/online_fbank.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kSegmentSamples = 5'120;

std::vector<float> read_waveform(const std::filesystem::path& path) {
    unsigned int channels = 0;
    unsigned int sample_rate = 0;
    drwav_uint64 frame_count = 0;
    float* interleaved = drwav_open_file_and_read_pcm_frames_f32(
        path.c_str(), &channels, &sample_rate, &frame_count, nullptr);
    if (interleaved == nullptr) {
        throw std::runtime_error("cannot decode WAV: " + path.string());
    }
    if (channels != 1 ||
        sample_rate != streamglass::OfficialOnlineFbank::kSampleRate) {
        drwav_free(interleaved, nullptr);
        throw std::runtime_error("WAV must be mono 16kHz PCM");
    }
    if (frame_count > std::numeric_limits<std::size_t>::max()) {
        drwav_free(interleaved, nullptr);
        throw std::length_error("WAV is too large for this process");
    }
    std::vector<float> waveform(
        interleaved, interleaved + static_cast<std::size_t>(frame_count));
    drwav_free(interleaved, nullptr);
    return waveform;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: streamglass-fbank-replay <input.wav> "
                         "<output-f32.raw>\n";
            return 2;
        }

        const auto waveform = read_waveform(argv[1]);
        streamglass::OfficialOnlineFbank frontend;
        std::array<
            float,
            streamglass::OfficialOnlineFbank::kMaxOutputFrames *
                streamglass::OfficialOnlineFbank::kFeatureDimension>
            block_features{};
        std::vector<float> features;
        features.reserve(
            (waveform.size() / streamglass::OfficialOnlineFbank::kShiftSamples +
             1) *
            streamglass::OfficialOnlineFbank::kFeatureDimension);

        std::size_t offset = 0;
        std::size_t segment_count = 0;
        while (offset < waveform.size()) {
            const auto count =
                std::min(kSegmentSamples, waveform.size() - offset);
            const auto frame_count = frontend.extract(
                std::span(waveform).subspan(offset, count), block_features);
            features.insert(
                features.end(),
                block_features.begin(),
                block_features.begin() + static_cast<std::ptrdiff_t>(
                                             frame_count *
                                             streamglass::OfficialOnlineFbank::
                                                 kFeatureDimension));
            offset += count;
            ++segment_count;
        }

        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        output.write(
            reinterpret_cast<const char*>(features.data()),
            static_cast<std::streamsize>(features.size() * sizeof(float)));
        if (!output) {
            throw std::runtime_error("cannot write fbank output");
        }

        std::cout << "samples=" << waveform.size() << '\n'
                  << "segments=" << segment_count << '\n'
                  << "valid_frames="
                  << features.size() /
                         streamglass::OfficialOnlineFbank::kFeatureDimension
                  << '\n'
                  << "residual_samples=" << frontend.residual_samples() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fbank replay failed: " << error.what() << '\n';
        return 1;
    }
}
