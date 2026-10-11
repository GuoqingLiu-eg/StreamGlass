#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "streamglass/encoder_runtime.hpp"
#include "streamglass/online_fbank.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSegmentSamples = 5'120;

std::vector<float> read_waveform(const std::filesystem::path& path) {
    unsigned int channels = 0;
    unsigned int sample_rate = 0;
    drwav_uint64 frame_count = 0;
    float* data = drwav_open_file_and_read_pcm_frames_f32(
        path.c_str(), &channels, &sample_rate, &frame_count, nullptr);
    if (data == nullptr) {
        throw std::runtime_error("cannot decode WAV: " + path.string());
    }
    if (channels != 1 ||
        sample_rate != streamglass::OfficialOnlineFbank::kSampleRate) {
        drwav_free(data, nullptr);
        throw std::runtime_error("WAV must be mono 16kHz PCM");
    }
    if (frame_count > std::numeric_limits<std::size_t>::max()) {
        drwav_free(data, nullptr);
        throw std::length_error("WAV is too large for this process");
    }
    std::vector<float> waveform(
        data, data + static_cast<std::size_t>(frame_count));
    drwav_free(data, nullptr);
    return waveform;
}

template <typename T>
void write_binary(const std::filesystem::path& path, const T* data, std::size_t count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: streamglass-encoder-replay <input.wav> "
                         "<encoder.engine> <output-f16.raw>\n";
            return 2;
        }

        const auto waveform = read_waveform(argv[1]);
        streamglass::EncoderRuntime encoder(argv[2]);
        streamglass::OfficialOnlineFbank frontend;
        auto fbank = encoder.fbank_input();

        std::size_t sample_offset = 0;
        std::size_t valid_fbank_frames = 0;
        while (sample_offset < waveform.size()) {
            const auto sample_count =
                std::min(kSegmentSamples, waveform.size() - sample_offset);
            const auto output_offset =
                valid_fbank_frames *
                streamglass::OfficialOnlineFbank::kFeatureDimension;
            const auto new_frames = frontend.extract(
                std::span(waveform).subspan(sample_offset, sample_count),
                fbank.subspan(output_offset));
            valid_fbank_frames += new_frames;
            sample_offset += sample_count;
        }

        const auto padded_fbank_frames =
            (valid_fbank_frames + 1U) & ~std::size_t{1};
        const auto result =
            encoder.run(valid_fbank_frames, padded_fbank_frames);
        const auto output_elements =
            result.tensor_frames * streamglass::EncoderRuntime::kEncoderDimension;
        std::vector<__half> output(output_elements);
        encoder.copy_output(output, result.tensor_frames);

        const std::filesystem::path output_path(argv[3]);
        write_binary(output_path, output.data(), output.size());
        const auto length_path = output_path.string() + ".len.i64";
        const auto valid_output = static_cast<std::int64_t>(result.valid_frames);
        write_binary(length_path, &valid_output, 1);

        std::cout << "samples=" << waveform.size() << '\n'
                  << "valid_fbank_frames=" << valid_fbank_frames << '\n'
                  << "padded_fbank_frames=" << padded_fbank_frames << '\n'
                  << "encoder_tensor_frames=" << result.tensor_frames << '\n'
                  << "encoder_valid_frames=" << result.valid_frames << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "encoder replay failed: " << error.what() << '\n';
        return 1;
    }
}
