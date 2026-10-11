#include "streamglass/online_fbank.hpp"

#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/feature-window.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace streamglass {

class OfficialOnlineFbank::Implementation final {
public:
    explicit Implementation(const float dither)
        : options_(make_options(dither)),
          computer_(options_),
          window_(computer_.GetFrameOptions()) {
        signal_frame_.reserve(
            static_cast<std::size_t>(
                computer_.GetFrameOptions().PaddedWindowSize()));
    }

    void reset() noexcept { residual_size_ = 0; }

    std::size_t extract(
        const std::span<const float> samples,
        const std::span<float> output_features) {
        if (samples.size() > kMaxInputSamples) {
            throw std::length_error("fbank input exceeds the preallocated block");
        }

        const auto total_samples = residual_size_ + samples.size();
        std::copy_n(residual_.begin(), residual_size_, combined_.begin());
        std::copy(
            samples.begin(),
            samples.end(),
            combined_.begin() + static_cast<std::ptrdiff_t>(residual_size_));

        if (total_samples < kWindowSamples) {
            std::copy_n(combined_.begin(), total_samples, residual_.begin());
            residual_size_ = total_samples;
            return 0;
        }

        const auto frame_count =
            (total_samples - (kWindowSamples - kShiftSamples)) / kShiftSamples;
        if (frame_count > kMaxOutputFrames) {
            throw std::logic_error("fbank frame count exceeds its fixed bound");
        }
        const auto required_output = frame_count * kFeatureDimension;
        if (output_features.size() < required_output) {
            throw std::length_error("fbank output span is too small");
        }

        const auto consumed_samples = frame_count * kShiftSamples;
        residual_size_ = total_samples - consumed_samples;
        std::copy_n(
            combined_.begin() + static_cast<std::ptrdiff_t>(consumed_samples),
            residual_size_,
            residual_.begin());

        const auto effective_samples =
            consumed_samples + (kWindowSamples - kShiftSamples);
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            signal_frame_.clear();
            knf::ExtractWindow(
                0,
                combined_.data(),
                effective_samples,
                static_cast<std::int32_t>(frame),
                options_.frame_opts,
                window_,
                &signal_frame_);
            computer_.Compute(
                0.0F,
                1.0F,
                &signal_frame_,
                output_features.data() + frame * kFeatureDimension);
        }
        return frame_count;
    }

    [[nodiscard]] std::size_t residual_samples() const noexcept {
        return residual_size_;
    }

private:
    static knf::FbankOptions make_options(const float dither) {
        knf::FbankOptions options{};
        options.frame_opts.samp_freq = static_cast<float>(kSampleRate);
        options.frame_opts.frame_length_ms = 25.0F;
        options.frame_opts.frame_shift_ms = 10.0F;
        options.frame_opts.dither = dither;
        options.mel_opts.num_bins = static_cast<std::int32_t>(kFeatureDimension);
        return options;
    }

    knf::FbankOptions options_;
    knf::FbankComputer computer_;
    knf::FeatureWindowFunction window_;
    std::array<float, kWindowSamples> residual_{};
    std::size_t residual_size_ = 0;
    std::array<float, kMaxInputSamples + kWindowSamples> combined_{};
    std::vector<float> signal_frame_;
};

OfficialOnlineFbank::OfficialOnlineFbank(const float dither)
    : implementation_(std::make_unique<Implementation>(dither)) {}

OfficialOnlineFbank::~OfficialOnlineFbank() = default;
OfficialOnlineFbank::OfficialOnlineFbank(OfficialOnlineFbank&&) noexcept = default;
OfficialOnlineFbank& OfficialOnlineFbank::operator=(
    OfficialOnlineFbank&&) noexcept = default;

void OfficialOnlineFbank::reset() noexcept { implementation_->reset(); }

std::size_t OfficialOnlineFbank::extract(
    const std::span<const float> samples,
    const std::span<float> output_features) {
    return implementation_->extract(samples, output_features);
}

std::size_t OfficialOnlineFbank::residual_samples() const noexcept {
    return implementation_->residual_samples();
}

}  // namespace streamglass
