#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace streamglass {

class OfficialOnlineFbank final {
public:
    static constexpr std::size_t kSampleRate = 16'000;
    static constexpr std::size_t kFeatureDimension = 80;
    static constexpr std::size_t kWindowSamples = 400;
    static constexpr std::size_t kShiftSamples = 160;
    static constexpr std::size_t kMaxInputSamples = 16'000;
    static constexpr std::size_t kMaxOutputFrames = 101;

    explicit OfficialOnlineFbank(float dither = 1.0F);
    ~OfficialOnlineFbank();

    OfficialOnlineFbank(const OfficialOnlineFbank&) = delete;
    OfficialOnlineFbank& operator=(const OfficialOnlineFbank&) = delete;
    OfficialOnlineFbank(OfficialOnlineFbank&&) noexcept;
    OfficialOnlineFbank& operator=(OfficialOnlineFbank&&) noexcept;

    void reset() noexcept;

    [[nodiscard]] std::size_t extract(
        std::span<const float> samples,
        std::span<float> output_features);

    [[nodiscard]] std::size_t residual_samples() const noexcept;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace streamglass
