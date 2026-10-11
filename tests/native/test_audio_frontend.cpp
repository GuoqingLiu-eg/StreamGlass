#include "streamglass/online_fbank.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    streamglass::OfficialOnlineFbank frontend(0.0F);
    std::array<float, 5'120> samples{};
    std::array<
        float,
        streamglass::OfficialOnlineFbank::kMaxOutputFrames *
            streamglass::OfficialOnlineFbank::kFeatureDimension>
        features{};

    const auto first = frontend.extract(samples, features);
    require(first == 30, "first 320ms block must emit 30 frames");
    require(frontend.residual_samples() == 320, "first residual must be 20ms");

    const auto second = frontend.extract(samples, features);
    require(second == 32, "second 320ms block must emit 32 frames");
    require(frontend.residual_samples() == 320, "steady residual must be 20ms");
    require(
        std::all_of(
            features.begin(),
            features.begin() + static_cast<std::ptrdiff_t>(
                                   second *
                                   streamglass::OfficialOnlineFbank::
                                       kFeatureDimension),
            [](const float value) { return std::isfinite(value); }),
        "fbank output must be finite");

    frontend.reset();
    require(frontend.residual_samples() == 0, "reset must clear residual");
    std::array<float, 100> short_first{};
    std::array<float, 300> short_second{};
    require(frontend.extract(short_first, features) == 0, "short block emitted");
    require(frontend.extract(short_second, features) == 1, "25ms emitted no frame");

    std::cout << "streamglass audio frontend tests passed\n";
    return 0;
}
