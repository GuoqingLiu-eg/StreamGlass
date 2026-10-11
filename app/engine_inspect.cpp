#include "streamglass/trt_engine.hpp"

#include <cuda_runtime_api.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::pair<std::string, std::filesystem::path> parse_engine_argument(
    const std::string_view argument) {
    const auto separator = argument.find('=');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 == argument.size()) {
        throw std::invalid_argument(
            "Engine arguments must use name=/absolute/path.engine");
    }
    return {
        std::string(argument.substr(0, separator)),
        std::filesystem::path(argument.substr(separator + 1)),
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " name=/path/to/model.engine [...]\n";
        return 2;
    }

    try {
        std::vector<std::unique_ptr<streamglass::TensorRtEngine>> engines;
        engines.reserve(static_cast<std::size_t>(argc - 1));
        for (int index = 1; index < argc; ++index) {
            auto [name, path] = parse_engine_argument(argv[index]);
            std::cout << "[load] " << name << " path=" << path << std::endl;
            engines.push_back(std::make_unique<streamglass::TensorRtEngine>(
                std::move(name), path));
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const auto cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemGetInfo failed: ") +
                cudaGetErrorString(cuda_status));
        }

        for (const auto& engine : engines) {
            std::cout << "[engine] " << engine->name() << '\n';
            for (const auto& tensor : engine->tensors()) {
                std::cout << "  " << streamglass::io_mode_to_string(tensor.mode)
                          << " name=" << tensor.name
                          << " dtype="
                          << streamglass::data_type_to_string(tensor.dtype)
                          << " location="
                          << streamglass::location_to_string(tensor.location)
                          << " shape="
                          << streamglass::dims_to_string(tensor.engine_shape)
                          << '\n';
                for (std::size_t profile = 0; profile < tensor.profiles.size();
                     ++profile) {
                    const auto& value = tensor.profiles[profile];
                    std::cout << "    profile=" << profile
                              << " min="
                              << streamglass::dims_to_string(value.minimum)
                              << " opt="
                              << streamglass::dims_to_string(value.optimum)
                              << " max="
                              << streamglass::dims_to_string(value.maximum)
                              << '\n';
                }
            }
        }
        std::cout << "[cuda-memory] free=" << free_bytes
                  << " total=" << total_bytes << '\n';
        std::cout << "[ok] loaded_engines=" << engines.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[error] " << error.what() << '\n';
        return 1;
    }
}
