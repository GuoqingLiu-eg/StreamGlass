#include "streamglass/tokenizer_assets.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

std::int64_t parse_index(const std::string& value) {
    std::int64_t index = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), index);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("Invalid token index: " + value);
    }
    return index;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: streamglass-tokenizer-inspect "
                         "<asset-directory> <target-language> [token-id ...]\n";
            return 2;
        }

        const streamglass::TokenizerAssets assets{
            std::filesystem::path(argv[1])};
        const auto& language = assets.language(argv[2]);
        std::cout << "vocabulary_size=" << assets.vocabulary_size() << '\n'
                  << "pad_index=" << assets.pad_index() << '\n'
                  << "unknown_index=" << assets.unknown_index() << '\n'
                  << "eos_index=" << assets.eos_index() << '\n'
                  << "comma_index=" << assets.comma_index() << '\n'
                  << "target_language=" << language.language << '\n'
                  << "prefix_indices=" << language.prefix_eos << ','
                  << language.prefix_language << '\n'
                  << "vocoder_conditioning=" << language.vocoder_language << ','
                  << language.vocoder_speaker << '\n';

        std::vector<std::int64_t> tokens;
        tokens.reserve(static_cast<std::size_t>(argc - 3));
        for (int index = 3; index < argc; ++index) {
            tokens.push_back(parse_index(argv[index]));
        }
        if (!tokens.empty()) {
            std::cout << "text=" << assets.detokenize(tokens) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tokenizer inspection failed: " << error.what() << '\n';
        return 1;
    }
}
