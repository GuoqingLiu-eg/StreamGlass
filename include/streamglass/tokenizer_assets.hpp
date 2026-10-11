#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace streamglass {

struct LanguageConditioning {
    std::string language;
    std::int64_t prefix_eos = 0;
    std::int64_t prefix_language = 0;
    std::int64_t vocoder_language = 0;
    std::int64_t vocoder_speaker = 0;
};

struct T2uTextInputSize {
    std::size_t character_count = 0;
    std::size_t token_count = 0;
};

class TokenizerAssets final {
public:
    explicit TokenizerAssets(const std::filesystem::path& directory);

    TokenizerAssets(const TokenizerAssets&) = delete;
    TokenizerAssets& operator=(const TokenizerAssets&) = delete;

    [[nodiscard]] std::int64_t pad_index() const noexcept { return pad_index_; }
    [[nodiscard]] std::int64_t unknown_index() const noexcept {
        return unknown_index_;
    }
    [[nodiscard]] std::int64_t eos_index() const noexcept { return eos_index_; }
    [[nodiscard]] std::int64_t comma_index() const noexcept {
        return comma_index_;
    }
    [[nodiscard]] std::size_t vocabulary_size() const noexcept {
        return tokens_.size();
    }

    [[nodiscard]] const LanguageConditioning& language(
        std::string_view language) const;

    [[nodiscard]] T2uTextInputSize build_t2u_inputs(
        std::span<const std::int64_t> token_indices,
        std::span<std::int64_t> character_indices,
        std::span<std::int64_t> character_lengths) const;

    [[nodiscard]] std::string detokenize(
        std::span<const std::int64_t> token_indices) const;

private:
    struct TokenRecord {
        std::string piece;
        std::uint32_t character_offset = 0;
        std::uint32_t character_size = 0;
        std::uint32_t codepoint_count = 0;
        bool starts_space = false;
        bool punctuation = false;
    };

    [[nodiscard]] const TokenRecord& token(std::int64_t index) const;

    std::int64_t pad_index_ = -1;
    std::int64_t unknown_index_ = -1;
    std::int64_t eos_index_ = -1;
    std::int64_t comma_index_ = -1;
    std::vector<TokenRecord> tokens_;
    std::vector<std::int64_t> token_characters_;
    std::unordered_map<std::string, LanguageConditioning> languages_;
};

}  // namespace streamglass

