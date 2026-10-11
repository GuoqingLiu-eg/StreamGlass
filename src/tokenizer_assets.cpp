#include "streamglass/tokenizer_assets.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace streamglass {
namespace {

std::vector<std::string_view> split(
    const std::string_view value, const char delimiter) {
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        if (end == std::string_view::npos) {
            fields.push_back(value.substr(start));
            break;
        }
        fields.push_back(value.substr(start, end - start));
        start = end + 1;
    }
    return fields;
}

template <typename Integer>
Integer parse_integer(const std::string_view value, const std::string_view field) {
    Integer parsed{};
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error(
            "Invalid integer for " + std::string(field) + ": " +
            std::string(value));
    }
    return parsed;
}

std::uint8_t hex_nibble(const char value) {
    if (value >= '0' && value <= '9') {
        return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<std::uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
        return static_cast<std::uint8_t>(value - 'A' + 10);
    }
    throw std::runtime_error("Invalid hexadecimal tokenizer byte");
}

std::string decode_hex(const std::string_view value) {
    if (value.size() % 2 != 0) {
        throw std::runtime_error("Tokenizer hex field has an odd length");
    }
    std::string output(value.size() / 2, '\0');
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto high = hex_nibble(value[index * 2]);
        const auto low = hex_nibble(value[index * 2 + 1]);
        output[index] = static_cast<char>((high << 4U) | low);
    }
    return output;
}

std::unordered_map<std::string, std::string> load_metadata(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open tokenizer metadata: " + path.string());
    }
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0) {
            throw std::runtime_error("Malformed tokenizer metadata line: " + line);
        }
        values.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    return values;
}

const std::string& require_metadata(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::string& name) {
    const auto iterator = metadata.find(name);
    if (iterator == metadata.end()) {
        throw std::runtime_error("Missing tokenizer metadata field: " + name);
    }
    return iterator->second;
}

}  // namespace

TokenizerAssets::TokenizerAssets(const std::filesystem::path& directory) {
    const auto metadata = load_metadata(directory / "runtime.meta");
    if (require_metadata(metadata, "schema_version") != "1") {
        throw std::runtime_error("Unsupported tokenizer runtime asset schema");
    }

    const auto vocab_size = parse_integer<std::size_t>(
        require_metadata(metadata, "text_vocab_size"), "text_vocab_size");
    pad_index_ = parse_integer<std::int64_t>(
        require_metadata(metadata, "text_pad_idx"), "text_pad_idx");
    unknown_index_ = parse_integer<std::int64_t>(
        require_metadata(metadata, "text_unk_idx"), "text_unk_idx");
    eos_index_ = parse_integer<std::int64_t>(
        require_metadata(metadata, "text_eos_idx"), "text_eos_idx");
    comma_index_ = parse_integer<std::int64_t>(
        require_metadata(metadata, "comma_idx"), "comma_idx");

    tokens_.resize(vocab_size);
    std::vector<bool> populated(vocab_size, false);
    std::ifstream input(directory / "text_tokens.tsv");
    if (!input) {
        throw std::runtime_error(
            "Cannot open text token table under " + directory.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split(line, '\t');
        if (fields.size() != 7) {
            throw std::runtime_error("Malformed text token row");
        }
        const auto index = parse_integer<std::size_t>(fields[0], "token index");
        if (index >= tokens_.size() || populated[index]) {
            throw std::runtime_error("Duplicate or out-of-range text token index");
        }
        auto& record = tokens_[index];
        record.piece = decode_hex(fields[1]);
        record.codepoint_count =
            parse_integer<std::uint32_t>(fields[2], "codepoint_count");
        record.starts_space = fields[3] == "1";
        record.punctuation = fields[4] == "1";
        record.character_offset =
            static_cast<std::uint32_t>(token_characters_.size());
        if (!fields[5].empty()) {
            for (const auto character : split(fields[5], ',')) {
                token_characters_.push_back(
                    parse_integer<std::int64_t>(character, "character index"));
            }
        }
        record.character_size = parse_integer<std::uint32_t>(
            fields[6], "character_size");
        if (record.character_size !=
            token_characters_.size() - record.character_offset) {
            throw std::runtime_error("Character table size does not match token row");
        }
        populated[index] = true;
    }
    if (std::find(populated.begin(), populated.end(), false) != populated.end()) {
        throw std::runtime_error("Text token table is incomplete");
    }

    const auto language_count = parse_integer<std::size_t>(
        require_metadata(metadata, "language_count"), "language_count");
    for (std::size_t index = 0; index < language_count; ++index) {
        const auto field = "language." + std::to_string(index);
        const auto values = split(require_metadata(metadata, field), ',');
        if (values.size() != 5) {
            throw std::runtime_error("Malformed language conditioning row: " + field);
        }
        LanguageConditioning language{
            .language = std::string(values[0]),
            .prefix_eos = parse_integer<std::int64_t>(values[1], field),
            .prefix_language = parse_integer<std::int64_t>(values[2], field),
            .vocoder_language = parse_integer<std::int64_t>(values[3], field),
            .vocoder_speaker = parse_integer<std::int64_t>(values[4], field),
        };
        languages_.emplace(language.language, std::move(language));
    }
}

const LanguageConditioning& TokenizerAssets::language(
    const std::string_view language_name) const {
    const auto iterator = languages_.find(std::string(language_name));
    if (iterator == languages_.end()) {
        throw std::out_of_range(
            "No speech output conditioning for language " +
            std::string(language_name));
    }
    return iterator->second;
}

T2uTextInputSize TokenizerAssets::build_t2u_inputs(
    const std::span<const std::int64_t> token_indices,
    const std::span<std::int64_t> character_indices,
    const std::span<std::int64_t> character_lengths) const {
    if (token_indices.size() < 2) {
        throw std::invalid_argument("T2U token input must include the two-token prefix");
    }
    if (character_lengths.size() < token_indices.size()) {
        throw std::length_error("T2U character length output is too small");
    }
    std::fill_n(character_lengths.begin(), token_indices.size(), 0);

    std::size_t processed_count = 0;
    for (std::size_t index = 2; index < token_indices.size(); ++index) {
        const auto normalized =
            token_indices[index] == eos_index_ ? pad_index_ : token_indices[index];
        if (normalized == pad_index_) {
            continue;
        }
        ++processed_count;
    }

    std::size_t character_count = 0;
    for (std::size_t index = 0; index < processed_count; ++index) {
        const auto token_index = token_indices[index + 2];
        const auto& current = token(token_index);
        const auto* previous = index == 0 ? nullptr : &token(token_indices[index + 1]);
        const auto* next =
            index + 1 == processed_count ? nullptr : &token(token_indices[index + 3]);

        std::int64_t length = token_index == unknown_index_
                                  ? 1
                                  : static_cast<std::int64_t>(
                                        current.codepoint_count);
        if (current.punctuation && next != nullptr && next->starts_space) {
            ++length;
        } else if (
            previous != nullptr && previous->punctuation && current.starts_space) {
            --length;
        }
        if (length < 0) {
            throw std::runtime_error("T2U character length became negative");
        }
        // Official TagManager adds one zero on each side after removing the
        // two-token NLLB target prefix, so the first content duration is at 1.
        character_lengths[index + 1] = length;

        if (character_count + current.character_size > character_indices.size()) {
            throw std::length_error("T2U character sequence output is too small");
        }
        const auto begin = token_characters_.begin() + current.character_offset;
        std::copy_n(
            begin,
            current.character_size,
            character_indices.begin() +
                static_cast<std::ptrdiff_t>(character_count));
        character_count += current.character_size;
    }
    return T2uTextInputSize{
        .character_count = character_count,
        .token_count = token_indices.size(),
    };
}

std::string TokenizerAssets::detokenize(
    const std::span<const std::int64_t> token_indices) const {
    constexpr std::string_view sentencepiece_space = "\xE2\x96\x81";
    std::string output;
    for (const auto index : token_indices) {
        if (index == pad_index_ || index == eos_index_) {
            continue;
        }
        const auto& piece = token(index).piece;
        if ((piece.starts_with('<') && piece.ends_with('>')) ||
            (piece.starts_with("__") && piece.ends_with("__"))) {
            continue;
        }
        std::size_t cursor = 0;
        while (cursor < piece.size()) {
            const auto found = piece.find(sentencepiece_space, cursor);
            if (found == std::string::npos) {
                output.append(piece, cursor, std::string::npos);
                break;
            }
            output.append(piece, cursor, found - cursor);
            output.push_back(' ');
            cursor = found + sentencepiece_space.size();
        }
    }
    if (!output.empty() && output.front() == ' ') {
        output.erase(output.begin());
    }
    return output;
}

const TokenizerAssets::TokenRecord& TokenizerAssets::token(
    const std::int64_t index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= tokens_.size()) {
        throw std::out_of_range("Text token index is outside the exported vocabulary");
    }
    return tokens_[static_cast<std::size_t>(index)];
}

}  // namespace streamglass

