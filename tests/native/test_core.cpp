#include "streamglass/buffer_pool.hpp"
#include "streamglass/hfp_button_packet.hpp"
#include "streamglass/pause_control.hpp"
#include "streamglass/spsc_ring.hpp"
#include "streamglass/tokenizer_assets.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

struct AudioBlock {
    std::uint64_t utterance_id = 0;
    std::uint32_t samples = 0;
    std::array<float, 16> pcm{};
};

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void test_ring_capacity_and_order() {
    streamglass::SpscRing<std::uint32_t, 3> ring;
    require(ring.empty(), "new ring must be empty");
    require(ring.try_push(10), "push 10");
    require(ring.try_push(20), "push 20");
    require(ring.try_push(30), "push 30");
    require(ring.full(), "ring must be full");
    require(!ring.try_push(40), "full ring accepted a value");

    std::uint32_t value = 0;
    require(ring.try_pop(value) && value == 10, "FIFO item 10");
    require(ring.try_pop(value) && value == 20, "FIFO item 20");
    require(ring.try_push(40), "wraparound push");
    require(ring.try_pop(value) && value == 30, "FIFO item 30");
    require(ring.try_pop(value) && value == 40, "FIFO item 40");
    require(!ring.try_pop(value), "empty ring returned a value");
}

void test_pool_rejects_stale_handle() {
    streamglass::SpscBufferPool<AudioBlock, 2> pool;
    const auto first = pool.try_acquire();
    const auto second = pool.try_acquire();
    require(first.has_value(), "first pool acquire");
    require(second.has_value(), "second pool acquire");
    require(!pool.try_acquire().has_value(), "exhausted pool acquire");

    auto* payload = pool.get(*first);
    require(payload != nullptr, "live handle lookup");
    payload->utterance_id = 17;
    payload->samples = 16;
    require(pool.release(*first), "first release");
    require(pool.get(*first) == nullptr, "stale handle lookup");
    require(!pool.release(*first), "stale handle release");

    const auto recycled = pool.try_acquire();
    require(recycled.has_value(), "recycled acquire");
    require(recycled->slot == first->slot, "slot was not recycled");
    require(
        recycled->generation != first->generation,
        "generation did not advance");
    require(pool.get(*recycled) != nullptr, "recycled handle lookup");
    require(pool.release(*second), "second release");
    require(pool.release(*recycled), "recycled release");
    require(pool.available() == 2, "pool availability after release");
}

void test_pause_control_is_edge_triggered() {
    streamglass::PauseControl control;
    require(!control.paused(), "pause control must start resumed");
    require(control.revision() == 0, "initial pause revision");
    require(control.set_paused(true), "first pause transition");
    require(control.paused(), "pause transition did not stick");
    require(control.revision() == 1, "pause revision did not advance");
    require(!control.set_paused(true), "duplicate pause was accepted");
    require(control.revision() == 1, "duplicate pause advanced revision");
    require(!control.toggle(), "toggle must return resumed state");
    require(!control.paused(), "toggle did not resume");
    require(control.revision() == 2, "toggle revision did not advance");
    require(!control.set_paused(false), "duplicate resume was accepted");
}

void test_hfp_button_packet_parser() {
    constexpr std::array<std::uint8_t, 27> packet{
        0x05, 0x00,  // HCI monitor ACL RX opcode
        0x01, 0x00,  // hci1
        0x15, 0x00,  // 21-byte monitor payload
        0x80, 0x20,  // ACL handle 128, PB flag 2
        0x11, 0x00,  // 17-byte ACL payload
        0x0d, 0x00,  // 13-byte L2CAP payload
        0x41, 0x00,  // dynamic RFCOMM channel
        0x11, 0xef, 0x10, 0x00,
        'A', 'T', '+', 'C', 'H', 'U', 'P', '\r',
        0xbf,
    };
    const auto parsed = streamglass::parse_hfp_button_packet(packet);
    require(parsed.has_value(), "HFP button packet was not recognized");
    require(parsed->controller_index == 1, "HFP controller index");
    require(parsed->connection_handle == 128, "HFP connection handle");

    auto wrong_opcode = packet;
    wrong_opcode[0] = 0x04;
    require(
        !streamglass::parse_hfp_button_packet(wrong_opcode).has_value(),
        "non-RX HCI monitor packet was accepted");

    auto truncated = packet;
    truncated[4] = 0xff;
    truncated[5] = 0x00;
    require(
        !streamglass::parse_hfp_button_packet(truncated).has_value(),
        "truncated HCI monitor packet was accepted");

    auto incomplete_command = packet;
    incomplete_command[25] = '\0';
    require(
        !streamglass::parse_hfp_button_packet(
             incomplete_command).has_value(),
        "incomplete HFP command was accepted");
}

void test_tokenizer_table_reproduces_punctuation_lengths() {
    const auto directory =
        std::filesystem::temp_directory_path() / "streamglass-tokenizer-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream metadata(directory / "runtime.meta");
        metadata << "schema_version=1\n"
                 << "text_vocab_size=7\n"
                 << "text_pad_idx=0\n"
                 << "text_unk_idx=1\n"
                 << "text_eos_idx=3\n"
                 << "comma_idx=5\n"
                 << "language_count=1\n"
                 << "language.0=eng,3,4,7,9\n";
    }
    {
        std::ofstream tokens(directory / "text_tokens.tsv");
        tokens << "0\t3c7061643e\t5\t0\t0\t10\t1\n"
               << "1\t3c756e6b3e\t1\t0\t0\t11\t1\n"
               << "2\t3c733e\t3\t0\t0\t12\t1\n"
               << "3\t3c2f733e\t4\t0\t0\t13\t1\n"
               << "4\t5f5f656e675f5f\t7\t0\t0\t14\t1\n"
               << "5\t2c\t1\t0\t1\t20\t1\n"
               << "6\te2968178\t2\t1\t0\t21,22\t2\n";
    }

    const streamglass::TokenizerAssets assets(directory);
    const std::array<std::int64_t, 5> text{3, 4, 5, 6, 3};
    std::array<std::int64_t, 8> characters{};
    std::array<std::int64_t, 5> lengths{};
    const auto size = assets.build_t2u_inputs(text, characters, lengths);
    require(size.character_count == 3, "T2U character count");
    require(size.token_count == 5, "T2U token count");
    require(characters[0] == 20, "comma character id");
    require(characters[1] == 21 && characters[2] == 22, "word char ids");
    require(lengths == std::array<std::int64_t, 5>{0, 2, 1, 0, 0},
            "official punctuation/space character lengths");
    require(assets.language("eng").vocoder_language == 7,
            "language conditioning");
    require(assets.detokenize(std::span{text}.subspan(2, 2)) == ", x",
            "SentencePiece display detokenization");
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    test_ring_capacity_and_order();
    test_pool_rejects_stale_handle();
    test_pause_control_is_edge_triggered();
    test_hfp_button_packet_parser();
    test_tokenizer_table_reproduces_punctuation_lengths();
    std::cout << "streamglass core tests passed\n";
    return 0;
}
