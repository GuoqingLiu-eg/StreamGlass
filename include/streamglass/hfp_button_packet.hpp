#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace streamglass {

struct HfpButtonPacket {
    std::uint16_t controller_index = 0;
    std::uint16_t connection_handle = 0;
};

// HCI monitor packets use a six-byte little-endian header. For an ACL RX
// packet, the following four bytes are the HCI ACL header. The RFCOMM command
// can appear after L2CAP/RFCOMM framing, so matching the complete command is
// sufficient without depending on dynamic channel identifiers.
inline std::optional<HfpButtonPacket> parse_hfp_button_packet(
    const std::span<const std::uint8_t> packet) noexcept {
    constexpr std::size_t kMonitorHeaderSize = 6;
    constexpr std::size_t kAclHeaderSize = 4;
    constexpr std::uint16_t kAclRxOpcode = 5;
    constexpr std::uint16_t kConnectionHandleMask = 0x0fff;
    constexpr std::array<std::uint8_t, 8> kCommand{
        'A', 'T', '+', 'C', 'H', 'U', 'P', '\r'};

    const auto read_le16 = [](const std::uint8_t* bytes) {
        return static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(bytes[0]) |
            (static_cast<std::uint16_t>(bytes[1]) << 8U));
    };

    if (packet.size() < kMonitorHeaderSize + kAclHeaderSize) {
        return std::nullopt;
    }
    const auto opcode = read_le16(packet.data());
    const auto controller_index = read_le16(packet.data() + 2);
    const auto payload_size =
        static_cast<std::size_t>(read_le16(packet.data() + 4));
    if (opcode != kAclRxOpcode ||
        payload_size < kAclHeaderSize ||
        payload_size > packet.size() - kMonitorHeaderSize) {
        return std::nullopt;
    }

    const auto payload = packet.subspan(kMonitorHeaderSize, payload_size);
    const auto command_area = payload.subspan(kAclHeaderSize);
    if (std::search(
            command_area.begin(), command_area.end(),
            kCommand.begin(), kCommand.end()) == command_area.end()) {
        return std::nullopt;
    }

    return HfpButtonPacket{
        .controller_index = controller_index,
        .connection_handle = static_cast<std::uint16_t>(
            read_le16(payload.data()) & kConnectionHandleMask),
    };
}

}  // namespace streamglass
