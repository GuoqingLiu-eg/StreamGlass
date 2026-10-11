#pragma once

#include "streamglass/decoder_runtime.hpp"
#include "streamglass/tokenizer_assets.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace streamglass {

enum class TextPolicyAction : std::int32_t {
    read = 0,
    write = 1,
};

struct DecoderCallDecision {
    std::int64_t selected_index = 0;
    float probability = 0.0F;
};

struct TextPolicyResult {
    TextPolicyAction action = TextPolicyAction::read;
    bool finished = false;
    std::span<const std::int64_t> emitted_tokens;
    std::span<const std::int64_t> committed_tokens;
    std::span<const std::int64_t> t2u_token_indices;
    const __half* t2u_features_device = nullptr;
    std::size_t t2u_feature_count = 0;
    std::span<const DecoderCallDecision> decoder_calls;
};

class StreamingTextPolicy final {
public:
    static constexpr float kDecisionThreshold = 0.5F;
    // Stale-KV probes under-estimate p_choose (observed gap up to ~0.42), so a
    // probe may only decide READ below this conservative margin; anything
    // higher is escalated to a reference-exact prefill decision.
    static constexpr float kProbeReadThreshold = 0.15F;
    // Force a prefill verification after this many consecutive probe-only
    // READs so a biased probe can never starve WRITE indefinitely.
    static constexpr std::size_t kMaximumConsecutiveProbeReads = 4;
    static constexpr std::size_t kMaximumConsecutiveWrites = 50;
    static constexpr std::size_t kMaximumContentTokens = 100;

    StreamingTextPolicy(
        DecoderRuntime& decoder,
        const TokenizerAssets& tokenizer,
        std::string_view target_language);

    StreamingTextPolicy(const StreamingTextPolicy&) = delete;
    StreamingTextPolicy& operator=(const StreamingTextPolicy&) = delete;

    [[nodiscard]] TextPolicyResult run(
        const __half* encoder_output,
        std::size_t encoder_frames,
        bool source_finished);

    void reset() noexcept;
    void set_target_language(std::string_view target_language);

    // Cached mode keeps decoder self-attention K/V and prefix features across
    // 320 ms policy calls (semantic deviation; must pass reference/cached A/B
    // before production use). Reference mode is the default.
    void set_cached_mode(const bool enabled) noexcept {
        cached_mode_ = enabled;
    }
    [[nodiscard]] bool cached_mode() const noexcept { return cached_mode_; }

    [[nodiscard]] std::span<const std::int64_t> committed_tokens()
        const noexcept {
        return target_indices_;
    }

private:
    void record_call(const DecoderResult& result);

    DecoderRuntime& decoder_;
    const TokenizerAssets& tokenizer_;
    const LanguageConditioning* language_ = nullptr;
    bool cached_mode_ = false;
    std::size_t consecutive_probe_reads_ = 0;
    std::vector<std::int64_t> target_indices_;
    std::vector<std::int64_t> policy_prefix_;
    std::vector<std::int64_t> predicted_indices_;
    std::vector<std::int64_t> t2u_token_indices_;
    std::vector<DecoderCallDecision> calls_;
};

}  // namespace streamglass
