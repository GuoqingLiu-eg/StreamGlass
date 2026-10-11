#include "streamglass/streaming_text_policy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace streamglass {

StreamingTextPolicy::StreamingTextPolicy(
    DecoderRuntime& decoder,
    const TokenizerAssets& tokenizer,
    const std::string_view target_language)
    : decoder_(decoder),
      tokenizer_(tokenizer),
      language_(&tokenizer.language(target_language)) {
    target_indices_.reserve(kMaximumContentTokens);
    policy_prefix_.reserve(DecoderRuntime::kMaximumTargetLength);
    predicted_indices_.reserve(kMaximumConsecutiveWrites);
    t2u_token_indices_.reserve(DecoderRuntime::kMaximumTargetLength);
    calls_.reserve(kMaximumConsecutiveWrites + 2U);
}

void StreamingTextPolicy::set_target_language(
    const std::string_view target_language) {
    const auto* next_language = &tokenizer_.language(target_language);
    reset();
    language_ = next_language;
}

void StreamingTextPolicy::record_call(const DecoderResult& result) {
    calls_.push_back(DecoderCallDecision{
        .selected_index = result.selected_index,
        .probability = result.decision_probability,
    });
}

void StreamingTextPolicy::reset() noexcept {
    target_indices_.clear();
    policy_prefix_.clear();
    predicted_indices_.clear();
    t2u_token_indices_.clear();
    calls_.clear();
    consecutive_probe_reads_ = 0;
    decoder_.invalidate_canonical();
}

TextPolicyResult StreamingTextPolicy::run(
    const __half* encoder_output,
    const std::size_t encoder_frames,
    const bool source_finished) {
    policy_prefix_.clear();
    policy_prefix_.push_back(language_->prefix_eos);
    policy_prefix_.push_back(language_->prefix_language);
    policy_prefix_.insert(
        policy_prefix_.end(), target_indices_.begin(), target_indices_.end());
    predicted_indices_.clear();
    calls_.clear();

    // Hybrid cached mode: probe the READ/WRITE decision with one step over the
    // persistent canonical KV. READ policies (the majority) skip the full
    // prefill; WRITE policies fall through to a reference-exact prefill so T2U
    // consumes fresh prefix features. source_finished always uses prefill.
    bool have_result = false;
    DecoderResult decoder_result{};
    if (cached_mode_ && !source_finished &&
        consecutive_probe_reads_ < kMaximumConsecutiveProbeReads &&
        decoder_.canonical_matches(policy_prefix_.size(), policy_prefix_.back())) {
        const auto probe = decoder_.resume_policy(encoder_output, encoder_frames);
        static const bool compare_probe =
            std::getenv("STREAMGLASS_DECODER_CACHED_COMPARE") != nullptr;
        if (compare_probe) {
            decoder_result = decoder_.begin_policy(
                policy_prefix_, encoder_output, encoder_frames);
            have_result = true;
            std::fprintf(
                stderr,
                "[probe-cmp] prefix=%zu probe_p=%.4f probe_tok=%lld "
                "fresh_p=%.4f fresh_tok=%lld\n",
                policy_prefix_.size(),
                static_cast<double>(probe.decision_probability),
                static_cast<long long>(probe.selected_index),
                static_cast<double>(decoder_result.decision_probability),
                static_cast<long long>(decoder_result.selected_index));
        } else if (
            probe.decision_probability < kProbeReadThreshold &&
            probe.selected_index != tokenizer_.eos_index()) {
            ++consecutive_probe_reads_;
            record_call(probe);
            return TextPolicyResult{
                .action = TextPolicyAction::read,
                .finished = false,
                .emitted_tokens = predicted_indices_,
                .committed_tokens = target_indices_,
                .t2u_token_indices = {},
                .t2u_features_device = nullptr,
                .t2u_feature_count = 0,
                .decoder_calls = calls_,
            };
        }
    }

    if (!have_result) {
        decoder_result = decoder_.begin_policy(
            policy_prefix_, encoder_output, encoder_frames);
    }
    consecutive_probe_reads_ = 0;
    if (cached_mode_ &&
        decoder_.cache_length() >= DecoderRuntime::kMinimumTargetLength + 1U) {
        // Seed the canonical cache from the prefill result before any step can
        // pollute the working buffers with a later-discarded suffix.
        decoder_.capture_canonical(policy_prefix_.back());
    }
    bool finished = false;

    while (true) {
        record_call(decoder_result);
        const auto index = decoder_result.selected_index;
        const auto probability = decoder_result.decision_probability;

        // Official streaming evaluate uses no_early_stop=True. It discards the
        // tentative policy suffix when EOS is selected with probability 1.
        if (!source_finished &&
            (probability < kDecisionThreshold ||
             index == tokenizer_.eos_index())) {
            if (probability == 1.0F) {
                predicted_indices_.clear();
            }
            break;
        }

        if (index == tokenizer_.eos_index() ||
            target_indices_.size() + predicted_indices_.size() >
                kMaximumContentTokens) {
            finished = true;
            break;
        }

        if (probability < kDecisionThreshold && !source_finished) {
            break;
        }

        if (target_indices_.size() + predicted_indices_.size() >=
                kMaximumContentTokens ||
            predicted_indices_.size() >= kMaximumConsecutiveWrites) {
            break;
        }

        predicted_indices_.push_back(index);
        decoder_result = decoder_.step(index);
    }

    target_indices_.insert(
        target_indices_.end(),
        predicted_indices_.begin(),
        predicted_indices_.end());

    if (cached_mode_ && !predicted_indices_.empty()) {
        // Canonical = committed prefix minus its last token, captured before
        // the temporary comma so the comma stays on scratch state only.
        decoder_.capture_canonical(predicted_indices_.back());
    }

    if (predicted_indices_.empty() && !finished) {
        return TextPolicyResult{
            .action = TextPolicyAction::read,
            .finished = false,
            .emitted_tokens = predicted_indices_,
            .committed_tokens = target_indices_,
            .t2u_token_indices = {},
            .t2u_features_device = nullptr,
            .t2u_feature_count = 0,
            .decoder_calls = calls_,
        };
    }

    // Preserve the official post-update max-length expression, including its
    // second accounting of the just-emitted suffix.
    finished = finished ||
               target_indices_.size() + predicted_indices_.size() >
                   kMaximumContentTokens;

    t2u_token_indices_.clear();
    t2u_token_indices_.push_back(language_->prefix_eos);
    t2u_token_indices_.push_back(language_->prefix_language);
    t2u_token_indices_.insert(
        t2u_token_indices_.end(),
        target_indices_.begin(),
        target_indices_.end());

    if (!predicted_indices_.empty() &&
        predicted_indices_.back() != tokenizer_.eos_index()) {
        t2u_token_indices_.push_back(tokenizer_.comma_index());
        record_call(decoder_.step(tokenizer_.comma_index()));
    }

    if (decoder_.decoder_feature_count() != t2u_token_indices_.size()) {
        throw std::runtime_error(
            "Official decoder features and T2U token sequence diverged");
    }

    return TextPolicyResult{
        .action = TextPolicyAction::write,
        .finished = finished,
        .emitted_tokens = predicted_indices_,
        .committed_tokens = target_indices_,
        .t2u_token_indices = t2u_token_indices_,
        .t2u_features_device = decoder_.decoder_features_device(),
        .t2u_feature_count = decoder_.decoder_feature_count(),
        .decoder_calls = calls_,
    };
}

}  // namespace streamglass
