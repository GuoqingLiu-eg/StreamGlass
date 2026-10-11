#include "streamglass/decoder_runtime.hpp"

#include "streamglass/cuda_kernels.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace streamglass {
namespace {

nvinfer1::Dims make_dims(
    const std::initializer_list<std::int64_t> dimensions) {
    nvinfer1::Dims result{};
    result.nbDims = static_cast<std::int32_t>(dimensions.size());
    std::size_t axis = 0;
    for (const auto value : dimensions) {
        result.d[axis] = value;
        ++axis;
    }
    return result;
}

void set_shape(
    nvinfer1::IExecutionContext& context,
    const char* name,
    const nvinfer1::Dims& shape) {
    if (!context.setInputShape(name, shape)) {
        throw std::runtime_error(
            "TensorRT rejected input shape for " + std::string(name) +
            ": " + dims_to_string(shape));
    }
}

void bind(
    nvinfer1::IExecutionContext& context, const char* name, void* pointer) {
    if (!context.setTensorAddress(name, pointer)) {
        throw std::runtime_error(
            "TensorRT rejected tensor address for " + std::string(name));
    }
}

void require_output_shape(
    nvinfer1::IExecutionContext& context,
    const char* name,
    const nvinfer1::Dims& expected) {
    const auto actual = context.getTensorShape(name);
    auto equal = actual.nbDims == expected.nbDims;
    for (std::int32_t axis = 0; equal && axis < actual.nbDims; ++axis) {
        equal = actual.d[axis] == expected.d[axis];
    }
    if (!equal) {
        throw std::runtime_error(
            "Unexpected TensorRT output shape for " + std::string(name) +
            ": expected " + dims_to_string(expected) + ", got " +
            dims_to_string(actual));
    }
}

}  // namespace

DecoderRuntime::DecoderRuntime(
    const std::filesystem::path& prefill_engine_path,
    const std::filesystem::path& step_engine_path)
    : prefill_engine_("decoder_prefill", prefill_engine_path),
      step_engine_("decoder_step", step_engine_path),
      stream_(),
      target_input_(kMaximumTargetLength),
      position_(1),
      selected_index_(1),
      decision_probability_(1),
      causal_mask_(kMaximumTargetLength * kMaximumTargetLength),
      decoder_features_(kMaximumTargetLength * kDecoderDimension),
      logits_(kVocabularySize),
      decision_values_(kDecisionCount),
      argmax_scratch_values_(argmax_half_scratch_count(kVocabularySize)),
      argmax_scratch_indices_(argmax_half_scratch_count(kVocabularySize)) {
    key_cache_[0].allocate(kMaximumCacheElements);
    key_cache_[1].allocate(kMaximumCacheElements);
    value_cache_[0].allocate(kMaximumCacheElements);
    value_cache_[1].allocate(kMaximumCacheElements);
    canonical_key_.allocate(kMaximumCacheElements);
    canonical_value_.allocate(kMaximumCacheElements);
    validate_engines();
}

void DecoderRuntime::validate_engines() {
    const auto& prefill = prefill_engine_.engine();
    const auto& step = step_engine_.engine();
    if (prefill.getNbIOTensors() != 8 || step.getNbIOTensors() != 10 ||
        prefill.getTensorDataType("target_input") !=
            nvinfer1::DataType::kINT64 ||
        prefill.getTensorDataType("encoder_output") !=
            nvinfer1::DataType::kHALF ||
        prefill.getTensorDataType("causal_mask") !=
            nvinfer1::DataType::kHALF ||
        prefill.getTensorDataType("decoder_output") !=
            nvinfer1::DataType::kHALF ||
        prefill.getTensorDataType("logits") != nvinfer1::DataType::kHALF ||
        prefill.getTensorDataType("decision_values") !=
            nvinfer1::DataType::kFLOAT ||
        step.getTensorDataType("target_input") !=
            nvinfer1::DataType::kINT64 ||
        step.getTensorDataType("encoder_output") !=
            nvinfer1::DataType::kHALF ||
        step.getTensorDataType("position") != nvinfer1::DataType::kINT64 ||
        step.getTensorDataType("self_k_cache") !=
            nvinfer1::DataType::kHALF ||
        step.getTensorDataType("self_v_cache") !=
            nvinfer1::DataType::kHALF ||
        step.getTensorDataType("decision_values") !=
            nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("Decoder TensorRT I/O contract changed");
    }
}

DecoderResult DecoderRuntime::begin_policy(
    const std::span<const std::int64_t> target_prefix,
    const __half* encoder_output,
    const std::size_t encoder_frames) {
    if (target_prefix.size() < kMinimumTargetLength ||
        target_prefix.size() > kMaximumTargetLength) {
        throw std::invalid_argument("Decoder prefill target length is outside profile");
    }
    if (encoder_output == nullptr || encoder_frames == 0 ||
        encoder_frames > kMaximumEncoderFrames) {
        throw std::invalid_argument("Decoder encoder input is outside profile");
    }

    std::copy(target_prefix.begin(), target_prefix.end(), target_input_.data());
    encoder_output_ = encoder_output;
    encoder_frames_ = encoder_frames;
    feature_count_ = 0;
    cache_length_ = 0;
    current_cache_index_ = 0;
    policy_active_ = false;

    launch_causal_mask(
        causal_mask_.data(), target_prefix.size(), stream_.get());

    auto& context = prefill_engine_.context();
    set_shape(
        context,
        "target_input",
        make_dims({1, static_cast<std::int64_t>(target_prefix.size())}));
    set_shape(
        context,
        "encoder_output",
        make_dims(
            {1,
             static_cast<std::int64_t>(encoder_frames),
             static_cast<std::int64_t>(kDecoderDimension)}));
    set_shape(
        context,
        "causal_mask",
        make_dims(
            {static_cast<std::int64_t>(target_prefix.size()),
             static_cast<std::int64_t>(target_prefix.size())}));

    bind(context, "target_input", target_input_.device_data());
    bind(
        context,
        "encoder_output",
        const_cast<__half*>(encoder_output_));
    bind(context, "causal_mask", causal_mask_.data());
    bind(context, "decoder_output", decoder_features_.data());
    bind(context, "logits", logits_.data());
    bind(context, "decision_values", decision_values_.data());
    bind(context, "self_k_cache_out", key_cache_[0].data());
    bind(context, "self_v_cache_out", value_cache_[0].data());

    const auto target_length = target_prefix.size();
    require_output_shape(
        context,
        "decoder_output",
        make_dims(
            {1,
             static_cast<std::int64_t>(target_length),
             static_cast<std::int64_t>(kDecoderDimension)}));
    require_output_shape(
        context,
        "self_k_cache_out",
        make_dims(
            {static_cast<std::int64_t>(kDecoderLayers),
             1,
             static_cast<std::int64_t>(kAttentionHeads),
             static_cast<std::int64_t>(target_length),
             static_cast<std::int64_t>(kHeadDimension)}));
    if (!context.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT decoder prefill enqueueV3 failed");
    }

    auto result = collect_result(0, target_length, target_length);
    feature_count_ = target_length;
    cache_length_ = target_length;
    policy_active_ = true;
    return result;
}

DecoderResult DecoderRuntime::step(const std::int64_t target_token) {
    if (!policy_active_) {
        throw std::logic_error("Decoder step requires an active policy");
    }
    if (cache_length_ < kMinimumTargetLength ||
        cache_length_ >= kMaximumTargetLength ||
        feature_count_ >= kMaximumTargetLength) {
        throw std::length_error("Decoder step would exceed the cache profile");
    }

    target_input_.data()[0] = target_token;
    position_.data()[0] = static_cast<std::int64_t>(cache_length_);
    const auto next_cache_index = 1U - current_cache_index_;

    auto& context = step_engine_.context();
    set_shape(context, "target_input", make_dims({1, 1}));
    set_shape(
        context,
        "encoder_output",
        make_dims(
            {1,
             static_cast<std::int64_t>(encoder_frames_),
             static_cast<std::int64_t>(kDecoderDimension)}));
    set_shape(context, "position", make_dims({1}));
    const auto cache_shape = make_dims(
        {static_cast<std::int64_t>(kDecoderLayers),
         1,
         static_cast<std::int64_t>(kAttentionHeads),
         static_cast<std::int64_t>(cache_length_),
         static_cast<std::int64_t>(kHeadDimension)});
    set_shape(context, "self_k_cache", cache_shape);
    set_shape(context, "self_v_cache", cache_shape);

    bind(context, "target_input", target_input_.device_data());
    bind(
        context,
        "encoder_output",
        const_cast<__half*>(encoder_output_));
    bind(context, "position", position_.device_data());
    bind(context, "self_k_cache", key_cache_[current_cache_index_].data());
    bind(context, "self_v_cache", value_cache_[current_cache_index_].data());
    bind(
        context,
        "decoder_output",
        decoder_features_.data() + feature_count_ * kDecoderDimension);
    bind(context, "logits", logits_.data());
    bind(context, "decision_values", decision_values_.data());
    bind(context, "self_k_cache_out", key_cache_[next_cache_index].data());
    bind(context, "self_v_cache_out", value_cache_[next_cache_index].data());

    const auto output_cache_length = cache_length_ + 1U;
    require_output_shape(
        context,
        "self_k_cache_out",
        make_dims(
            {static_cast<std::int64_t>(kDecoderLayers),
             1,
             static_cast<std::int64_t>(kAttentionHeads),
             static_cast<std::int64_t>(output_cache_length),
             static_cast<std::int64_t>(kHeadDimension)}));
    if (!context.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT decoder step enqueueV3 failed");
    }

    auto result = collect_result(feature_count_, 1, output_cache_length);
    feature_count_ += 1U;
    cache_length_ = output_cache_length;
    current_cache_index_ = next_cache_index;
    return result;
}

DecoderResult DecoderRuntime::resume_policy(
    const __half* encoder_output, const std::size_t encoder_frames) {
    if (!canonical_valid_ || canonical_length_ < kMinimumTargetLength - 1U ||
        canonical_length_ + 1U > kMaximumTargetLength) {
        throw std::logic_error("Decoder resume requires a valid canonical cache");
    }
    if (encoder_output == nullptr || encoder_frames == 0 ||
        encoder_frames > kMaximumEncoderFrames) {
        throw std::invalid_argument("Decoder encoder input is outside profile");
    }

    encoder_output_ = encoder_output;
    encoder_frames_ = encoder_frames;
    // Prefix features from earlier policies stay in decoder_features_; the
    // re-evaluated last token overwrites its stale feature slot.
    feature_count_ = canonical_length_;
    cache_length_ = canonical_length_;
    current_cache_index_ = 0;
    policy_active_ = true;

    target_input_.data()[0] = canonical_last_token_;
    position_.data()[0] = static_cast<std::int64_t>(canonical_length_);

    auto& context = step_engine_.context();
    set_shape(context, "target_input", make_dims({1, 1}));
    set_shape(
        context,
        "encoder_output",
        make_dims(
            {1,
             static_cast<std::int64_t>(encoder_frames_),
             static_cast<std::int64_t>(kDecoderDimension)}));
    set_shape(context, "position", make_dims({1}));
    const auto cache_shape = make_dims(
        {static_cast<std::int64_t>(kDecoderLayers),
         1,
         static_cast<std::int64_t>(kAttentionHeads),
         static_cast<std::int64_t>(canonical_length_),
         static_cast<std::int64_t>(kHeadDimension)});
    set_shape(context, "self_k_cache", cache_shape);
    set_shape(context, "self_v_cache", cache_shape);

    bind(context, "target_input", target_input_.device_data());
    bind(
        context,
        "encoder_output",
        const_cast<__half*>(encoder_output_));
    bind(context, "position", position_.device_data());
    bind(context, "self_k_cache", canonical_key_.data());
    bind(context, "self_v_cache", canonical_value_.data());
    bind(
        context,
        "decoder_output",
        decoder_features_.data() + canonical_length_ * kDecoderDimension);
    bind(context, "logits", logits_.data());
    bind(context, "decision_values", decision_values_.data());
    bind(context, "self_k_cache_out", key_cache_[0].data());
    bind(context, "self_v_cache_out", value_cache_[0].data());

    const auto output_cache_length = canonical_length_ + 1U;
    require_output_shape(
        context,
        "self_k_cache_out",
        make_dims(
            {static_cast<std::int64_t>(kDecoderLayers),
             1,
             static_cast<std::int64_t>(kAttentionHeads),
             static_cast<std::int64_t>(output_cache_length),
             static_cast<std::int64_t>(kHeadDimension)}));
    if (!context.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT decoder resume enqueueV3 failed");
    }

    auto result = collect_result(canonical_length_, 1, output_cache_length);
    feature_count_ = output_cache_length;
    cache_length_ = output_cache_length;
    return result;
}

void DecoderRuntime::capture_canonical(const std::int64_t last_token) {
    if (cache_length_ < kMinimumTargetLength + 1U) {
        throw std::logic_error(
            "Decoder canonical capture requires at least three cached tokens");
    }
    const auto trimmed_length = cache_length_ - 1U;
    // Cache layout is [layers, 1, heads, length, head_dim]; trimming the
    // length axis is a strided row copy over layers*heads rows.
    const auto rows = kDecoderLayers * kAttentionHeads;
    const auto source_pitch = cache_length_ * kHeadDimension * sizeof(__half);
    const auto destination_pitch =
        trimmed_length * kHeadDimension * sizeof(__half);
    check_cuda(
        cudaMemcpy2DAsync(
            canonical_key_.data(),
            destination_pitch,
            key_cache_[current_cache_index_].data(),
            source_pitch,
            destination_pitch,
            rows,
            cudaMemcpyDeviceToDevice,
            stream_.get()),
        "cudaMemcpy2DAsync canonical key cache");
    check_cuda(
        cudaMemcpy2DAsync(
            canonical_value_.data(),
            destination_pitch,
            value_cache_[current_cache_index_].data(),
            source_pitch,
            destination_pitch,
            rows,
            cudaMemcpyDeviceToDevice,
            stream_.get()),
        "cudaMemcpy2DAsync canonical value cache");
    stream_.synchronize();
    canonical_length_ = trimmed_length;
    canonical_last_token_ = last_token;
    canonical_valid_ = true;
}

DecoderResult DecoderRuntime::collect_result(
    const std::size_t feature_offset,
    const std::size_t feature_count,
    const std::size_t cache_length) {
    launch_argmax_half(
        logits_.data(),
        kVocabularySize,
        argmax_scratch_values_.data(),
        argmax_scratch_indices_.data(),
        argmax_scratch_values_.size(),
        selected_index_.device_data(),
        stream_.get());
    launch_min_float(
        decision_values_.data(),
        kDecisionCount,
        decision_probability_.device_data(),
        stream_.get());
    stream_.synchronize();

    const auto selected = selected_index_.data()[0];
    const auto probability = decision_probability_.data()[0];
    if (selected < 0 ||
        selected >= static_cast<std::int64_t>(kVocabularySize) ||
        !std::isfinite(probability)) {
        throw std::runtime_error("Decoder returned invalid control values");
    }
    return DecoderResult{
        .selected_index = selected,
        .decision_probability = probability,
        .feature_offset = feature_offset,
        .feature_count = feature_count,
        .cache_length = cache_length,
    };
}

void DecoderRuntime::copy_result_features(
    const DecoderResult& result,
    const std::span<__half> destination) const {
    const auto element_count = result.feature_count * kDecoderDimension;
    if (result.feature_offset + result.feature_count > feature_count_ ||
        destination.size() < element_count) {
        throw std::length_error("Decoder feature destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            decoder_features_.data() +
                result.feature_offset * kDecoderDimension,
            element_count * sizeof(__half),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy decoder features");
}

}  // namespace streamglass
