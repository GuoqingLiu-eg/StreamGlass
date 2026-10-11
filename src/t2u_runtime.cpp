#include "streamglass/t2u_runtime.hpp"

#include "streamglass/cuda_kernels.hpp"

#include <NvInfer.h>

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

bool dimensions_equal(
    const nvinfer1::Dims& left, const nvinfer1::Dims& right) {
    if (left.nbDims != right.nbDims) {
        return false;
    }
    for (std::int32_t axis = 0; axis < left.nbDims; ++axis) {
        if (left.d[axis] != right.d[axis]) {
            return false;
        }
    }
    return true;
}

void require_output_shape(
    nvinfer1::IExecutionContext& context,
    const char* name,
    const nvinfer1::Dims& expected) {
    const auto actual = context.getTensorShape(name);
    if (!dimensions_equal(actual, expected)) {
        throw std::runtime_error(
            "Unexpected TensorRT output shape for " + std::string(name) +
            ": expected " + dims_to_string(expected) + ", got " +
            dims_to_string(actual));
    }
}

}  // namespace

T2uRuntime::T2uRuntime(
    const std::filesystem::path& frontend_engine_path,
    const std::filesystem::path& decoder_engine_path,
    const TokenizerAssets& tokenizer)
    : tokenizer_(tokenizer),
      frontend_engine_("t2u_frontend", frontend_engine_path),
      decoder_engine_("t2u_decoder", decoder_engine_path),
      stream_(),
      character_indices_(kMaximumCharacterCount),
      character_lengths_(kMaximumTextLength),
      total_unit_count_(1),
      output_range_(2),
      output_state_(2),
      character_features_(kMaximumCharacterCount * kFeatureDimension),
      durations_(kMaximumCharacterCount),
      duration_offsets_(kMaximumCharacterCount + 1U),
      expanded_features_(kMaximumUnitCount * kFeatureDimension),
      unit_ids_int32_(kMaximumUnitCount),
      unit_ids_int64_(kMaximumUnitCount) {
    validate_engines();
}

void T2uRuntime::validate_engines() {
    const auto& frontend = frontend_engine_.engine();
    const auto& decoder = decoder_engine_.engine();
    if (frontend.getNbIOTensors() != 5 || decoder.getNbIOTensors() != 2 ||
        frontend.getTensorDataType("text_decoder_output") !=
            nvinfer1::DataType::kHALF ||
        frontend.getTensorDataType("char_seqs") !=
            nvinfer1::DataType::kINT64 ||
        frontend.getTensorDataType("char_lens") !=
            nvinfer1::DataType::kINT64 ||
        frontend.getTensorDataType("char_features") !=
            nvinfer1::DataType::kHALF ||
        frontend.getTensorDataType("durations") !=
            nvinfer1::DataType::kINT64 ||
        decoder.getTensorDataType("expanded_char_features") !=
            nvinfer1::DataType::kHALF ||
        decoder.getTensorDataType("unit_ids") !=
            nvinfer1::DataType::kINT32) {
        throw std::runtime_error("T2U TensorRT I/O contract changed");
    }
}

T2uResult T2uRuntime::run(
    const std::span<const std::int64_t> token_indices,
    const __half* decoder_features,
    const std::size_t decoder_feature_count,
    const std::size_t duration_start_index,
    const bool source_finished,
    const std::size_t minimum_unit_chunk,
    const std::uint64_t text_version) {
    if (text_version != 0 && memo_valid_ &&
        text_version == memo_text_version_) {
        // The committed text and its decoder features are unchanged, so the
        // frontend and unit decoder products are bit-identical and stay
        // resident. Only the cursor-dependent policy slice must be redone.
        if (duration_start_index > last_character_count_ ||
            (duration_start_index == last_character_count_ &&
             !source_finished)) {
            throw std::invalid_argument(
                "T2U memoized duration cursor is outside the character range");
        }
        launch_t2u_policy_slice(
            duration_offsets_.data(),
            last_character_count_,
            duration_start_index,
            source_finished,
            minimum_unit_chunk,
            output_range_.device_data(),
            output_state_.device_data(),
            stream_.get());
        stream_.synchronize();

        const auto unit_offset_raw = output_range_.data()[0];
        const auto unit_count_raw = output_range_.data()[1];
        const auto action_raw = output_state_.data()[0];
        const auto next_duration_raw = output_state_.data()[1];
        if (unit_offset_raw < 0 || unit_count_raw < 0 ||
            unit_offset_raw + unit_count_raw >
                static_cast<std::int64_t>(memo_total_units_) ||
            action_raw < static_cast<std::int32_t>(T2uAction::read) ||
            action_raw > static_cast<std::int32_t>(T2uAction::finished) ||
            next_duration_raw < 0 ||
            next_duration_raw >
                static_cast<std::int32_t>(last_character_count_) ||
            (next_duration_raw ==
                 static_cast<std::int32_t>(last_character_count_) &&
             action_raw != static_cast<std::int32_t>(T2uAction::finished))) {
            throw std::runtime_error(
                "T2U duration policy returned invalid controls");
        }
        return T2uResult{
            .action = static_cast<T2uAction>(action_raw),
            .character_count = last_character_count_,
            .total_unit_count = memo_total_units_,
            .unit_offset = static_cast<std::size_t>(unit_offset_raw),
            .unit_count = static_cast<std::size_t>(unit_count_raw),
            .next_duration_start_index =
                static_cast<std::size_t>(next_duration_raw),
        };
    }
    memo_valid_ = false;
    if (decoder_features == nullptr ||
        token_indices.size() != decoder_feature_count ||
        decoder_feature_count < kMinimumTextLength ||
        decoder_feature_count > kMaximumTextLength) {
        throw std::invalid_argument("T2U text input is outside profile");
    }

    const auto input_size = tokenizer_.build_t2u_inputs(
        token_indices, character_indices_.span(), character_lengths_.span());
    if (input_size.token_count != decoder_feature_count ||
        input_size.character_count < kMinimumCharacterCount ||
        input_size.character_count > kMaximumCharacterCount ||
        duration_start_index > input_size.character_count ||
        (duration_start_index == input_size.character_count &&
         !source_finished)) {
        throw std::invalid_argument("T2U character input is outside profile");
    }
    last_text_length_ = input_size.token_count;
    last_character_count_ = input_size.character_count;

    auto& frontend = frontend_engine_.context();
    set_shape(
        frontend,
        "text_decoder_output",
        make_dims(
            {1,
             static_cast<std::int64_t>(decoder_feature_count),
             static_cast<std::int64_t>(kFeatureDimension)}));
    set_shape(
        frontend,
        "char_seqs",
        make_dims(
            {1, static_cast<std::int64_t>(input_size.character_count)}));
    set_shape(
        frontend,
        "char_lens",
        make_dims({1, static_cast<std::int64_t>(input_size.token_count)}));
    bind(
        frontend,
        "text_decoder_output",
        const_cast<__half*>(decoder_features));
    bind(frontend, "char_seqs", character_indices_.device_data());
    bind(frontend, "char_lens", character_lengths_.device_data());
    bind(frontend, "char_features", character_features_.data());
    bind(frontend, "durations", durations_.data());
    const auto character_shape = make_dims(
        {1,
         static_cast<std::int64_t>(input_size.character_count),
         static_cast<std::int64_t>(kFeatureDimension)});
    require_output_shape(frontend, "char_features", character_shape);
    require_output_shape(
        frontend,
        "durations",
        make_dims(
            {1, static_cast<std::int64_t>(input_size.character_count)}));
    if (!frontend.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT T2U frontend enqueueV3 failed");
    }

    launch_duration_prefix_sum(
        durations_.data(),
        input_size.character_count,
        duration_offsets_.data(),
        total_unit_count_.device_data(),
        stream_.get());
    launch_t2u_policy_slice(
        duration_offsets_.data(),
        input_size.character_count,
        duration_start_index,
        source_finished,
        minimum_unit_chunk,
        output_range_.device_data(),
        output_state_.device_data(),
        stream_.get());
    stream_.synchronize();

    const auto total_units_raw = total_unit_count_.data()[0];
    const auto unit_offset_raw = output_range_.data()[0];
    const auto unit_count_raw = output_range_.data()[1];
    const auto action_raw = output_state_.data()[0];
    const auto next_duration_raw = output_state_.data()[1];
    if (total_units_raw <= 0 ||
        total_units_raw > static_cast<std::int64_t>(kMaximumUnitCount) ||
        unit_offset_raw < 0 || unit_count_raw < 0 ||
        unit_offset_raw + unit_count_raw > total_units_raw ||
        action_raw < static_cast<std::int32_t>(T2uAction::read) ||
        action_raw > static_cast<std::int32_t>(T2uAction::finished) ||
        next_duration_raw < 0 ||
        next_duration_raw >
            static_cast<std::int32_t>(input_size.character_count) ||
        (next_duration_raw ==
             static_cast<std::int32_t>(input_size.character_count) &&
         action_raw != static_cast<std::int32_t>(T2uAction::finished))) {
        throw std::runtime_error("T2U duration policy returned invalid controls");
    }
    const auto total_units = static_cast<std::size_t>(total_units_raw);

    launch_expand_by_duration(
        character_features_.data(),
        duration_offsets_.data(),
        input_size.character_count,
        kFeatureDimension,
        expanded_features_.data(),
        stream_.get());

    auto& decoder = decoder_engine_.context();
    set_shape(
        decoder,
        "expanded_char_features",
        make_dims(
            {1,
             static_cast<std::int64_t>(total_units),
             static_cast<std::int64_t>(kFeatureDimension)}));
    bind(decoder, "expanded_char_features", expanded_features_.data());
    bind(decoder, "unit_ids", unit_ids_int32_.data());
    require_output_shape(
        decoder,
        "unit_ids",
        make_dims({1, static_cast<std::int64_t>(total_units)}));
    if (!decoder.enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT T2U decoder enqueueV3 failed");
    }
    launch_int32_to_int64(
        unit_ids_int32_.data(),
        unit_ids_int64_.data(),
        total_units,
        stream_.get());
    stream_.synchronize();

    if (text_version != 0) {
        memo_text_version_ = text_version;
        memo_total_units_ = total_units;
        memo_valid_ = true;
    }

    return T2uResult{
        .action = static_cast<T2uAction>(action_raw),
        .character_count = input_size.character_count,
        .total_unit_count = total_units,
        .unit_offset = static_cast<std::size_t>(unit_offset_raw),
        .unit_count = static_cast<std::size_t>(unit_count_raw),
        .next_duration_start_index =
            static_cast<std::size_t>(next_duration_raw),
    };
}

void T2uRuntime::copy_durations(
    const std::span<std::int64_t> destination,
    const std::size_t character_count) const {
    if (character_count > last_character_count_ ||
        destination.size() < character_count) {
        throw std::length_error("T2U duration destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            durations_.data(),
            character_count * sizeof(std::int64_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy T2U durations");
}

void T2uRuntime::copy_unit_ids(
    const std::span<std::int32_t> destination,
    const std::size_t unit_count) const {
    if (unit_count > kMaximumUnitCount || destination.size() < unit_count) {
        throw std::length_error("T2U unit destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            unit_ids_int32_.data(),
            unit_count * sizeof(std::int32_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy T2U unit IDs");
}

void T2uRuntime::copy_character_features(
    const std::span<__half> destination,
    const std::size_t character_count) const {
    const auto element_count = character_count * kFeatureDimension;
    if (character_count > last_character_count_ ||
        destination.size() < element_count) {
        throw std::length_error("T2U character feature destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            character_features_.data(),
            element_count * sizeof(__half),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy T2U character features");
}

void T2uRuntime::copy_expanded_features(
    const std::span<__half> destination,
    const std::size_t unit_count) const {
    const auto element_count = unit_count * kFeatureDimension;
    if (unit_count > kMaximumUnitCount || destination.size() < element_count) {
        throw std::length_error("T2U expanded feature destination is too small");
    }
    check_cuda(
        cudaMemcpy(
            destination.data(),
            expanded_features_.data(),
            element_count * sizeof(__half),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy T2U expanded features");
}

}  // namespace streamglass
