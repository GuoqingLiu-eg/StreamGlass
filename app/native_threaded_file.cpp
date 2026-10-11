#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "streamglass/buffer_pool.hpp"
#include "streamglass/spsc_ring.hpp"
#include "streamglass/streaming_pipeline.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <syncstream>
#include <thread>

#include <sys/resource.h>

namespace {

constexpr std::size_t kAudioSlotCount = 16;
constexpr std::size_t kPlaybackSlotCount = 8;
// Official policy semantics require one decision for every 320 ms segment.
// Queue backpressure is allowed, but policy calls must never be coalesced.
constexpr std::size_t kMaximumCoalescedBlocks = 1;
constexpr auto kQueuePoll = std::chrono::microseconds(100);
constexpr double kInferenceDeadlineMs = 200.0;

struct AudioBlock {
    std::uint64_t utterance_id = 0;
    std::uint64_t sequence = 0;
    std::size_t sample_count = 0;
    bool source_finished = false;
    std::chrono::steady_clock::time_point captured_at{};
    std::array<float, streamglass::StreamingPipeline::kSegmentSamples> pcm{};
};

struct PlaybackBlock {
    PlaybackBlock()
        : pcm(streamglass::VocoderRuntime::kMaximumSampleCount) {}

    streamglass::MappedHostBuffer<float> pcm;
    std::uint64_t utterance_id = 0;
    std::uint64_t write_sequence = 0;
    std::size_t sample_count = 0;
    bool utterance_finished = false;
};

struct RuntimeQueues {
    streamglass::SpscBufferPool<AudioBlock, kAudioSlotCount> audio_pool;
    streamglass::SpscRing<streamglass::BufferHandle, kAudioSlotCount>
        capture_to_inference;
    streamglass::SpscBufferPool<PlaybackBlock, kPlaybackSlotCount>
        playback_pool;
    streamglass::SpscRing<streamglass::BufferHandle, kPlaybackSlotCount>
        inference_to_playback;
};

struct RuntimeState {
    std::atomic<bool> stop{false};
    std::atomic<bool> capture_done{false};
    std::atomic<bool> inference_done{false};
    std::mutex failure_mutex;
    std::exception_ptr failure;

    std::size_t input_samples = 0;
    std::size_t audio_queue_high_water = 0;
    std::size_t playback_queue_high_water = 0;
    std::size_t coalesced_batch_count = 0;
    std::size_t maximum_coalesced_blocks = 1;
    std::size_t policy_count = 0;
    std::size_t write_count = 0;
    std::size_t inference_deadline_misses = 0;
    double maximum_inference_ms = 0.0;
    double maximum_frontend_ms = 0.0;
    double maximum_encoder_ms = 0.0;
    double maximum_decoder_ms = 0.0;
    double maximum_t2u_ms = 0.0;
    double maximum_vocoder_ms = 0.0;
    double maximum_capture_queue_ms = 0.0;
    std::string translation;
    std::size_t output_samples = 0;
    double output_peak = 0.0;
    double output_square_sum = 0.0;
    std::uint64_t output_checksum = 1'469'598'103'934'665'603ULL;
};

void fail(RuntimeState& state, const std::exception_ptr error) {
    const auto was_stopped = state.stop.exchange(true, std::memory_order_acq_rel);
    if (!was_stopped) {
        std::lock_guard lock(state.failure_mutex);
        state.failure = error;
    }
}

std::string environment_or(const char* name, const std::string& fallback) {
    const auto* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : value;
}

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const auto character : value) {
        switch (character) {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result.push_back(character);
            break;
        }
    }
    return result;
}

std::size_t current_rss_kib() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            std::size_t value = 0;
            std::string unit;
            status >> value >> unit;
            if (unit != "kB") {
                throw std::runtime_error("Unexpected VmRSS unit: " + unit);
            }
            return value;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    throw std::runtime_error("Cannot read VmRSS from /proc/self/status");
}

std::size_t peak_rss_kib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage(RUSAGE_SELF) failed");
    }
    return static_cast<std::size_t>(usage.ru_maxrss);
}

std::size_t cuda_free_bytes() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemGetInfo failed: ") +
            cudaGetErrorString(status));
    }
    return free_bytes;
}

template <typename Pool>
std::optional<streamglass::BufferHandle> acquire_until_stopped(
    Pool& pool, const RuntimeState& state) {
    while (!state.stop.load(std::memory_order_acquire)) {
        if (auto handle = pool.try_acquire()) {
            return handle;
        }
        std::this_thread::sleep_for(kQueuePoll);
    }
    return std::nullopt;
}

template <typename Queue>
bool push_until_stopped(
    Queue& queue,
    const streamglass::BufferHandle handle,
    const RuntimeState& state) {
    while (!state.stop.load(std::memory_order_acquire)) {
        if (queue.try_push(handle)) {
            return true;
        }
        std::this_thread::sleep_for(kQueuePoll);
    }
    return false;
}

void capture_file(
    const std::filesystem::path& path,
    const bool realtime,
    const std::size_t repeat_count,
    RuntimeQueues& queues,
    RuntimeState& state) {
    try {
        drwav wav{};
        if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
            throw std::runtime_error("Cannot open input WAV: " + path.string());
        }
        struct WavGuard {
            drwav* wav;
            ~WavGuard() { drwav_uninit(wav); }
        } guard{&wav};
        if (wav.channels != 1 ||
            wav.sampleRate != streamglass::OfficialOnlineFbank::kSampleRate) {
            throw std::runtime_error("Input WAV must be mono 16kHz PCM");
        }
        if (wav.totalPCMFrameCount == 0 ||
            wav.totalPCMFrameCount > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Input WAV frame count is invalid");
        }

        const auto frames_per_utterance =
            static_cast<std::size_t>(wav.totalPCMFrameCount);
        const auto started_at = std::chrono::steady_clock::now();
        std::size_t captured_samples = 0;
        for (std::size_t utterance = 0;
             utterance < repeat_count &&
             !state.stop.load(std::memory_order_acquire);
             ++utterance) {
            if (utterance > 0U && !drwav_seek_to_pcm_frame(&wav, 0U)) {
                throw std::runtime_error("Cannot rewind input WAV");
            }
            std::size_t utterance_samples = 0;
            std::uint64_t sequence = 0;
            while (!state.stop.load(std::memory_order_acquire) &&
                   utterance_samples < frames_per_utterance) {
                const auto handle = acquire_until_stopped(
                    queues.audio_pool, state);
                if (!handle) {
                    break;
                }
                auto* block = queues.audio_pool.get(*handle);
                if (block == nullptr) {
                    throw std::logic_error("Audio pool returned a stale handle");
                }
                const auto remaining =
                    frames_per_utterance - utterance_samples;
                const auto requested = std::min(
                    remaining, streamglass::StreamingPipeline::kSegmentSamples);
                const auto read = static_cast<std::size_t>(
                    drwav_read_pcm_frames_f32(
                        &wav, requested, block->pcm.data()));
                if (read != requested) {
                    static_cast<void>(queues.audio_pool.release(*handle));
                    throw std::runtime_error(
                        "Input WAV ended before its frame count");
                }
                utterance_samples += read;
                captured_samples += read;
                block->utterance_id = utterance + 1U;
                block->sequence = sequence;
                block->sample_count = read;
                block->source_finished =
                    utterance_samples == frames_per_utterance;

                if (realtime) {
                    const auto elapsed = std::chrono::duration<double>(
                        static_cast<double>(captured_samples) /
                        static_cast<double>(
                            streamglass::OfficialOnlineFbank::kSampleRate));
                    std::this_thread::sleep_until(started_at + elapsed);
                }
                block->captured_at = std::chrono::steady_clock::now();
                if (!push_until_stopped(
                        queues.capture_to_inference, *handle, state)) {
                    static_cast<void>(queues.audio_pool.release(*handle));
                    break;
                }
                state.audio_queue_high_water = std::max(
                    state.audio_queue_high_water,
                    queues.capture_to_inference.size());
                ++sequence;
            }
        }
        state.input_samples = captured_samples;
    } catch (...) {
        fail(state, std::current_exception());
    }
    state.capture_done.store(true, std::memory_order_release);
}

void warmup_pipeline(
    streamglass::StreamingPipeline& pipeline,
    const std::filesystem::path& path) {
    drwav wav{};
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw std::runtime_error("Cannot open warmup WAV: " + path.string());
    }
    struct WavGuard {
        drwav* wav;
        ~WavGuard() { drwav_uninit(wav); }
    } guard{&wav};
    if (wav.channels != 1 ||
        wav.sampleRate != streamglass::OfficialOnlineFbank::kSampleRate ||
        wav.totalPCMFrameCount == 0 ||
        wav.totalPCMFrameCount > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Warmup WAV must be non-empty mono 16kHz PCM");
    }

    std::array<float, streamglass::StreamingPipeline::kSegmentSamples> pcm{};
    streamglass::MappedHostBuffer<float> playback(
        streamglass::VocoderRuntime::kMaximumSampleCount);
    std::size_t total_read = 0;
    const auto total_frames =
        static_cast<std::size_t>(wav.totalPCMFrameCount);
    while (total_read < total_frames) {
        const auto requested = std::min(
            total_frames - total_read,
            streamglass::StreamingPipeline::kSegmentSamples);
        const auto read = static_cast<std::size_t>(
            drwav_read_pcm_frames_f32(&wav, requested, pcm.data()));
        if (read != requested) {
            throw std::runtime_error("Warmup WAV ended before its frame count");
        }
        total_read += read;
        const auto step = pipeline.process_audio(
            std::span<const float>(pcm.data(), read),
            total_read == total_frames);
        if (step.output_unit_count > 0) {
            static_cast<void>(pipeline.synthesize(step, playback));
        }
    }
    pipeline.reset();
}

void run_inference(
    streamglass::StreamingPipeline& pipeline,
    RuntimeQueues& queues,
    RuntimeState& state) {
    try {
        std::uint64_t write_sequence = 0;
        std::optional<streamglass::BufferHandle> deferred_audio;
        std::optional<std::uint64_t> active_utterance;
        bool active_utterance_finished = false;
        while (!state.stop.load(std::memory_order_acquire)) {
            streamglass::BufferHandle audio_handle{};
            if (deferred_audio) {
                audio_handle = *deferred_audio;
                deferred_audio.reset();
            } else if (!queues.capture_to_inference.try_pop(audio_handle)) {
                if (state.capture_done.load(std::memory_order_acquire) &&
                    queues.capture_to_inference.empty()) {
                    break;
                }
                std::this_thread::sleep_for(kQueuePoll);
                continue;
            }
            auto* first_audio = queues.audio_pool.get(audio_handle);
            if (first_audio == nullptr) {
                throw std::logic_error("Inference received a stale audio handle");
            }
            if (!active_utterance) {
                active_utterance = first_audio->utterance_id;
            } else if (*active_utterance != first_audio->utterance_id) {
                if (!active_utterance_finished) {
                    throw std::runtime_error(
                        "A new utterance arrived before the previous EOS");
                }
                pipeline.reset();
                active_utterance = first_audio->utterance_id;
                active_utterance_finished = false;
            }

            std::array<streamglass::BufferHandle, kMaximumCoalescedBlocks>
                audio_handles{};
            std::array<streamglass::PipelineAudioSegment,
                       kMaximumCoalescedBlocks>
                segments{};
            audio_handles[0] = audio_handle;
            segments[0] = streamglass::PipelineAudioSegment{
                .samples = std::span<const float>(
                    first_audio->pcm.data(), first_audio->sample_count),
                .source_finished = first_audio->source_finished,
            };
            std::size_t block_count = 1;
            auto expected_sequence = first_audio->sequence + 1U;
            while (block_count < kMaximumCoalescedBlocks &&
                   !segments[block_count - 1U].source_finished) {
                streamglass::BufferHandle next_handle{};
                if (!queues.capture_to_inference.try_pop(next_handle)) {
                    break;
                }
                auto* next_audio = queues.audio_pool.get(next_handle);
                if (next_audio == nullptr) {
                    throw std::logic_error(
                        "Inference received a stale coalesced audio handle");
                }
                if (next_audio->utterance_id != *active_utterance ||
                    next_audio->sequence != expected_sequence) {
                    deferred_audio = next_handle;
                    break;
                }
                audio_handles[block_count] = next_handle;
                segments[block_count] = streamglass::PipelineAudioSegment{
                    .samples = std::span<const float>(
                        next_audio->pcm.data(), next_audio->sample_count),
                    .source_finished = next_audio->source_finished,
                };
                ++block_count;
                ++expected_sequence;
            }

            double queue_ms = 0.0;
            for (std::size_t index = 0; index < block_count; ++index) {
                const auto* audio = queues.audio_pool.get(audio_handles[index]);
                if (audio == nullptr) {
                    throw std::logic_error(
                        "Coalesced audio handle expired before inference");
                }
                queue_ms = std::max(
                    queue_ms,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - audio->captured_at)
                        .count());
            }
            state.maximum_capture_queue_ms =
                std::max(state.maximum_capture_queue_ms, queue_ms);

            const auto utterance_id = *active_utterance;
            const auto source_finished = segments[block_count - 1U].source_finished;
            const auto step = pipeline.process_audio_batch(
                std::span(segments).first(block_count));
            for (std::size_t index = 0; index < block_count; ++index) {
                if (!queues.audio_pool.release(audio_handles[index])) {
                    throw std::logic_error(
                        "Inference could not release coalesced audio slot");
                }
            }
            if (block_count > 1U) {
                ++state.coalesced_batch_count;
                state.maximum_coalesced_blocks = std::max(
                    state.maximum_coalesced_blocks, block_count);
            }

            double total_inference_ms = step.inference_ms();
            double vocoder_ms = 0.0;
            if (step.policy_executed) {
                ++state.policy_count;
            }
            if (step.output_unit_count > 0) {
                const auto playback_handle =
                    acquire_until_stopped(queues.playback_pool, state);
                if (!playback_handle) {
                    break;
                }
                auto* playback = queues.playback_pool.get(*playback_handle);
                if (playback == nullptr) {
                    throw std::logic_error(
                        "Playback pool returned a stale handle");
                }
                const auto synthesis = pipeline.synthesize(step, playback->pcm);
                vocoder_ms = synthesis.vocoder_ms;
                total_inference_ms += vocoder_ms;
                playback->utterance_id = utterance_id;
                playback->write_sequence = write_sequence++;
                playback->sample_count = synthesis.sample_count;
                playback->utterance_finished = step.t2u_finished;
                if (!push_until_stopped(
                        queues.inference_to_playback,
                        *playback_handle,
                        state)) {
                    static_cast<void>(
                        queues.playback_pool.release(*playback_handle));
                    break;
                }
                state.playback_queue_high_water = std::max(
                    state.playback_queue_high_water,
                    queues.inference_to_playback.size());
                ++state.write_count;
            }
            state.maximum_inference_ms =
                std::max(state.maximum_inference_ms, total_inference_ms);
            state.maximum_frontend_ms =
                std::max(state.maximum_frontend_ms, step.frontend_ms);
            state.maximum_encoder_ms =
                std::max(state.maximum_encoder_ms, step.encoder_ms);
            state.maximum_decoder_ms =
                std::max(state.maximum_decoder_ms, step.decoder_ms);
            state.maximum_t2u_ms =
                std::max(state.maximum_t2u_ms, step.t2u_ms);
            state.maximum_vocoder_ms =
                std::max(state.maximum_vocoder_ms, vocoder_ms);
            if (step.policy_executed &&
                total_inference_ms > kInferenceDeadlineMs) {
                ++state.inference_deadline_misses;
            }

            if (step.policy_executed) {
                std::osyncstream(std::cout)
                    << "[infer] utterance=" << utterance_id
                    << " serial=" << step.serial
                    << " blocks=" << block_count
                    << " fbank=" << step.fbank_frames
                    << " action="
                    << (step.text_action == streamglass::TextPolicyAction::write
                            ? "WRITE"
                            : "READ")
                    << " units=" << step.output_unit_count
                    << " queue_ms=" << queue_ms
                    << " frontend_ms=" << step.frontend_ms
                    << " encoder_ms=" << step.encoder_ms
                    << " decoder_ms=" << step.decoder_ms
                    << " t2u_ms=" << step.t2u_ms
                    << " vocoder_ms=" << vocoder_ms
                    << " inference_ms=" << total_inference_ms << '\n';
            }
            if (source_finished) {
                state.translation = pipeline.translation();
                active_utterance_finished = true;
            }
        }
    } catch (...) {
        fail(state, std::current_exception());
    }
    state.inference_done.store(true, std::memory_order_release);
}

void consume_playback(
    const bool realtime,
    RuntimeQueues& queues,
    RuntimeState& state) {
    try {
        while (!state.stop.load(std::memory_order_acquire)) {
            streamglass::BufferHandle handle{};
            if (!queues.inference_to_playback.try_pop(handle)) {
                if (state.inference_done.load(std::memory_order_acquire) &&
                    queues.inference_to_playback.empty()) {
                    break;
                }
                std::this_thread::sleep_for(kQueuePoll);
                continue;
            }
            auto* block = queues.playback_pool.get(handle);
            if (block == nullptr) {
                throw std::logic_error("Playback received a stale handle");
            }
            double square_sum = 0.0;
            double peak = 0.0;
            for (std::size_t index = 0; index < block->sample_count; ++index) {
                const auto sample = block->pcm.data()[index];
                square_sum += static_cast<double>(sample) * sample;
                peak = std::max(peak, std::abs(static_cast<double>(sample)));
                state.output_checksum ^=
                    static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(sample));
                state.output_checksum *= 1'099'511'628'211ULL;
            }
            state.output_samples += block->sample_count;
            state.output_square_sum += square_sum;
            state.output_peak = std::max(state.output_peak, peak);

            std::osyncstream(std::cout)
                << "[playback-null] utterance=" << block->utterance_id
                << " write=" << block->write_sequence
                << " samples=" << block->sample_count
                << " finished="
                << (block->utterance_finished ? "true" : "false") << '\n';
            if (realtime) {
                std::this_thread::sleep_for(std::chrono::duration<double>(
                    static_cast<double>(block->sample_count) /
                    static_cast<double>(streamglass::VocoderRuntime::kSampleRate)));
            }
            if (!queues.playback_pool.release(handle)) {
                throw std::logic_error("Playback could not release output slot");
            }
        }
    } catch (...) {
        fail(state, std::current_exception());
    }
}

}  // namespace

std::size_t parse_positive_size(
    const std::string_view value, const std::string_view option) {
    std::size_t parsed = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed == 0U) {
        throw std::invalid_argument(
            std::string(option) + " requires a positive integer");
    }
    return parsed;
}

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::cerr
                << "usage: streamglass-native-threaded-file <input.wav> "
                   "<engine-dir> <runtime-assets> <target-language> "
                   "[--fast] [--repeat N]\n";
            return 2;
        }
        bool realtime = true;
        std::size_t repeat_count = 1;
        bool repeat_seen = false;
        for (int index = 5; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--fast") {
                if (!realtime) {
                    throw std::invalid_argument("--fast was specified twice");
                }
                realtime = false;
            } else if (option == "--repeat") {
                if (repeat_seen || index + 1 >= argc) {
                    throw std::invalid_argument(
                        "--repeat requires exactly one positive integer");
                }
                repeat_count = parse_positive_size(argv[++index], "--repeat");
                repeat_seen = true;
            } else {
                throw std::invalid_argument(
                    "Unknown option: " + std::string(option));
            }
        }

        const std::filesystem::path input_path(argv[1]);
        const std::filesystem::path engine_dir(argv[2]);
        const std::filesystem::path runtime_assets(argv[3]);
        const std::string target_language(argv[4]);
        const streamglass::StreamingPipelinePaths paths{
            .encoder = engine_dir / environment_or(
                "STREAMGLASS_ENCODER_ENGINE_NAME",
                "encoder_cumulative_fp16.engine"),
            .decoder_prefill = engine_dir / environment_or(
                "STREAMGLASS_DECODER_PREFILL_ENGINE_NAME",
                "decoder_prefill_mixed.engine"),
            .decoder_step = engine_dir / environment_or(
                "STREAMGLASS_DECODER_STEP_ENGINE_NAME",
                "decoder_step_mixed.engine"),
            .t2u_frontend = engine_dir / environment_or(
                "STREAMGLASS_T2U_FRONTEND_ENGINE_NAME",
                "t2u_frontend_fp16.engine"),
            .t2u_decoder = engine_dir / "t2u_decoder_fp32.engine",
            .vocoder = engine_dir / "vocoder_fp16.engine",
            .runtime_assets = runtime_assets,
        };

        const streamglass::StreamingPipelineOptions pipeline_options{
            .cached_decoder =
                environment_or("STREAMGLASS_DECODER_CACHED", "0") == "1",
        };
        const auto load_started = std::chrono::steady_clock::now();
        streamglass::StreamingPipeline pipeline(
            paths, target_language, pipeline_options);
        const auto load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_started)
                                 .count();
        const auto warmup_started = std::chrono::steady_clock::now();
        if (environment_or("STREAMGLASS_SKIP_WARMUP", "0") != "1") {
            const auto warmup_path = std::filesystem::path(environment_or(
                "STREAMGLASS_WARMUP_WAV", input_path.string()));
            warmup_pipeline(pipeline, warmup_path);
        }
        const auto warmup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - warmup_started)
                                   .count();
        auto queues = std::make_unique<RuntimeQueues>();
        RuntimeState state;
        const auto rss_before_kib = current_rss_kib();
        const auto cuda_free_before_bytes = cuda_free_bytes();

        const auto run_started = std::chrono::steady_clock::now();
        std::thread capture(
            capture_file,
            std::cref(input_path),
            realtime,
            repeat_count,
            std::ref(*queues),
            std::ref(state));
        std::thread inference(
            run_inference,
            std::ref(pipeline),
            std::ref(*queues),
            std::ref(state));
        std::thread playback(
            consume_playback,
            realtime,
            std::ref(*queues),
            std::ref(state));
        capture.join();
        inference.join();
        playback.join();
        const auto wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - run_started)
                                 .count();
        const auto rss_after_kib = current_rss_kib();
        const auto peak_rss = peak_rss_kib();
        const auto cuda_free_after_bytes = cuda_free_bytes();
        const auto cuda_free_delta_bytes =
            static_cast<std::int64_t>(cuda_free_after_bytes) -
            static_cast<std::int64_t>(cuda_free_before_bytes);

        {
            std::lock_guard lock(state.failure_mutex);
            if (state.failure) {
                std::rethrow_exception(state.failure);
            }
        }
        const auto rms = state.output_samples == 0
            ? 0.0
            : std::sqrt(
                  state.output_square_sum /
                  static_cast<double>(state.output_samples));
        std::cout
            << "RESULT {\"translation\":\""
            << json_escape(state.translation)
            << "\",\"input_samples\":" << state.input_samples
            << ",\"output_samples\":" << state.output_samples
            << ",\"utterances\":" << repeat_count
            << ",\"policies\":" << state.policy_count
            << ",\"writes\":" << state.write_count
            << ",\"load_ms\":" << load_ms
            << ",\"warmup_ms\":" << warmup_ms
            << ",\"wall_ms\":" << wall_ms
            << ",\"rss_before_kib\":" << rss_before_kib
            << ",\"rss_after_kib\":" << rss_after_kib
            << ",\"peak_rss_kib\":" << peak_rss
            << ",\"cuda_free_before_bytes\":"
            << cuda_free_before_bytes
            << ",\"cuda_free_after_bytes\":"
            << cuda_free_after_bytes
            << ",\"cuda_free_delta_bytes\":"
            << cuda_free_delta_bytes
            << ",\"max_inference_ms\":" << state.maximum_inference_ms
            << ",\"max_frontend_ms\":" << state.maximum_frontend_ms
            << ",\"max_encoder_ms\":" << state.maximum_encoder_ms
            << ",\"max_decoder_ms\":" << state.maximum_decoder_ms
            << ",\"max_t2u_ms\":" << state.maximum_t2u_ms
            << ",\"max_vocoder_ms\":" << state.maximum_vocoder_ms
            << ",\"deadline_misses\":"
            << state.inference_deadline_misses
            << ",\"max_capture_queue_ms\":"
            << state.maximum_capture_queue_ms
            << ",\"audio_queue_high_water\":"
            << state.audio_queue_high_water
            << ",\"playback_queue_high_water\":"
            << state.playback_queue_high_water
            << ",\"coalesced_batches\":"
            << state.coalesced_batch_count
            << ",\"max_coalesced_blocks\":"
            << state.maximum_coalesced_blocks
            << ",\"output_rms\":" << rms
            << ",\"output_peak\":" << state.output_peak
            << ",\"output_checksum\":" << state.output_checksum
            << ",\"mode\":\"" << (realtime ? "realtime" : "fast")
            << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
