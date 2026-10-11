import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NativeRuntimeContractTests(unittest.TestCase):
    def test_core_headers_compile_and_run_without_target_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "core-tests"
            subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Werror",
                    "-I",
                    str(ROOT / "include"),
                    str(ROOT / "tests/native/test_core.cpp"),
                    str(ROOT / "src/tokenizer_assets.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
                cwd=ROOT,
            )
            completed = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
            self.assertIn("core tests passed", completed.stdout)

    def test_deployment_scripts_keep_build_outputs_on_nvme(self) -> None:
        build = (ROOT / "deploy/build_native.sh").read_text()
        provision = (ROOT / "deploy/provision.sh").read_text()
        self.assertIn("/opt/streamglass/tmp/native-build", build)
        self.assertIn("/opt/streamglass/tmp/native-install", build)
        self.assertNotIn("/var/tmp", build)
        self.assertIn("cache tmp", provision)
        self.assertNotIn("usermod", provision)

    def test_inspector_selects_policy_semantic_r3_engines(self) -> None:
        script = (ROOT / "deploy/inspect_engines.sh").read_text()
        self.assertIn("decoder_prefill_mixed.engine", script)
        self.assertIn("decoder_step_mixed.engine", script)
        self.assertIn("t2u_decoder_fp32.engine", script)
        self.assertNotIn("t2u_decoder_mixed.engine", script)
        self.assertNotIn("pchoose_fp32", script)

    def test_full_tokenizer_asset_inspector_is_installed(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text()
        source = (ROOT / "app/tokenizer_inspect.cpp").read_text()
        self.assertIn("streamglass-tokenizer-inspect", cmake)
        self.assertIn("TokenizerAssets", source)
        self.assertIn("vocoder_conditioning", source)

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "tokenizer-inspect"
            subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Werror",
                    "-I",
                    str(ROOT / "include"),
                    str(ROOT / "app/tokenizer_inspect.cpp"),
                    str(ROOT / "src/tokenizer_assets.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
                cwd=ROOT,
            )

    def test_audio_frontend_uses_bundled_official_fbank(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text()
        source = (ROOT / "src/online_fbank.cpp").read_text()
        self.assertIn("kaldi-native-fbank", cmake)
        self.assertIn("knf::FbankComputer", source)
        self.assertIn("knf::ExtractWindow", source)
        self.assertIn("kWindowSamples - kShiftSamples", source)

    def test_native_fbank_parity_uses_real_official_capture(self) -> None:
        runner = (ROOT / "deploy/export/run_native_fbank_parity.sh").read_text()
        verifier = (ROOT / "deploy/export/verify_native_fbank.py").read_text()
        self.assertIn("sample_spa.wav", runner)
        self.assertIn("call_0006.npz", runner)
        self.assertIn("fbank_seq_lens", verifier)
        self.assertIn("expected[:valid_frames]", verifier)
        self.assertIn('report["passed"]', verifier)

    def test_native_encoder_reuses_mapped_fbank_and_real_engine(self) -> None:
        runtime = (ROOT / "src/encoder_runtime.cpp").read_text()
        header = (ROOT / "include/streamglass/encoder_runtime.hpp").read_text()
        runner = (ROOT / "deploy/export/run_native_encoder_parity.sh").read_text()
        verifier = (ROOT / "deploy/export/verify_native_encoder.py").read_text()
        self.assertIn("MappedHostBuffer<float>", header)
        self.assertIn("launch_float_to_half", runtime)
        self.assertIn("enqueueV3", runtime)
        self.assertIn("encoder_cumulative_fp16.engine", runner)
        self.assertIn("sample_spa.wav", runner)
        self.assertIn("encoder_seq_lens", verifier)
        self.assertIn("minimum-cosine", verifier)

    def test_native_decoder_keeps_kv_and_control_reductions_on_cuda(self) -> None:
        header = (ROOT / "include/streamglass/decoder_runtime.hpp").read_text()
        runtime = (ROOT / "src/decoder_runtime.cpp").read_text()
        kernels = (ROOT / "src/cuda_kernels.cu").read_text()
        runner = (ROOT / "deploy/export/run_native_decoder_parity.sh").read_text()
        fixture = (
            ROOT / "deploy/export/prepare_native_decoder_fixture.py"
        ).read_text()
        cmake = (ROOT / "CMakeLists.txt").read_text()

        self.assertIn("std::array<DeviceBuffer<__half>, 2> key_cache_", header)
        self.assertIn("std::array<DeviceBuffer<__half>, 2> value_cache_", header)
        self.assertIn("launch_argmax_half", runtime)
        self.assertIn("launch_min_float", runtime)
        self.assertIn("next_cache_index", runtime)
        self.assertIn("argmax_half_blocks_kernel", kernels)
        self.assertIn("causal_mask_kernel", kernels)
        self.assertIn("decoder_prefill_mixed.engine", runner)
        self.assertIn("decoder_step_mixed.engine", runner)
        self.assertIn("SGDEC001", fixture)
        self.assertIn('call["source_finished"]', fixture)
        self.assertIn("decision_probability_used_by_policy", (
            ROOT / "app/decoder_replay.cpp"
        ).read_text())
        self.assertIn("streamglass-decoder-replay", cmake)

    def test_native_t2u_uses_official_duration_cursor_and_gpu_suffix(self) -> None:
        header = (ROOT / "include/streamglass/t2u_runtime.hpp").read_text()
        runtime = (ROOT / "src/t2u_runtime.cpp").read_text()
        kernels = (ROOT / "src/cuda_kernels.cu").read_text()
        runner = (ROOT / "deploy/export/run_native_t2u_parity.sh").read_text()
        fixture = (ROOT / "deploy/export/prepare_native_t2u_fixture.py").read_text()

        self.assertIn("duration_start_index", header)
        self.assertIn("unit_ids_device", header)
        self.assertIn("build_t2u_inputs", runtime)
        self.assertIn("launch_t2u_policy_slice", runtime)
        self.assertIn("launch_expand_by_duration", runtime)
        self.assertIn("duration_prefix_sum_kernel", kernels)
        self.assertIn("duration_count - 1U", kernels)
        self.assertIn(
            "duration_start_index == duration_count && !source_finished",
            kernels,
        )
        self.assertIn(
            "duration_start_index == input_size.character_count &&",
            runtime,
        )
        self.assertIn("t2u_frontend_fp16.engine", runner)
        self.assertIn("t2u_decoder_fp32.engine", runner)
        self.assertNotIn("t2u_decoder_mixed.engine", runner)
        self.assertIn("policy_unit_ids", fixture)

    def test_native_vocoder_keeps_units_on_gpu_until_playback(self) -> None:
        header = (ROOT / "include/streamglass/vocoder_runtime.hpp").read_text()
        runtime = (ROOT / "src/vocoder_runtime.cpp").read_text()
        kernels = (ROOT / "src/cuda_kernels.cu").read_text()
        runner = (ROOT / "deploy/export/run_native_vocoder_parity.sh").read_text()
        fixture = (
            ROOT / "deploy/export/prepare_native_vocoder_fixture.py"
        ).read_text()
        cmake = (ROOT / "CMakeLists.txt").read_text()

        self.assertIn("const std::int64_t* unit_ids", header)
        self.assertIn("MappedHostBuffer<float> playback_waveform_", header)
        self.assertIn('bind(context, "unit_ids"', runtime)
        self.assertIn("launch_half_to_float", runtime)
        self.assertIn("half_to_float_kernel", kernels)
        self.assertIn("vocoder_fp16.engine", runner)
        self.assertIn("--network none", runner)
        self.assertIn("SGVOC001", fixture)
        self.assertIn("streamglass-vocoder-replay", cmake)

    def test_native_file_pipeline_preserves_official_policy_semantics(self) -> None:
        header = (
            ROOT / "include/streamglass/streaming_text_policy.hpp"
        ).read_text()
        policy = (ROOT / "src/streaming_text_policy.cpp").read_text()
        replay = (ROOT / "app/native_file_replay.cpp").read_text()
        fixture = (
            ROOT / "deploy/export/prepare_native_pipeline_fixture.py"
        ).read_text()
        runner = (ROOT / "deploy/export/run_native_file_pipeline.sh").read_text()
        cmake = (ROOT / "CMakeLists.txt").read_text()

        self.assertIn("kDecisionThreshold = 0.5F", header)
        self.assertIn("kMaximumConsecutiveWrites = 50", header)
        self.assertIn("kMaximumContentTokens = 100", header)
        self.assertIn("probability == 1.0F", policy)
        self.assertIn("tokenizer_.comma_index()", policy)
        self.assertIn("decoder_.decoder_feature_count()", policy)
        self.assertIn("cudaMemcpyDeviceToDevice", replay)
        self.assertIn("kMinimumStartingFbankFrames = 192", replay)
        self.assertIn("kSegmentSamples = 5'120", replay)
        self.assertIn("t2u_decoder_fp32.engine", replay)
        self.assertIn("policy_semantic_passed", replay)
        self.assertIn("strict_unit_parity_passed", replay)
        self.assertIn("STREAMGLASS_REQUIRE_STRICT_UNIT_PARITY", replay)
        self.assertIn("SGPIP001", fixture)
        self.assertIn("target_indices_after", fixture)
        self.assertIn("--network none", runner)
        self.assertIn("/opt/streamglass/tmp/native-file-pipeline", runner)
        self.assertIn("streamglass-native-file-replay", cmake)

    def test_t2u_precision_probe_keeps_production_engine_immutable(self) -> None:
        build = (ROOT / "deploy/trt/build_t2u_engines.sh").read_text()
        replay = (ROOT / "app/native_file_replay.cpp").read_text()

        self.assertIn("STREAMGLASS_T2U_BUILD_MODE", build)
        self.assertIn("strong-frontend-probe", build)
        self.assertIn("t2u_frontend_strongly_typed", build)
        self.assertIn("--stronglyTyped", build)
        self.assertIn("STREAMGLASS_T2U_FRONTEND_ENGINE_NAME", replay)
        self.assertIn('"t2u_frontend_fp16.engine"', replay)

    def test_encoder_precision_probe_keeps_production_engine_immutable(self) -> None:
        build = (ROOT / "deploy/trt/build_encoder_engine.sh").read_text()
        replay = (ROOT / "app/native_file_replay.cpp").read_text()

        self.assertIn("STREAMGLASS_ENCODER_BUILD_MODE", build)
        self.assertIn("strong-probe", build)
        self.assertIn("encoder_cumulative_strongly_typed", build)
        self.assertIn("--stronglyTyped", build)
        self.assertIn("--builderOptimizationLevel=5", build)
        self.assertIn("STREAMGLASS_ENCODER_ENGINE_NAME", replay)
        self.assertIn('"encoder_cumulative_fp16.engine"', replay)

    def test_threaded_runtime_uses_preallocated_spsc_zero_copy_path(self) -> None:
        header = (ROOT / "include/streamglass/streaming_pipeline.hpp").read_text()
        pipeline = (ROOT / "src/streaming_pipeline.cpp").read_text()
        runtime = (ROOT / "app/native_threaded_file.cpp").read_text()
        vocoder = (ROOT / "src/vocoder_runtime.cpp").read_text()
        cmake = (ROOT / "CMakeLists.txt").read_text()

        self.assertIn("process_audio", header)
        self.assertIn("synthesize", header)
        self.assertIn("cudaMemcpyDeviceToDevice", pipeline)
        self.assertIn("SpscBufferPool<AudioBlock", runtime)
        self.assertIn("SpscRing<streamglass::BufferHandle", runtime)
        self.assertIn("drwav_read_pcm_frames_f32", runtime)
        self.assertIn("process_audio_batch", pipeline)
        self.assertIn("kMaximumCoalescedBlocks = 1", runtime)
        self.assertIn("policy calls must never be coalesced", runtime)
        self.assertIn("warmup_pipeline", runtime)
        self.assertIn("STREAMGLASS_SKIP_WARMUP", runtime)
        self.assertIn("MappedHostBuffer<float>", runtime)
        self.assertIn("materialize_for_playback(result, destination)", pipeline)
        self.assertIn("destination.device_data()", vocoder)
        self.assertIn("streamglass-native-threaded-file", cmake)
        self.assertIn("STREAMGLASS_DECODER_PREFILL_ENGINE_NAME", runtime)
        self.assertIn("STREAMGLASS_DECODER_STEP_ENGINE_NAME", runtime)
        self.assertIn("cudaMemGetInfo", runtime)
        self.assertIn("VmRSS:", runtime)
        self.assertIn("cuda_free_delta_bytes", runtime)

        validator = (
            ROOT / "deploy/validate_native_threaded_runtime.py"
        ).read_text()
        self.assertIn('default=128.0', validator)
        self.assertIn('"device-global-advisory"', validator)

    def test_tensorrt_uses_one_process_logger(self) -> None:
        engine_header = (ROOT / "include/streamglass/trt_engine.hpp").read_text()
        engine_source = (ROOT / "src/trt_engine.cpp").read_text()
        runtime_headers = "\n".join(
            (ROOT / "include/streamglass" / name).read_text()
            for name in (
                "encoder_runtime.hpp",
                "decoder_runtime.hpp",
                "t2u_runtime.hpp",
                "vocoder_runtime.hpp",
            )
        )

        self.assertIn("process_logger()", engine_source)
        self.assertNotIn("TensorRtLogger& logger", engine_header)
        self.assertNotIn("TensorRtLogger logger_", runtime_headers)

    def test_realtime_pipe_does_not_deadlock_or_persist_silence(self) -> None:
        runtime = (ROOT / "app/native_realtime.cpp").read_text()
        runner = (ROOT / "deploy/run_realtime_glasses.sh").read_text()
        aec_args = (
            ROOT / "config/pipewire/streamglass-echo-cancel.args"
        ).read_text()

        self.assertIn("std::cerr.tie(nullptr)", runtime)
        self.assertIn("if (sample_count == 0)", runtime)
        self.assertIn("std::vector<float> buffer(512)", runtime)
        self.assertIn("STREAMGLASS_PLAYBACK_DRAIN_MS", runtime)
        self.assertIn("STREAMGLASS_VAD_START_RMS", runtime)
        self.assertIn("STREAMGLASS_VAD_CONTINUE_RMS", runtime)
        self.assertIn("STREAMGLASS_VAD_START_MS", runtime)
        self.assertIn("STREAMGLASS_VAD_PREROLL_MS", runtime)
        self.assertIn("STREAMGLASS_VAD_SILENCE_MS", runtime)
        self.assertIn("STREAMGLASS_UTTERANCE_HARD_CAP_MS", runtime)
        self.assertIn('"10000"', runtime)
        self.assertIn("extend_playback_end(playback_end_ns", runtime)
        self.assertIn("device_playback_pending", runtime)
        self.assertIn("STREAMGLASS_PLAYBACK_ENQUEUE_TIMEOUT_MS", runtime)
        self.assertIn("playback.push_wait(", runtime)
        self.assertNotIn("write_index++", runtime)
        self.assertIn("++write_index", runtime)
        self.assertIn("STREAMGLASS_PLAYBACK_TAP", runner)
        self.assertIn("STREAMGLASS_AUDIO_RETRY_SECONDS", runner)
        self.assertIn("streamglass-capture-$$.fifo", runner)
        self.assertIn("streamglass-playback-$$.fifo", runner)
        self.assertIn('exec 7<>"$AUDIO_CAPTURE_FIFO"', runner)
        self.assertIn('exec 8<>"$AUDIO_PLAYBACK_FIFO"', runner)
        self.assertIn('MAIN_BASHPID="$BASHPID"', runner)
        self.assertIn("capture endpoint unavailable", runner)
        self.assertIn("playback endpoint unavailable", runner)
        self.assertIn("TensorRT", runner)
        self.assertNotIn("tee /tmp/rt_tap.raw", runner)
        self.assertNotIn("/tmp/probe_cap.raw", runner)
        self.assertIn('AEC_ENABLED="${STREAMGLASS_AEC:-1}"', runner)
        self.assertIn("streamglass_aec_source", runner)
        self.assertIn("streamglass_aec_sink", runner)
        self.assertIn("HALF_DUPLEX_DEFAULT=0", runner)
        self.assertIn("mkfifo", runner)
        self.assertIn('pw-cli <"$AEC_FIFO"', runner)
        self.assertIn('pw-cli <"$AEC_FIFO" >/dev/null', runner)
        self.assertIn('2>"$AEC_LOG"', runner)
        self.assertIn("tr '\\n\\r\\t' '   '", runner)
        self.assertIn('"$aec_args" >&9', runner)
        self.assertIn("RAW_SRC_NAME", runner)
        self.assertIn("libpipewire-module-echo-cancel", runner)
        self.assertIn("aec/libspa-aec-webrtc", aec_args)
        self.assertIn("audio.rate = 16000", aec_args)
        self.assertIn("buffer.play_delay = 180/1000", aec_args)
        self.assertIn("webrtc.delay_agnostic = true", aec_args)
        self.assertIn("webrtc.extended_filter = true", aec_args)
        self.assertIn("webrtc.gain_control = false", aec_args)
        self.assertEqual(aec_args.count("node.autoconnect = true"), 2)
        self.assertEqual(aec_args.count("node.dont-reconnect = false"), 2)
        self.assertEqual(aec_args.count("node.dont-fallback = true"), 2)
        self.assertGreaterEqual(aec_args.count("node.pause-on-idle = false"), 4)
        self.assertEqual(aec_args.count("node.always-process = true"), 2)
        self.assertGreaterEqual(
            aec_args.count("session.suspend-timeout-seconds = 0"), 4
        )

    def test_boot_services_restore_bluetooth_and_realtime_audio(self) -> None:
        realtime_service = (
            ROOT / "deploy/systemd/streamglass-realtime.service"
        ).read_text()
        web_service = (
            ROOT / "deploy/systemd/streamglass-web.service"
        ).read_text()
        bluetooth_service = (
            ROOT / "deploy/systemd/streamglass-bluetooth.service"
        ).read_text()
        boot_installer = (ROOT / "deploy/install_boot_services.sh").read_text()
        connector = (ROOT / "deploy/connect_barrot_glasses.sh").read_text()
        controller = (ROOT / "deploy/realtime_service.sh").read_text()
        audio_wait = (ROOT / "deploy/wait_for_glasses_audio.sh").read_text()
        native_realtime = (ROOT / "app/native_realtime.cpp").read_text()
        hfp_monitor = (ROOT / "src/hfp_button_monitor.cpp").read_text()
        runner = (ROOT / "deploy/run_realtime_glasses.sh").read_text()
        wireplumber = (
            ROOT / "config/wireplumber/51-streamglass-bluez.lua"
        ).read_text()

        self.assertIn("modprobe --ignore-install btusb", connector)
        self.assertIn("STREAMGLASS_BT_CONTROLLER", connector)
        self.assertIn("rfkill unblock bluetooth", connector)
        self.assertIn('loginctl enable-linger "$TARGET_USER"', connector)
        self.assertIn("user@${target_uid}.service", connector)
        self.assertIn('setpriv --reuid="$target_uid"', connector)
        self.assertIn("pipewire.service pipewire-pulse.service", connector)
        self.assertIn("agent NoInputNoOutput", connector)
        self.assertIn('STREAMGLASS_BT_MONITOR:-0', connector)
        self.assertIn("endpoint lost", connector)
        self.assertNotIn("disconnect %s", connector)
        self.assertIn("pw-dump", connector)
        self.assertIn("JSONDecoder().raw_decode", connector)
        self.assertIn("if not isinstance(obj, dict)", connector)
        self.assertIn("device_connected", connector)
        self.assertIn("STREAMGLASS_BT_DIRECT_CONNECT_ATTEMPTS", connector)
        self.assertIn("STREAMGLASS_BT_CONNECT_TIMEOUT_SECONDS", connector)
        self.assertIn("STREAMGLASS_BT_BREDR_SCAN_SECONDS", connector)
        self.assertIn("scan bredr", connector)
        self.assertIn("scan off", connector)
        self.assertIn("post-scan", connector)
        self.assertIn("yielding to systemd retry", connector)
        self.assertNotIn("until device_connected && audio_endpoint_available", connector)
        self.assertIn("bluez_input.${device_token}.0", connector)
        self.assertIn("bluez_output.${device_token}.1", connector)
        self.assertIn("l2ping", connector)
        self.assertIn("runtime_state", connector)
        self.assertIn("STREAMGLASS_START_PAUSED", connector)
        self.assertIn("STREAMGLASS_RESTORE_TARGET_LANGUAGE", connector)
        self.assertIn("restart streamglass-realtime.service", connector)
        self.assertIn("resident_model_ready", connector)
        self.assertIn("pipewire_graph_ready", connector)
        self.assertIn("repair_aec_links", connector)
        self.assertIn("wait_for_hot_rebind", connector)
        self.assertIn("pw-link -L", connector)
        self.assertIn("keeping engines resident", connector)
        self.assertIn("hot rebound complete", connector)
        self.assertIn("resident model session preserved", connector)
        self.assertIn("endpoint restored and realtime session rebound", connector)
        self.assertNotIn("hci0", connector)
        self.assertNotIn("hci1", connector)

        self.assertIn("Wants=pipewire.service", realtime_service)
        self.assertIn("Wants=pipewire.service pipewire-pulse.service", realtime_service)
        self.assertNotIn("ExecStartPre", realtime_service)
        self.assertNotIn("TimeoutStartSec=infinity", realtime_service)
        self.assertIn("KillMode=control-group", realtime_service)
        self.assertIn(
            "StandardOutput=append:/dev/shm/streamglass", realtime_service
        )
        self.assertIn("WantedBy=default.target", realtime_service)
        self.assertIn("STREAMGLASS_WEB_HOST=0.0.0.0", web_service)
        self.assertIn("STREAMGLASS_WEB_PORT=8080", web_service)
        self.assertIn("pw-dump", audio_wait)
        self.assertIn("JSONDecoder().raw_decode", audio_wait)
        self.assertIn("if not isinstance(obj, dict)", audio_wait)
        self.assertIn("bluez_input.${device_token}.0", audio_wait)
        self.assertIn("bluez_output.${device_token}.1", audio_wait)
        self.assertIn("STREAMGLASS_AUDIO_WAIT_TIMEOUT_SECONDS", audio_wait)
        self.assertIn("timeout \"${PW_DUMP_TIMEOUT_SECONDS}s\" pw-dump", audio_wait)
        self.assertIn("STREAMGLASS_VAD_START_RMS=0.030", controller)
        self.assertIn("STREAMGLASS_VAD_CONTINUE_RMS=0.016", controller)
        self.assertIn("STREAMGLASS_VAD_START_MS=192", controller)
        self.assertIn("STREAMGLASS_VAD_PREROLL_MS=192", controller)
        self.assertIn("STREAMGLASS_VAD_SILENCE_MS=544", controller)
        self.assertIn("STREAMGLASS_UTTERANCE_HARD_CAP_MS=12800", controller)
        self.assertIn("systemctl --user enable", controller)
        self.assertIn("start|stop|restart|enable|disable|status|logs", controller)
        self.assertIn("STREAMGLASS_BT_MONITOR=1", bluetooth_service)
        self.assertIn("Restart=on-failure", bluetooth_service)
        self.assertIn("RestartSec=15", bluetooth_service)
        self.assertIn("WantedBy=multi-user.target", bluetooth_service)
        self.assertIn("streamglass-bluetooth.service", boot_installer)
        self.assertIn('loginctl enable-linger "$TARGET_USER"', boot_installer)
        self.assertIn('realtime_service.sh" install', boot_installer)
        self.assertIn(
            "systemctl --user disable --now streamglass-realtime.service",
            boot_installer,
        )
        self.assertIn('["with-logind"] = false', wireplumber)
        self.assertIn('["bluez5.enable-msbc"] = true', wireplumber)
        self.assertIn('["bluez5.auto-connect"]', wireplumber)
        self.assertIn('["node.pause-on-idle"] = false', wireplumber)
        self.assertIn(
            '["session.suspend-timeout-seconds"] = 0', wireplumber
        )
        self.assertIn("STREAMGLASS_START_PAUSED", native_realtime)
        self.assertIn("STREAMGLASS_RESTORE_TARGET_LANGUAGE", runner)
        self.assertIn("open_monitor_socket", hfp_monitor)
        self.assertIn("wait_before_reopen", hfp_monitor)
        self.assertIn("POLLERR | POLLHUP | POLLNVAL", hfp_monitor)
        self.assertIn("transport changed", hfp_monitor)


if __name__ == "__main__":
    unittest.main()
