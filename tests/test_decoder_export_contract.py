from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DecoderExportContractTest(unittest.TestCase):
    def test_export_keeps_official_modules_and_explicit_cache(self) -> None:
        source = (ROOT / "deploy/export/export_decoder.py").read_text()
        self.assertIn("load_monotonic_decoder_model", source)
        self.assertIn("layer._forward_encoder_decoder_attn", source)
        self.assertIn("layer._forward_ffn", source)
        self.assertIn('"mathematical_algorithm_changed": False', source)
        self.assertIn("self_k_cache", source)
        self.assertIn("self_v_cache", source)
        self.assertIn("position", source)
        self.assertIn("causal_mask", source)

    def test_export_validates_every_official_call_before_onnx(self) -> None:
        source = (ROOT / "deploy/export/export_decoder.py").read_text()
        self.assertLess(source.index("validate_references("), source.index("export_onnx("))
        self.assertIn("selected_index != expected_index", source)
        self.assertIn("cache_k_cosine < 0.999", source)
        self.assertIn("probability_error > 0.02", source)

    def test_prefill_and_step_axes_are_dynamic(self) -> None:
        source = (ROOT / "deploy/export/export_decoder.py").read_text()
        self.assertIn('"target_length"', source)
        self.assertIn('"encoder_length"', source)
        self.assertIn('"cache_length"', source)
        self.assertIn('"cache_length_out"', source)
        self.assertIn('"causal_mask": {0: "target_length", 1: "target_length"}', source)

    def test_large_external_weights_are_isolated_per_graph(self) -> None:
        source = (ROOT / "deploy/export/export_decoder.py").read_text()
        self.assertIn('prefill_dir = output_dir / "prefill"', source)
        self.assertIn('step_dir = output_dir / "step"', source)
        self.assertIn("prefill_path.relative_to(args.output_dir)", source)
        self.assertIn("step_path.relative_to(args.output_dir)", source)

    def test_runner_is_versioned_offline_and_nvme_scoped(self) -> None:
        source = (ROOT / "deploy/export/run_decoder_export.sh").read_text()
        self.assertIn("--network none", source)
        self.assertIn("/opt/streamglass/onnx", source)
        self.assertIn("STREAMGLASS_DECODER_CAPTURE_ID", source)
        self.assertIn("STREAMGLASS_DECODER_EXPORT_ID", source)

    def test_tensorrt_parser_checks_all_bindings_before_build(self) -> None:
        parser = (ROOT / "deploy/trt/parse_decoder_onnx.py").read_text()
        runner = (ROOT / "deploy/trt/run_decoder_parse.sh").read_text()
        self.assertIn("trt.OnnxParser", parser)
        self.assertIn("parser.parse_from_file", parser)
        self.assertIn("network.num_inputs", parser)
        self.assertIn("network.num_outputs", parser)
        self.assertIn("/opt/streamglass/golden/trt/decoder_parse.json", runner)

    def test_engine_build_is_full_tactic_strongly_typed_and_dynamic(self) -> None:
        source = (ROOT / "deploy/trt/build_decoder_engines.sh").read_text()
        self.assertIn("--stronglyTyped", source)
        self.assertIn("--builderOptimizationLevel=5", source)
        self.assertIn("--memPoolSize=workspace:16384", source)
        self.assertNotIn("--tacticSources", source)
        self.assertIn("target_input:1x512", source)
        self.assertIn("causal_mask:512x512", source)
        self.assertIn("self_k_cache:24x1x16x512x64", source)
        self.assertIn("verify_decoder_export.py", source)
        self.assertIn("STREAMGLASS_DECODER_ENGINE_TAG", source)
        self.assertIn("STREAMGLASS_DECODER_BUILD_COMPONENTS", source)

    def test_decoder_artifact_verifier_requires_official_parity(self) -> None:
        source = (ROOT / "deploy/export/verify_decoder_export.py").read_text()
        self.assertIn('manifest.get("mathematical_algorithm_changed") is not False', source)
        self.assertIn('manifest.get("validated_call_count", 0)) < 20', source)
        self.assertIn("sha256_file(path)", source)

    def test_tensorrt_parity_covers_prefill_step_tokens_and_decisions(self) -> None:
        source = (ROOT / "deploy/trt/decoder_parity.py").read_text()
        self.assertIn('"causal_mask": causal_mask(target_length)', source)
        self.assertIn('"self_k_cache": input_k', source)
        self.assertIn('selected_index == int(call["selected_index"])', source)
        self.assertIn("decision_error <= args.max_decision_error", source)
        self.assertIn('decision_probability_used = not bool(call["source_finished"])', source)
        self.assertIn("decision_probability_gate_passed", source)
        self.assertIn('cache_k["cosine"] >= args.min_cosine', source)
        self.assertIn('cache_v["cosine"] >= args.min_cosine', source)
        self.assertIn("decision_layer_metrics", source)

    def test_benchmark_covers_real_and_profile_limit_shapes(self) -> None:
        source = (ROOT / "deploy/trt/benchmark_decoder.sh").read_text()
        summary = (ROOT / "deploy/trt/summarize_decoder_trtexec.py").read_text()
        self.assertIn("t0002_s0014", source)
        self.assertIn("t0512_s0256", source)
        self.assertIn("c0002_s0014", source)
        self.assertIn("c0512_s0256", source)
        self.assertIn("--noDataTransfers", source)
        self.assertIn("--useCudaGraph", source)
        self.assertIn('"zero_copy_runtime_target": True', summary)
        self.assertIn('mode != "compute" or p99 < args.budget_ms', summary)


if __name__ == "__main__":
    unittest.main()
