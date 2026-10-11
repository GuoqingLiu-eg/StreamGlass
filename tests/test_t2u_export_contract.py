from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class T2UExportContractTest(unittest.TestCase):
    def test_split_keeps_official_modules(self) -> None:
        source = (ROOT / "deploy/export/export_t2u.py").read_text()
        self.assertIn("self.encoder = model.encoder", source)
        self.assertIn("self.decoder = copy.deepcopy(model.decoder)", source)
        self.assertIn("self.final_proj = copy.deepcopy(model.final_proj)", source)
        self.assertIn("frontend.variance_adaptor.duration_predictor", source)

    def test_only_data_dependent_hard_upsampling_is_host_side(self) -> None:
        source = (ROOT / "deploy/export/export_t2u.py").read_text()
        self.assertIn("repeat_interleave", source)
        self.assertIn("expanded_char_features", source)
        self.assertIn('"host_stage"', source)

    def test_export_requires_real_official_capture(self) -> None:
        runner = (ROOT / "deploy/export/run_t2u_export.sh").read_text()
        self.assertIn("STREAMGLASS_T2U_CAPTURE_ID", runner)
        self.assertIn("manifest.json", runner)
        self.assertIn("--network none", runner)

    def test_hidden_state_tolerance_keeps_discrete_outputs_exact(self) -> None:
        source = (ROOT / "deploy/export/export_t2u.py").read_text()
        self.assertIn("char_cosine < 0.9999", source)
        self.assertIn("not duration_equal", source)
        self.assertIn("not unit_equal", source)

    def test_tensorrt_profiles_cover_realtime_and_long_utterances(self) -> None:
        source = (ROOT / "deploy/trt/build_t2u_engines.sh").read_text()
        self.assertIn("text_decoder_output:1x128x1024", source)
        self.assertIn("char_seqs:1x512", source)
        self.assertIn("expanded_char_features:1x1024x1024", source)
        self.assertIn("--builderOptimizationLevel=5", source)
        self.assertIn("--stronglyTyped", source)

    def test_decoder_output_head_is_fp32(self) -> None:
        source = (ROOT / "deploy/export/export_t2u.py").read_text()
        self.assertIn("decoder_output.float()", source)
        self.assertIn("self.final_proj.weight.float()", source)
        self.assertIn('"decoder_precision"', source)

    def test_accuracy_candidate_decouples_and_promotes_the_decoder(self) -> None:
        source = (ROOT / "deploy/export/export_t2u.py").read_text()
        self.assertIn("copy.deepcopy(model.decoder)", source)
        self.assertIn('args.decoder_precision == "fp32"', source)
        self.assertIn("decoder = decoder.float()", source)

    def test_tensorrt_parity_checks_the_full_chain(self) -> None:
        source = (ROOT / "deploy/trt/t2u_parity.py").read_text()
        self.assertIn("decoder_isolated_unit_ids", source)
        self.assertIn("chained_unit_ids", source)
        self.assertIn("duration_result[\"equal\"]", source)

    def test_benchmark_covers_real_and_profile_shapes(self) -> None:
        source = (ROOT / "deploy/trt/benchmark_t2u.sh").read_text()
        self.assertIn("text_decoder_output:1x7x1024", source)
        self.assertIn("text_decoder_output:1x14x1024", source)
        self.assertIn("text_decoder_output:1x32x1024", source)
        self.assertIn("expanded_char_features:1x57x1024", source)
        self.assertIn("expanded_char_features:1x131x1024", source)
        self.assertIn("expanded_char_features:1x256x1024", source)
        self.assertIn("for mode in compute end_to_end", source)
        self.assertIn("--useCudaGraph", source)
        self.assertIn("--useSpinWait", source)
        self.assertIn("summarize_t2u_trtexec.py", source)

    def test_benchmark_gate_is_scoped_to_t2u(self) -> None:
        source = (ROOT / "deploy/trt/summarize_t2u_trtexec.py").read_text()
        self.assertIn('"scope": "t2u_frontend_plus_decoder_only"', source)
        self.assertIn('"host_upsample_included": False', source)
        self.assertIn('"summed_p99_ms"', source)
        self.assertIn("p99_sum < args.budget_ms", source)


if __name__ == "__main__":
    unittest.main()
