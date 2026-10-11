from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EncoderExportContractTest(unittest.TestCase):
    def test_wrapper_keeps_official_encoder_modules(self) -> None:
        source = (ROOT / "deploy/export/export_encoder.py").read_text()
        self.assertIn("self.speech_encoder_frontend = model.speech_encoder_frontend", source)
        self.assertIn("self.speech_encoder = model.speech_encoder", source)
        self.assertIn("self.speech_encoder_frontend(fbank, padding_mask)", source)
        self.assertIn("self.speech_encoder(seqs, padding_mask)", source)

    def test_dynamic_export_uses_the_official_naive_sdpa_fallback(self) -> None:
        source = (ROOT / "deploy/export/export_encoder.py").read_text()
        self.assertIn("NaiveSDPA", source)
        self.assertIn("isinstance(child, TorchSDPA)", source)
        self.assertIn("attn_dropout_p=child.attn_dropout_p", source)
        self.assertIn('"official-naive-export"', source)
        self.assertIn('"sdpa_replacement_count"', source)

    def test_padding_is_explicit_dynamic_io(self) -> None:
        source = (ROOT / "deploy/export/export_encoder.py").read_text()
        self.assertIn("PaddingMask(fbank_seq_lens, fbank.shape[1])", source)
        self.assertIn('"fbank_seq_lens"', source)
        self.assertIn('"encoder_seq_lens"', source)
        self.assertIn('"fbank_length"', source)
        self.assertIn('"encoder_length"', source)

    def test_export_validates_every_real_capture_before_onnx(self) -> None:
        source = (ROOT / "deploy/export/export_encoder.py").read_text()
        self.assertIn('for call in manifest["calls"]', source)
        self.assertIn("cosine < 0.9999", source)
        self.assertIn("not lengths_equal", source)
        self.assertIn("validate_and_save_references", source)

    def test_runner_is_offline_and_versioned(self) -> None:
        runner = (ROOT / "deploy/export/run_encoder_export.sh").read_text()
        self.assertIn("STREAMGLASS_ENCODER_CAPTURE_ID", runner)
        self.assertIn("STREAMGLASS_ENCODER_EXPORT_ID", runner)
        self.assertIn("--network none", runner)
        self.assertIn("/exports/$export_id", runner)

    def test_large_onnx_uses_a_filename_for_external_data(self) -> None:
        source = (ROOT / "deploy/export/export_encoder.py").read_text()
        self.assertIn("str(onnx_path)", source)
        self.assertIn("index_onnx_artifacts(onnx_path)", source)

    def test_external_data_is_read_from_the_onnx_metadata(self) -> None:
        source = (ROOT / "deploy/export/onnx_artifacts.py").read_text()
        self.assertIn("load_external_data=False", source)
        self.assertIn('entries.get("location")', source)
        self.assertIn('".." in relative.parts', source)
        self.assertIn("sha256_file(path)", source)

    def test_encoder_build_uses_full_tactics_and_even_profiles(self) -> None:
        source = (ROOT / "deploy/trt/build_encoder_engine.sh").read_text()
        self.assertIn("verify_onnx_artifacts.py", source)
        self.assertIn("encoder_trt.onnx", source)
        self.assertIn("encoder_trt_manifest.json", source)
        self.assertIn("fbank:1x222x80", source)
        self.assertIn("fbank:1x600x80", source)
        self.assertIn("fbank:1x2000x80", source)
        self.assertIn("--builderOptimizationLevel=5", source)
        self.assertIn("workspace:16384", source)
        self.assertIn('export CUDA_CACHE_PATH="$cache_dir/cuda"', source)
        self.assertIn('export TMPDIR="$tmp_dir"', source)
        self.assertNotIn("--tacticSources=-", source)

    def test_trt_normalization_is_math_preserving_and_count_guarded(self) -> None:
        source = (
            ROOT / "deploy/export/normalize_encoder_onnx_for_trt.py"
        ).read_text()
        self.assertIn('where_node.name.endswith("/conv/Where")', source)
        self.assertIn('"/self_attn/sdpa/" in where_node.name', source)
        self.assertIn("TensorProto.FLOAT16", source)
        self.assertIn('einsum_equation(node) != "nhsk,stk->nhst"', source)
        self.assertIn('"shaw_slice_bypass_count"', source)
        self.assertIn('"shaw_dynamic_restoration_count"', source)
        self.assertIn("np.array_equal(frozen_indices, expected_indices)", source)
        self.assertIn('"Range"', source)
        self.assertIn('"full_self_attention_assumption"', source)
        self.assertIn('transformation["casts_inserted"] != 2', source)
        self.assertIn('"mathematical_algorithm_changed": False', source)
        self.assertIn("onnx.checker.check_model", source)

    def test_encoder_parity_covers_all_official_references(self) -> None:
        source = (ROOT / "deploy/trt/encoder_parity.py").read_text()
        self.assertIn('glob("call_*.npz")', source)
        self.assertIn('outputs["encoder_output"]', source)
        self.assertIn('outputs["encoder_seq_lens"]', source)
        self.assertIn('length_metrics["equal"]', source)

    def test_encoder_benchmark_is_scoped_and_budgeted(self) -> None:
        runner = (ROOT / "deploy/trt/benchmark_encoder.sh").read_text()
        self.assertIn("222 396 600 2000", runner)
        self.assertIn("for mode in compute end_to_end", runner)
        summary = (ROOT / "deploy/trt/summarize_encoder_trtexec.py").read_text()
        self.assertIn('"scope": "cumulative_encoder_only"', summary)
        self.assertIn('"frontend_included": False', summary)
        self.assertIn("p99 < args.budget_ms", summary)


if __name__ == "__main__":
    unittest.main()
