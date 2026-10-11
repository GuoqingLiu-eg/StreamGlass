from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class T2UCaptureContractTest(unittest.TestCase):
    def test_capture_is_observational(self) -> None:
        source = (ROOT / "deploy/export/capture_t2u_reference.py").read_text()
        self.assertIn("ORIGINAL_MODEL_FORWARD", source)
        self.assertIn("ORIGINAL_TOKEN_DECODE", source)
        self.assertNotIn("torch.manual_seed", source)
        self.assertNotIn("random", source)

    def test_capture_records_split_boundaries(self) -> None:
        source = (ROOT / "deploy/export/capture_t2u_reference.py").read_text()
        for name in (
            "text_decoder_output",
            "text_seqs",
            "char_seqs",
            "char_lens",
            "char_features",
            "durations",
            "unit_frontend_output",
            "decoder_output",
            "logits_argmax",
            "policy_unit_ids",
        ):
            self.assertIn(f'"{name}"', source)

    def test_runner_preserves_official_320ms_contract(self) -> None:
        source = (ROOT / "deploy/export/run_t2u_capture.sh").read_text()
        self.assertIn("--source-segment-size 320", source)
        self.assertIn("--network none", source)
        self.assertIn("/opt/streamglass/onnx", source)


if __name__ == "__main__":
    unittest.main()
