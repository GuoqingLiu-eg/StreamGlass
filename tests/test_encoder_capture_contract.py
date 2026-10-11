from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EncoderCaptureContractTest(unittest.TestCase):
    def test_capture_is_observational(self) -> None:
        source = (ROOT / "deploy/export/capture_encoder_reference.py").read_text()
        self.assertIn("ORIGINAL_ENCODE_SPEECH(self, seqs, padding_mask)", source)
        self.assertIn("ORIGINAL_POLICY(self, states)", source)
        self.assertNotIn("copy_", source)
        self.assertNotIn("return WriteAction", source)

    def test_capture_records_reference_boundaries(self) -> None:
        source = (ROOT / "deploy/export/capture_encoder_reference.py").read_text()
        self.assertIn('"fbank"', source)
        self.assertIn('"fbank_seq_lens"', source)
        self.assertIn('"encoder_output"', source)
        self.assertIn('"encoder_seq_lens"', source)
        self.assertIn('"source_finished"', source)
        self.assertIn('"action"', source)

    def test_runner_preserves_official_320ms_contract(self) -> None:
        runner = (ROOT / "deploy/export/run_encoder_capture.sh").read_text()
        self.assertIn("--source-segment-size 320", runner)
        self.assertIn("--network none", runner)
        self.assertIn("--dtype fp16", runner)
        self.assertIn("--no-strip-silence", runner)


if __name__ == "__main__":
    unittest.main()
