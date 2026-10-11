from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DecoderCaptureContractTest(unittest.TestCase):
    def test_capture_is_observational(self) -> None:
        source = (ROOT / "deploy/export/capture_decoder_reference.py").read_text()
        self.assertIn("ORIGINAL_DECODE(", source)
        self.assertIn("ORIGINAL_PROJECT(self, decoder_output)", source)
        self.assertIn("ORIGINAL_RUN_DECODER(", source)
        self.assertIn("ORIGINAL_POLICY(self, states)", source)
        self.assertNotIn("return ReadAction", source)
        self.assertNotIn("return WriteAction", source)

    def test_capture_records_prefill_step_and_policy_boundaries(self) -> None:
        source = (ROOT / "deploy/export/capture_decoder_reference.py").read_text()
        for name in (
            "target_input",
            "encoder_output",
            "decoder_output",
            "p_choose",
            "logits",
            "selected_index",
            "decision_probability",
            'f"self_k_{suffix}',
            'f"self_v_{suffix}',
            "state_step_before_decode",
            "target_indices_before",
            "target_indices_after",
        ):
            self.assertIn(name, source)

    def test_runner_preserves_official_320ms_contract(self) -> None:
        runner = (ROOT / "deploy/export/run_decoder_capture.sh").read_text()
        self.assertIn("--source-segment-size 320", runner)
        self.assertIn("--network none", runner)
        self.assertIn("--dtype fp16", runner)
        self.assertIn("--no-strip-silence", runner)


if __name__ == "__main__":
    unittest.main()
