from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy/validate_full_duplex_aec.py"
SPEC = importlib.util.spec_from_file_location("validate_full_duplex_aec", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FullDuplexAecValidationTest(unittest.TestCase):
    def test_add_play_delay_injects_module_property(self) -> None:
        args = "{ audio.rate = 16000 }"
        self.assertEqual(MODULE.add_play_delay(args, None), args)
        self.assertIn(
            "buffer.play_delay = 80/1000",
            MODULE.add_play_delay(args, 80),
        )
        existing = "{ buffer.play_delay = 180/1000 audio.rate = 16000 }"
        updated = MODULE.add_play_delay(existing, 140)
        self.assertEqual(updated.count("buffer.play_delay"), 1)
        self.assertEqual(MODULE.configured_play_delay_ms(updated), 140)

    def test_repeat_has_exact_duration_and_finite_samples(self) -> None:
        source = np.linspace(-0.2, 0.2, 2_000, dtype=np.float32)
        output = MODULE.repeat_with_crossfade(source, 1.5, 2.0)
        self.assertEqual(output.shape, (24_000,))
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertLessEqual(float(np.max(np.abs(output))), 0.98)

    def test_evaluate_accepts_suppressed_echo_below_vad(self) -> None:
        seconds = 4.0
        settle = 1.0
        total = int((seconds + settle + 1.0) * MODULE.SAMPLE_RATE)
        raw = np.full(total, 0.08, dtype=np.float32)
        aec = np.full(total, 0.006, dtype=np.float32)
        result = MODULE.evaluate(raw, aec, settle, seconds, 0.02)
        self.assertTrue(result["passed"])
        self.assertLess(result["attenuation_db"], -6.0)

    def test_evaluate_rejects_residual_above_vad(self) -> None:
        seconds = 4.0
        settle = 1.0
        total = int((seconds + settle + 1.0) * MODULE.SAMPLE_RATE)
        raw = np.full(total, 0.08, dtype=np.float32)
        aec = np.full(total, 0.03, dtype=np.float32)
        result = MODULE.evaluate(raw, aec, settle, seconds, 0.02)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["acceptance"]["aec_window_rms_p95_below_vad"]
        )

    def test_echo_delay_estimator_finds_known_offset(self) -> None:
        rng = np.random.default_rng(42)
        playback = rng.normal(0.0, 0.1, MODULE.SAMPLE_RATE).astype(np.float32)
        start_seconds = 0.5
        delay_ms = 137.0
        start = int(start_seconds * MODULE.SAMPLE_RATE)
        delay = int(delay_ms * MODULE.SAMPLE_RATE / 1000.0)
        capture = np.zeros(start + delay + playback.size + 1_000, dtype=np.float32)
        capture[start + delay : start + delay + playback.size] = playback
        estimated, correlation = MODULE.estimate_echo_delay_ms(
            capture, playback, start_seconds
        )
        self.assertAlmostEqual(estimated, delay_ms, delta=0.1)
        self.assertGreater(correlation, 0.99)


if __name__ == "__main__":
    unittest.main()
