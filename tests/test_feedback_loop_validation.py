from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deploy/validate_feedback_loop.py"
DEPLOY_DIR = str(MODULE_PATH.parent)
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)
SPEC = importlib.util.spec_from_file_location("validate_feedback_loop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FeedbackLoopValidationTest(unittest.TestCase):
    def test_utterance_pattern_only_matches_completed_translation(self) -> None:
        match = MODULE.UTTERANCE_PATTERN.match(
            "[realtime] utterance 7: the first one"
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "7")
        self.assertEqual(match.group(2), "the first one")
        self.assertIsNone(
            MODULE.UTTERANCE_PATTERN.match(
                "[realtime] utterance 7 skipped: profile mismatch"
            )
        )


if __name__ == "__main__":
    unittest.main()
