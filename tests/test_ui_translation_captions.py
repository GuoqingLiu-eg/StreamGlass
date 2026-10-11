import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TranslationCaptionContractTests(unittest.TestCase):
    def test_ui_consumes_runtime_translation_status_without_disk_io(self) -> None:
        html = (ROOT / "ui-prototype" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "ui-prototype" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="translationCaptions"', html)
        self.assertIn('id="captionCurrent"', html)
        self.assertIn('id="captionPrevious"', html)
        self.assertIn('id="captionFeed"', html)
        self.assertIn('id="captionPhase"', html)
        self.assertIn("status.latestTranslation", script)
        self.assertIn("status.processedUtterances", script)
        self.assertIn("status.translationRevision", script)
        self.assertIn("status.translationUtterance", script)
        self.assertIn("status.translationFinal", script)
        self.assertIn("showTranslationCaption", script)
        self.assertNotIn("localStorage", script)

    def test_runtime_streams_real_partial_updates_over_sse(self) -> None:
        script = (ROOT / "ui-prototype" / "app.js").read_text(encoding="utf-8")
        server = (ROOT / "ui-prototype" / "server.py").read_text(
            encoding="utf-8"
        )
        runtime = (ROOT / "app" / "native_realtime.cpp").read_text(
            encoding="utf-8"
        )

        self.assertIn('new EventSource("/api/events")', script)
        self.assertIn('event: runtime', server)
        self.assertIn('interval: float = 0.05', server)
        self.assertIn('TextPolicyAction::write', runtime)
        self.assertIn('false,\n                        processed_utterances', runtime)

    def test_language_change_suppresses_stale_translation(self) -> None:
        script = (ROOT / "ui-prototype" / "app.js").read_text(encoding="utf-8")

        self.assertIn("captionSuppressThrough", script)
        self.assertIn(
            "clearTranslationCaptions(code, lastTranslationRevision)",
            script,
        )


if __name__ == "__main__":
    unittest.main()
