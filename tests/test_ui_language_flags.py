import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LanguageShortcutFlagTests(unittest.TestCase):
    def test_each_globe_shortcut_has_the_expected_flag_and_code(self) -> None:
        html = (ROOT / "ui-prototype" / "index.html").read_text(encoding="utf-8")
        expected = {
            "cmn": ("🇨🇳", "CMN"),
            "eng": ("🇬🇧", "ENG"),
            "fra": ("🇫🇷", "FRA"),
            "deu": ("🇩🇪", "DEU"),
            "spa": ("🇪🇸", "SPA"),
            "ita": ("🇮🇹", "ITA"),
            "jpn": ("🇯🇵", "JPN"),
            "kor": ("🇰🇷", "KOR"),
            "rus": ("🇷🇺", "RUS"),
            "por": ("🇵🇹", "POR"),
            "arb": ("🇸🇦", "ARB"),
            "vie": ("🇻🇳", "VIE"),
        }

        for language, (flag, code) in expected.items():
            button = re.search(
                rf'<button class="[^"]*globe-target[^"]*"[^>]*data-code="{language}"[^>]*>(.*?)</button>',
                html,
                re.DOTALL,
            )
            self.assertIsNotNone(button, language)
            self.assertIn(f'class="globe-target-flag" aria-hidden="true">{flag}</span>', button.group(1))
            self.assertIn(f'class="globe-target-code">{code}</span>', button.group(1))

    def test_desktop_shortcuts_use_a_balanced_grid_clear_of_project_leads(self) -> None:
        css = (ROOT / "ui-prototype" / "styles.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            r"\.experience-stage \.globe-targets \{[^}]*"
            r"grid-template-columns: repeat\(6, minmax\(0, 1fr\)\);",
        )
        self.assertRegex(css, r"\.project-leads \{[^}]*bottom: 101px;")


if __name__ == "__main__":
    unittest.main()
