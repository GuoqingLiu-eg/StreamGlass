import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OfflineAssetTest(unittest.TestCase):
    def test_nar_t2u_tokenizers_are_local(self) -> None:
        card_path = PROJECT_ROOT / "config/assets/seamlessM4T_v2_large.yaml"
        card = json.loads(card_path.read_text())

        self.assertEqual(card["name"], "seamlessM4T_v2_large@user")
        self.assertEqual(
            card["char_tokenizer"], "file:///models/spm_char_lang38_tc.model"
        )
        self.assertEqual(card["tokenizer"], "file:///models/tokenizer.model")


if __name__ == "__main__":
    unittest.main()
