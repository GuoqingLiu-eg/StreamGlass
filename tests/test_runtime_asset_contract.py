import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeAssetContractTests(unittest.TestCase):
    def test_export_uses_official_tokenizers_and_vocoder_map(self) -> None:
        path = ROOT / "deploy/export/export_runtime_assets.py"
        source = path.read_text()
        tree = ast.parse(source)
        self.assertIn("load_unity_text_tokenizer", source)
        self.assertIn("load_unity_char_tokenizer", source)
        self.assertIn("load_vocoder_model", source)
        self.assertIn("prefix_indices", source)
        self.assertIn("lang_spkr_idx_map", source)
        self.assertIn("excluded_unit_languages_without_vocoder", source)
        self.assertIn("language not in vocoder_languages", source)
        self.assertNotIn("torch.onnx.export", source)
        self.assertGreater(len(tree.body), 0)

    def test_export_parity_checks_real_t2u_captures(self) -> None:
        source = (ROOT / "deploy/export/export_runtime_assets.py").read_text()
        self.assertIn('data["text_seqs"]', source)
        self.assertIn('data["char_seqs"]', source)
        self.assertIn('data["char_lens"]', source)
        self.assertIn("np.array_equal", source)

    def test_runner_is_offline_and_nvme_scoped(self) -> None:
        source = (
            ROOT / "deploy/export/run_runtime_assets_export.sh"
        ).read_text()
        self.assertIn("--network none", source)
        self.assertIn("/opt/streamglass/tmp/runtime-assets", source)
        self.assertIn("/models:ro", source)
        self.assertIn("/capture:ro", source)


if __name__ == "__main__":
    unittest.main()
