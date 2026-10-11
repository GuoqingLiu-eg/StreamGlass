import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerContextTest(unittest.TestCase):
    def test_root_model_directory_does_not_hide_vendored_python_packages(self) -> None:
        patterns = {
            line.strip()
            for line in (PROJECT_ROOT / ".dockerignore").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("/models", patterns)
        self.assertNotIn("models", patterns)
        self.assertTrue(
            (PROJECT_ROOT / "third_party/torchaudio/src/torchaudio/models/__init__.py").is_file()
        )

    def test_project_sync_only_excludes_root_model_artifacts(self) -> None:
        script = (PROJECT_ROOT / "deploy/sync_project.sh").read_text()

        self.assertIn("--exclude '/models/'", script)
        self.assertNotIn("--exclude 'models/'", script)


if __name__ == "__main__":
    unittest.main()
