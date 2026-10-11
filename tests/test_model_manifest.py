import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from model_manifest import ROLES, create_manifest, verify_manifest


class ModelManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)
        for index, name in enumerate(ROLES):
            (self.model_dir / name).write_bytes(f"artifact-{index}".encode())

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_and_verify(self) -> None:
        manifest = create_manifest(self.model_dir)
        self.assertEqual([], verify_manifest(self.model_dir, manifest))
        self.assertEqual(set(ROLES), {entry["name"] for entry in manifest["files"]})

    def test_detects_corruption_without_hashing_wrong_size(self) -> None:
        manifest = create_manifest(self.model_dir)
        changed = self.model_dir / next(iter(ROLES))
        changed.write_bytes(b"different-length-data")
        errors = verify_manifest(self.model_dir, manifest)
        self.assertTrue(any("size mismatch" in error for error in errors))

    def test_detects_same_size_corruption(self) -> None:
        manifest = create_manifest(self.model_dir)
        name = next(iter(ROLES))
        path = self.model_dir / name
        path.write_bytes(b"x" * path.stat().st_size)
        errors = verify_manifest(self.model_dir, manifest)
        self.assertTrue(any("sha256 mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
