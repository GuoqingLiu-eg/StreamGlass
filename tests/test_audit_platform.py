import unittest

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from audit_platform import canonical_fingerprint, parse_cuda_version, parse_os_release


class AuditPlatformTest(unittest.TestCase):
    def test_parse_os_release(self) -> None:
        self.assertEqual(
            parse_os_release('NAME="Ubuntu"\nVERSION_ID="24.04"\n# comment\n'),
            {"NAME": "Ubuntu", "VERSION_ID": "24.04"},
        )

    def test_parse_cuda_version(self) -> None:
        self.assertEqual(parse_cuda_version("Cuda compilation tools, release 13.2, V13.2.78"), "13.2")

    def test_fingerprint_ignores_timestamp_and_existing_fingerprint(self) -> None:
        left = {"schema_version": 1, "generated_at": "a", "value": 4}
        right = {"schema_version": 1, "generated_at": "b", "fingerprint": "stale", "value": 4}
        self.assertEqual(canonical_fingerprint(left), canonical_fingerprint(right))


if __name__ == "__main__":
    unittest.main()

