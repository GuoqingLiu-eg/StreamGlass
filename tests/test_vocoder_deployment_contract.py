import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VocoderDeploymentContractTest(unittest.TestCase):
    def test_export_uses_official_generator_without_duration_prediction(self) -> None:
        source = (ROOT / "deploy/export/export_vocoder.py").read_text()
        self.assertIn("self.code_generator(sample, False)", source)
        self.assertIn('"unit_ids": {1: "unit_length"}', source)
        self.assertIn('"waveform": {2: "waveform_samples"}', source)
        self.assertNotIn("repeat_interleave", source)

    def test_engine_profile_covers_streaming_unit_lengths(self) -> None:
        source = (ROOT / "deploy/trt/build_vocoder_engine.sh").read_text()
        self.assertIn("--minShapes=unit_ids:1x1", source)
        self.assertIn("--optShapes=unit_ids:1x64", source)
        self.assertIn("--maxShapes=unit_ids:1x256", source)
        self.assertIn("--memPoolSize=workspace:8192", source)
        self.assertNotIn("workspace:8192MiB", source)
        self.assertIn("--builderOptimizationLevel=5", source)
        self.assertIn('engine_tmp="$engine_path.building.$$"', source)
        self.assertIn('mv -f "$engine_tmp" "$engine_path"', source)
        self.assertNotIn("/var/lib", source)

    def test_parity_report_directory_is_initialized_on_nvme(self) -> None:
        source = (ROOT / "deploy/trt/run_vocoder_parity.sh").read_text()
        self.assertIn("sudo install -d -m 0775", source)
        self.assertIn("/opt/streamglass/golden/trt/vocoder_parity.json", source)

    def test_benchmark_covers_compute_and_end_to_end_latency(self) -> None:
        source = (ROOT / "deploy/trt/benchmark_vocoder.sh").read_text()
        self.assertIn("for unit_length in 1 8 64 256", source)
        self.assertIn("for mode in compute end_to_end", source)
        self.assertIn("--useCudaGraph", source)
        self.assertIn("--useSpinWait", source)
        self.assertNotIn("--spinWait", source)
        self.assertIn("--exportTimes=", source)
        self.assertIn("summarize_trtexec.py", source)


if __name__ == "__main__":
    unittest.main()
