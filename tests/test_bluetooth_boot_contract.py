import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BluetoothBootContractTests(unittest.TestCase):
    def test_recovery_is_bounded_and_has_bredr_fallback(self) -> None:
        source = (ROOT / "deploy/connect_barrot_glasses.sh").read_text()
        self.assertIn("STREAMGLASS_BT_DIRECT_CONNECT_ATTEMPTS", source)
        self.assertIn("STREAMGLASS_BT_CONNECT_TIMEOUT_SECONDS", source)
        self.assertIn("STREAMGLASS_BT_BREDR_SCAN_SECONDS", source)
        self.assertIn("scan bredr", source)
        self.assertIn("scan off", source)
        self.assertIn("post-scan", source)
        self.assertIn("yielding to systemd retry", source)
        self.assertIn("org.bluez.Device1 Connect", source)
        self.assertIn("bluez_device_path", source)
        self.assertNotIn("connect %s\\nquit", source)
        self.assertNotIn("until device_connected && audio_endpoint_available", source)
        recovery = source[
            source.index("connect_until_ready()") :
            source.index('echo "[barrot] loading btusb')
        ]
        self.assertNotIn("restart wireplumber.service", recovery)

    def test_bluetooth_readiness_precedes_realtime_start(self) -> None:
        source = (ROOT / "deploy/connect_barrot_glasses.sh").read_text()
        main = source[source.index('echo "[barrot] loading btusb') :]
        connect = main.index("if ! connect_until_ready; then")
        stable = main.index('wait_for_stable_endpoints "$ENDPOINT_WAIT_SECONDS"')
        start = main.index("start_realtime_session")
        monitor = main.index("monitoring the controller")
        self.assertLess(connect, stable)
        self.assertLess(stable, start)
        self.assertLess(start, monitor)

    def test_realtime_unit_has_no_unbounded_endpoint_prestart(self) -> None:
        source = (
            ROOT / "deploy/systemd/streamglass-realtime.service"
        ).read_text()
        self.assertNotIn("ExecStartPre", source)
        self.assertNotIn("TimeoutStartSec=infinity", source)

    def test_boot_installer_delegates_start_to_bluetooth_service(self) -> None:
        source = (ROOT / "deploy/install_boot_services.sh").read_text()
        self.assertIn('realtime_service.sh" install', source)
        self.assertIn(
            "systemctl --user disable --now streamglass-realtime.service",
            source,
        )
        self.assertIn(
            "systemctl restart --no-block streamglass-bluetooth.service",
            source,
        )


if __name__ == "__main__":
    unittest.main()
