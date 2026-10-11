import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "streamglass_ui_server",
    ROOT / "ui-prototype" / "server.py",
)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def pipewire_node(
    node_id: int,
    name: str,
    description: str,
    media_class: str = "Audio/Sink",
) -> dict:
    return {
        "id": node_id,
        "info": {
            "props": {
                "node.name": name,
                "node.description": description,
                "media.class": media_class,
            }
        },
    }


class SpeakerOutputTests(unittest.TestCase):
    def test_transient_null_pipewire_entries_are_ignored(self) -> None:
        completed = mock.Mock(
            stdout=json.dumps(
                [
                    None,
                    {"id": 2, "info": None},
                    pipewire_node(
                        3,
                        "alsa_output.usb-speaker",
                        "USB Speaker",
                    ),
                ]
            )
        )
        with mock.patch.object(
            SERVER.subprocess,
            "run",
            return_value=completed,
        ):
            self.assertEqual(len(SERVER.pipewire_nodes()), 2)
            self.assertEqual(
                SERVER.pipewire_node_names(),
                {"alsa_output.usb-speaker"},
            )

    def test_only_physical_non_glasses_sinks_are_listed(self) -> None:
        nodes = [
            pipewire_node(1, "streamglass_aec_sink", "AEC sink"),
            pipewire_node(2, "bluez_output.glasses.1", "Glasses"),
            pipewire_node(3, "alsa_output.usb-speaker", "USB Speaker"),
            pipewire_node(
                4,
                "alsa_input.usb-microphone",
                "USB Microphone",
                "Audio/Source",
            ),
        ]
        with mock.patch.object(SERVER, "pipewire_nodes", return_value=nodes):
            self.assertEqual(
                SERVER.speaker_outputs(),
                [
                    {
                        "id": 3,
                        "nodeName": "alsa_output.usb-speaker",
                        "name": "USB Speaker",
                        "deviceAlias": "",
                    }
                ],
            )


class SpeakerMirrorTests(unittest.TestCase):
    def test_volume_is_persisted_and_old_configs_default_to_ninety_percent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "playback.json"
            config_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "nodeName": "alsa_output.usb-speaker",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                SERVER.SpeakerMirrorManager,
                "_run",
                return_value=None,
            ):
                manager = SERVER.SpeakerMirrorManager(config_path)
            try:
                self.assertEqual(manager.volume, 0.9)
                with manager.lock:
                    manager.volume = 0.65
                    manager._save_locked()
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["volume"], 0.65)
            finally:
                manager.close()

    def test_volume_targets_current_pipewire_node_once_per_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                SERVER.SpeakerMirrorManager,
                "_run",
                return_value=None,
            ):
                manager = SERVER.SpeakerMirrorManager(
                    Path(directory) / "playback.json"
                )
            manager.volume = 0.75
            try:
                with mock.patch.object(
                    SERVER.subprocess,
                    "run",
                ) as run:
                    manager._apply_volume_locked(17)
                    manager._apply_volume_locked(17)
                run.assert_called_once_with(
                    ["wpctl", "set-volume", "17", "0.75"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
            finally:
                manager.close()

    def test_mirror_uses_aec_sink_monitor_without_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SERVER.SpeakerMirrorManager(
                Path(directory) / "playback.json"
            )
            manager.node_name = "alsa_output.usb-speaker"
            process = mock.Mock()
            process.poll.return_value = None
            process.stderr = io.StringIO("")
            try:
                with mock.patch.object(
                    SERVER.subprocess,
                    "Popen",
                    return_value=process,
                ) as popen:
                    manager._start_process_locked()
                command = popen.call_args.args[0]
                self.assertIn("-m", command)
                self.assertNotIn("--channel-map", command)
                self.assertIn("streamglass_aec_sink", command)
                self.assertTrue(
                    any("stream.capture.sink=true" in token for token in command)
                )
                self.assertIn("alsa_output.usb-speaker", command)
                self.assertFalse(
                    any(token.endswith((".wav", ".pcm")) for token in command)
                )
            finally:
                manager.process = None
                manager.close()


class SpeakerVolumeUiTests(unittest.TestCase):
    def test_ui_exposes_persistent_volume_control(self) -> None:
        html = (ROOT / "ui-prototype" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (ROOT / "ui-prototype" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="speakerVolume"', html)
        self.assertIn('id="speakerVolumeValue"', html)
        self.assertIn("JSON.stringify({ enabled, nodeName, volume })", script)
        self.assertIn("Number(speakerVolume.value) / 100", script)
        self.assertIn('speakerVolume.addEventListener("input"', script)


if __name__ == "__main__":
    unittest.main()
