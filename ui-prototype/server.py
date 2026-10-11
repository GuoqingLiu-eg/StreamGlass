#!/usr/bin/env python3
"""Serve the OUTSMART UI and proxy controls to the native runtime."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import socket
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


STATIC_ROOT = Path(__file__).resolve().parent
MAXIMUM_REQUEST_BYTES = 4096
MAXIMUM_RESPONSE_BYTES = 65536
SPEAKER_MIRROR_NODE = "streamglass-speaker-mirror"
AEC_REFERENCE_SINK = "streamglass_aec_sink"
DEFAULT_SPEAKER_VOLUME = 0.9


def default_control_socket() -> Path:
    configured = os.environ.get("STREAMGLASS_CONTROL_SOCKET")
    if configured:
        return Path(configured)
    runtime_directory = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_directory:
        return Path(runtime_directory) / "streamglass" / "control.sock"
    return Path("/tmp") / f"streamglass-{os.getuid()}.sock"


def default_playback_config() -> Path:
    configured = os.environ.get("STREAMGLASS_PLAYBACK_CONFIG")
    if configured:
        return Path(configured)
    return Path.home() / ".config" / "streamglass" / "playback.json"


def runtime_command(command: str, control_socket: Path) -> dict[str, Any]:
    if "\n" in command or "\r" in command:
        raise ValueError("control command must be one line")

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(5.0)
    try:
        client.connect(str(control_socket))
        client.sendall(command.encode("utf-8") + b"\n")
        response = bytearray()
        while len(response) < MAXIMUM_RESPONSE_BYTES:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
            if b"\n" in chunk:
                break
    finally:
        client.close()

    if not response:
        raise RuntimeError("native runtime returned no control response")
    return json.loads(response.decode("utf-8"))


class RuntimeStatusBroker:
    """Poll one local control socket and fan changes out to SSE clients."""

    def __init__(self, control_socket: Path, interval: float = 2.0) -> None:
        self.control_socket = control_socket
        self.interval = interval
        self.condition = threading.Condition()
        self.shutdown_event = threading.Event()
        self.generation = 0
        self.signature = ""
        self.status: dict[str, Any] = {
            "ok": True,
            "online": False,
            "paused": False,
            "error": "runtime status is initializing",
        }
        self.thread = threading.Thread(
            target=self._run,
            name="runtime-status-broker",
            daemon=True,
        )
        self.thread.start()

    def publish(self, status: dict[str, Any]) -> None:
        signature = json.dumps(
            status,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.condition:
            if signature == self.signature:
                return
            self.status = dict(status)
            self.signature = signature
            self.generation += 1
            self.condition.notify_all()

    def _run(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                status = runtime_command("status", self.control_socket)
            except (OSError, RuntimeError, json.JSONDecodeError) as error:
                status = {
                    "ok": True,
                    "online": False,
                    "paused": False,
                    "error": str(error),
                }
            self.publish(status)
            self.shutdown_event.wait(self.interval)

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        with self.condition:
            return self.generation, dict(self.status)

    def wait_after(
        self,
        generation: int,
        timeout: float,
    ) -> tuple[int, dict[str, Any], bool]:
        with self.condition:
            changed = self.condition.wait_for(
                lambda: self.generation != generation
                or self.shutdown_event.is_set(),
                timeout=timeout,
            )
            return self.generation, dict(self.status), changed

    def close(self) -> None:
        self.shutdown_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


def pipewire_nodes() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["pw-dump"],
        check=True,
        capture_output=True,
        text=True,
        timeout=1.5,
    )
    payload = result.stdout.lstrip()
    objects, _ = json.JSONDecoder().raw_decode(payload)
    if not isinstance(objects, list):
        raise ValueError("PipeWire registry is not a JSON array")
    # PipeWire can briefly emit null registry slots while a Bluetooth or USB
    # node is being removed. They are normal transient state, not devices.
    return [item for item in objects if isinstance(item, dict)]


def pipewire_properties(item: dict[str, Any]) -> dict[str, Any]:
    info = item.get("info")
    if not isinstance(info, dict):
        return {}
    properties = info.get("props")
    return properties if isinstance(properties, dict) else {}


def pipewire_node_names() -> set[str]:
    names: set[str] = set()
    for item in pipewire_nodes():
        props = pipewire_properties(item)
        name = props.get("node.name")
        if isinstance(name, str):
            names.add(name)
    return names


def speaker_outputs() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for item in pipewire_nodes():
        props = pipewire_properties(item)
        if props.get("media.class") != "Audio/Sink":
            continue
        name = props.get("node.name")
        if not isinstance(name, str):
            continue
        # The glasses and the AEC virtual sink are owned by the primary
        # playback route. Mirroring either one would create a feedback loop.
        if name == AEC_REFERENCE_SINK or name.startswith("bluez_output."):
            continue
        description = (
            props.get("node.description")
            or props.get("device.description")
            or props.get("device.alias")
            or name
        )
        outputs.append(
            {
                "id": item.get("id", 0)
                if isinstance(item.get("id", 0), int)
                else 0,
                "nodeName": name,
                "name": str(description),
                "deviceAlias": str(props.get("device.alias", "")),
            }
        )
    outputs.sort(key=lambda output: (output["name"], output["nodeName"]))
    return outputs


class SpeakerMirrorManager:
    """Keep an optional, disk-free PipeWire mirror alive."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.wake_event = threading.Event()
        self.process: subprocess.Popen[str] | None = None
        self.process_stderr: deque[str] = deque(maxlen=8)
        self.enabled = False
        self.node_name = ""
        self.volume = DEFAULT_SPEAKER_VOLUME
        self.applied_volume: float | None = None
        self.applied_volume_output_id: int | None = None
        self.last_error = ""
        self._load()
        self.thread = threading.Thread(
            target=self._run,
            name="speaker-mirror-manager",
            daemon=True,
        )
        self.thread.start()

    def _load(self) -> None:
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.enabled = bool(payload.get("enabled", False))
            node_name = payload.get("nodeName", "")
            self.node_name = node_name if isinstance(node_name, str) else ""
            volume = payload.get("volume", DEFAULT_SPEAKER_VOLUME)
            if (
                isinstance(volume, bool)
                or not isinstance(volume, (int, float))
                or not 0.0 <= float(volume) <= 1.0
            ):
                raise ValueError("volume must be between 0 and 1")
            self.volume = float(volume)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            self.last_error = f"invalid playback config: {error}"

    def _save_locked(self) -> None:
        self.config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "enabled": self.enabled,
                    "nodeName": self.node_name,
                    "volume": self.volume,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.config_path)

    def configure(
        self,
        enabled: bool,
        node_name: str,
        volume: float | None = None,
    ) -> dict[str, Any]:
        if volume is not None:
            if (
                isinstance(volume, bool)
                or not isinstance(volume, (int, float))
                or not 0.0 <= float(volume) <= 1.0
            ):
                raise ValueError("volume must be between 0 and 1")
            requested_volume = float(volume)
        else:
            requested_volume = self.volume
        if enabled:
            available = {output["nodeName"] for output in speaker_outputs()}
            if node_name not in available:
                raise ValueError("selected speaker output is not available")
        with self.lock:
            route_changed = (
                enabled != self.enabled or node_name != self.node_name
            )
            volume_changed = requested_volume != self.volume
            self.enabled = enabled
            self.node_name = node_name
            self.volume = requested_volume
            self.last_error = ""
            if route_changed:
                self._stop_process_locked()
            if route_changed or volume_changed:
                self.applied_volume = None
                self.applied_volume_output_id = None
            self._save_locked()
        self.wake_event.set()
        return self.status()

    def _apply_volume_locked(self, output_id: int) -> None:
        if (
            self.applied_volume_output_id == output_id
            and self.applied_volume == self.volume
        ):
            return
        subprocess.run(
            ["wpctl", "set-volume", str(output_id), f"{self.volume:.2f}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        self.applied_volume_output_id = output_id
        self.applied_volume = self.volume

    def _stop_process_locked(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _start_process_locked(self) -> None:
        command = [
            "pw-loopback",
            "-n",
            SPEAKER_MIRROR_NODE,
            "-c",
            "1",
            # PipeWire 1.0.5 on JetPack 6.2.2 advertises the long form but
            # rejects it at runtime. The short options work across versions.
            "-m",
            "[ MONO ]",
            "-l",
            "40",
            "-C",
            AEC_REFERENCE_SINK,
            "--capture-props",
            (
                "stream.capture.sink=true node.passive=true "
                "audio.rate=16000 audio.channels=1 audio.position=[ MONO ]"
            ),
            "-P",
            self.node_name,
            "--playback-props",
            "node.passive=true audio.rate=16000 audio.channels=1",
        ]
        self.process_stderr.clear()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        threading.Thread(
            target=self._drain_process_stderr,
            args=(self.process,),
            name="speaker-mirror-stderr",
            daemon=True,
        ).start()

    def _drain_process_stderr(
        self,
        process: subprocess.Popen[str],
    ) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                message = line.strip()
                if message:
                    self.process_stderr.append(message)
        except (OSError, TypeError, ValueError):
            return

    def _process_failure(self) -> str:
        if not self.process_stderr:
            return "speaker mirror could not connect"
        detail = " | ".join(self.process_stderr)
        return f"speaker mirror could not connect: {detail[-768:]}"

    def _reconcile(self) -> None:
        with self.lock:
            active = self.process is not None and self.process.poll() is None
            if not self.enabled:
                self._stop_process_locked()
                return

            try:
                outputs = speaker_outputs()
                names = pipewire_node_names()
            except (
                OSError,
                subprocess.SubprocessError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                self.last_error = f"PipeWire unavailable: {error}"
                return
            selected_output = next(
                (
                    output
                    for output in outputs
                    if output["nodeName"] == self.node_name
                ),
                None,
            )
            if self.node_name not in names or selected_output is None:
                self._stop_process_locked()
                self.applied_volume = None
                self.applied_volume_output_id = None
                self.last_error = "selected speaker output is disconnected"
                return
            try:
                self._apply_volume_locked(int(selected_output["id"]))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.last_error = f"speaker volume could not be applied: {error}"
                return
            if AEC_REFERENCE_SINK not in names:
                self._stop_process_locked()
                self.last_error = "waiting for the translation AEC sink"
                return

            if not active:
                try:
                    self._start_process_locked()
                except OSError as error:
                    self.last_error = f"speaker mirror failed to start: {error}"
                    self.process = None
                    return
                time.sleep(0.08)
                if self.process.poll() is not None:
                    time.sleep(0.02)
                    self.last_error = self._process_failure()
                    self.process = None
                    return
            self.last_error = ""

    def _run(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                self._reconcile()
            except Exception as error:
                # Device hot-plug changes must not permanently kill the
                # manager thread. Keep the failure visible and retry.
                with self.lock:
                    self._stop_process_locked()
                    self.last_error = (
                        f"speaker mirror manager recovered from: {error}"
                    )
            self.wake_event.wait(1.0)
            self.wake_event.clear()

    def status(self) -> dict[str, Any]:
        try:
            outputs = speaker_outputs()
            discovery_error = ""
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            outputs = []
            discovery_error = f"PipeWire unavailable: {error}"
        try:
            aec_ready = AEC_REFERENCE_SINK in pipewire_node_names()
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            ValueError,
        ):
            aec_ready = False
        with self.lock:
            active = self.process is not None and self.process.poll() is None
            return {
                "ok": True,
                "enabled": self.enabled,
                "active": active,
                "nodeName": self.node_name,
                "volume": self.volume,
                "outputs": outputs,
                "aecReady": aec_ready or active,
                "captureMode": "full-duplex-aec",
                "error": self.last_error or discovery_error,
            }

    def close(self) -> None:
        self.shutdown_event.set()
        self.wake_event.set()
        self.thread.join(timeout=2.0)
        with self.lock:
            self._stop_process_locked()


class StreamGlassHandler(SimpleHTTPRequestHandler):
    server_version = "StreamGlassUI/1.0"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def end_headers(self) -> None:
        # This device may boot with an invalid clock, so stale browser validators
        # can otherwise keep an older UI indefinitely during development.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    @property
    def control_socket(self) -> Path:
        return self.server.control_socket  # type: ignore[attr-defined]

    @property
    def playback_manager(self) -> SpeakerMirrorManager:
        return self.server.playback_manager  # type: ignore[attr-defined]

    @property
    def runtime_broker(self) -> RuntimeStatusBroker:
        broker = self.server.runtime_broker  # type: ignore[attr-defined]
        if broker is None:
            raise RuntimeError("runtime status broker is unavailable")
        return broker

    def send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def runtime_status(self) -> dict[str, Any]:
        _, status = self.runtime_broker.snapshot()
        return status

    def send_runtime_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        generation = -1
        try:
            while True:
                generation, status, changed = self.runtime_broker.wait_after(
                    generation,
                    timeout=15.0,
                )
                if changed:
                    payload = json.dumps(
                        status,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    message = f"id: {generation}\nevent: runtime\ndata: {payload}\n\n"
                else:
                    message = ": keep-alive\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "service": "streamglass-ui",
                    "runtime": self.runtime_status(),
                }
            )
            return
        if route == "/api/status":
            self.send_json(self.runtime_status())
            return
        if route == "/api/events":
            self.send_runtime_events()
            return
        if route == "/api/playback":
            self.send_json(self.playback_manager.status())
            return
        super().do_GET()

    def do_POST(self) -> None:
        route = urlsplit(self.path).path
        if route not in {"/api/control", "/api/playback"}:
            self.send_json(
                {"ok": False, "error": "route not found"},
                HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAXIMUM_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(
                self.rfile.read(content_length).decode("utf-8")
            )
            if route == "/api/playback":
                enabled = payload.get("enabled")
                node_name = payload.get("nodeName", "")
                volume = payload.get("volume")
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be a boolean")
                if not isinstance(node_name, str):
                    raise ValueError("nodeName must be a string")
                if volume is not None and (
                    isinstance(volume, bool)
                    or not isinstance(volume, (int, float))
                ):
                    raise ValueError("volume must be a number")
                self.send_json(
                    self.playback_manager.configure(
                        enabled,
                        node_name,
                        float(volume) if volume is not None else None,
                    )
                )
                return

            action = payload.get("action")
            if action in {"pause", "resume", "toggle"}:
                command = action
            elif action == "language":
                language = payload.get("language")
                if not isinstance(language, str):
                    raise ValueError("language must be a string")
                command = f"language {language}"
            else:
                raise ValueError("unsupported control action")

            response = runtime_command(command, self.control_socket)
            self.runtime_broker.publish(response)
            status = (
                HTTPStatus.OK
                if response.get("ok")
                else HTTPStatus.BAD_REQUEST
            )
            self.send_json(response, status)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.send_json(
                {
                    "ok": False,
                    "online": False,
                    "error": str(error),
                },
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

    def log_message(self, message: str, *args: Any) -> None:
        print(
            f"[web] {self.address_string()} "
            f"{self.log_date_time_string()} {message % args}",
            flush=True,
        )


class StreamGlassServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        control_socket: Path,
        playback_config: Path,
    ) -> None:
        self.control_socket = control_socket
        self.runtime_broker: RuntimeStatusBroker | None = None
        self.playback_manager: SpeakerMirrorManager | None = None
        super().__init__(address, StreamGlassHandler)
        self.runtime_broker = RuntimeStatusBroker(control_socket)
        self.playback_manager = SpeakerMirrorManager(playback_config)

    def server_close(self) -> None:
        if self.runtime_broker is not None:
            self.runtime_broker.close()
        if self.playback_manager is not None:
            self.playback_manager.close()
        super().server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.environ.get("STREAMGLASS_WEB_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("STREAMGLASS_WEB_PORT", "8080")),
    )
    parser.add_argument(
        "--control-socket",
        type=Path,
        default=default_control_socket(),
    )
    parser.add_argument(
        "--playback-config",
        type=Path,
        default=default_playback_config(),
    )
    arguments = parser.parse_args()

    server = StreamGlassServer(
        (arguments.host, arguments.port),
        arguments.control_socket,
        arguments.playback_config,
    )
    print(
        f"[web] listening http://{arguments.host}:{arguments.port} "
        f"control={arguments.control_socket}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
