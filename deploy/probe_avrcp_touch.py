#!/usr/bin/env python3
"""Register a temporary BlueZ AVRCP target and print headset media actions."""

from __future__ import annotations

import argparse
import signal
import sys
import time

import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib


PLAYER_INTERFACE = "org.mpris.MediaPlayer2.Player"
PROPERTIES_INTERFACE = "org.freedesktop.DBus.Properties"
PLAYER_PATH = "/org/streamglass/avrcp"


class AvrcpProbe(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus) -> None:
        super().__init__(bus, PLAYER_PATH)
        self._status = "Playing"

    def _event(self, name: str) -> None:
        print(f"[avrcp] {time.monotonic():.6f} {name}", flush=True)

    @dbus.service.method(PLAYER_INTERFACE, in_signature="", out_signature="")
    def Play(self) -> None:
        self._status = "Playing"
        self._event("Play")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="", out_signature="")
    def Pause(self) -> None:
        self._status = "Paused"
        self._event("Pause")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="", out_signature="")
    def PlayPause(self) -> None:
        self._status = "Paused" if self._status == "Playing" else "Playing"
        self._event("PlayPause")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="", out_signature="")
    def Stop(self) -> None:
        self._status = "Stopped"
        self._event("Stop")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="", out_signature="")
    def Next(self) -> None:
        self._event("Next")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="", out_signature="")
    def Previous(self) -> None:
        self._event("Previous")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="x", out_signature="")
    def Seek(self, offset: int) -> None:
        self._event(f"Seek offset={int(offset)}")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="ox", out_signature="")
    def SetPosition(self, track_id: dbus.ObjectPath, position: int) -> None:
        self._event(f"SetPosition track={track_id} position={int(position)}")

    @dbus.service.method(PLAYER_INTERFACE, in_signature="s", out_signature="")
    def OpenUri(self, uri: str) -> None:
        self._event(f"OpenUri uri={uri}")

    @dbus.service.method(
        PROPERTIES_INTERFACE, in_signature="ss", out_signature="v"
    )
    def Get(self, interface: str, name: str) -> object:
        return self.GetAll(interface)[name]

    @dbus.service.method(
        PROPERTIES_INTERFACE, in_signature="s", out_signature="a{sv}"
    )
    def GetAll(self, interface: str) -> dict[str, object]:
        if interface != PLAYER_INTERFACE:
            return {}
        return {
            "PlaybackStatus": dbus.String(self._status),
            "LoopStatus": dbus.String("None"),
            "Rate": dbus.Double(1.0),
            "Shuffle": dbus.Boolean(False),
            "Metadata": dbus.Dictionary(
                {
                    "mpris:trackid": dbus.ObjectPath(
                        "/org/streamglass/avrcp/translation"
                    ),
                    "xesam:title": dbus.String("StreamGlass Translation"),
                },
                signature="sv",
            ),
            "Volume": dbus.Double(1.0),
            "Position": dbus.Int64(0),
            "MinimumRate": dbus.Double(1.0),
            "MaximumRate": dbus.Double(1.0),
            "CanGoNext": dbus.Boolean(False),
            "CanGoPrevious": dbus.Boolean(False),
            "CanPlay": dbus.Boolean(True),
            "CanPause": dbus.Boolean(True),
            "CanSeek": dbus.Boolean(False),
            "CanControl": dbus.Boolean(True),
        }

    @dbus.service.method(
        PROPERTIES_INTERFACE, in_signature="ssv", out_signature=""
    )
    def Set(self, interface: str, name: str, value: object) -> None:
        self._event(f"Set interface={interface} name={name} value={value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="/org/bluez/hci1")
    parser.add_argument(
        "--device",
        default="/org/bluez/hci1/dev_41_42_84_69_36_F8",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    probe = AvrcpProbe(bus)
    media = dbus.Interface(
        bus.get_object("org.bluez", args.adapter), "org.bluez.Media1"
    )
    properties = dbus.Dictionary(
        {
            "PlaybackStatus": dbus.String("playing"),
            "CanPlay": dbus.Boolean(True),
            "CanPause": dbus.Boolean(True),
            "CanControl": dbus.Boolean(True),
            "Identity": dbus.String("StreamGlass"),
        },
        signature="sv",
    )
    media.RegisterPlayer(PLAYER_PATH, properties)
    device = dbus.Interface(
        bus.get_object("org.bluez", args.device), "org.bluez.Device1"
    )
    try:
        device.ConnectProfile("0000110e-0000-1000-8000-00805f9b34fb")
    except dbus.DBusException as error:
        if error.get_dbus_name() not in {
            "org.bluez.Error.AlreadyConnected",
            "org.bluez.Error.InProgress",
        }:
            raise
    print(
        f"[avrcp] registered adapter={args.adapter} device={args.device} "
        f"duration={args.seconds:.1f}s",
        flush=True,
    )

    loop = GLib.MainLoop()
    timer_id = GLib.timeout_add(
        max(1, int(args.seconds * 1000.0)),
        lambda: (loop.quit(), False)[1],
    )

    def stop(_signum: int, _frame: object) -> None:
        loop.quit()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        loop.run()
    finally:
        try:
            media.UnregisterPlayer(PLAYER_PATH)
        except dbus.DBusException:
            pass
        probe.remove_from_connection()
    print("[avrcp] unregistered", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
