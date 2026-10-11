# StreamGlass UI Prototype

This directory contains the OUTSMART website and its lightweight AGX control
bridge. The page still renders when the translation runtime is offline.

Open `index.html` directly, or run a local static server:

```bash
python3 ui-prototype/server.py --port 4173
```

Then visit `http://127.0.0.1:4173`.

When `streamglass-native-realtime` is running, the page connects to its local
Unix control socket and provides:

- online target-language switching without reloading TensorRT engines;
- pause or resume of capture, inference, and playback;
- live runtime state synchronization;
- optional in-memory PipeWire mirroring to a selected speaker while the
  Bluetooth glasses remain the primary output;
- preview the unified world-map project poster;
- test responsive desktop and mobile layouts.

The speaker mirror captures the monitor of `streamglass_aec_sink`, so both
outputs share the same WebRTC AEC reference. It does not write PCM or WAV files.
Physical full-duplex validation is still required after connecting the intended
speaker because its acoustic delay can differ from the glasses.

The HTTP server binds to `127.0.0.1` by default. Access an AGX deployment with
an SSH tunnel rather than exposing the control surface to the LAN:

```bash
ssh -N -L 8080:127.0.0.1:8080 jetson@AGX_HOST
```
