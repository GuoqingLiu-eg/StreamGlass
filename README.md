# Seamless Streaming on Jetson AGX Orin

> 本仓库于 2026-10-11 从 Jetson 当前 staging 工程恢复。部署入口、备份范围和重建说明见 [备份与重建](docs/BACKUP_AND_REBUILD.md)。下面的性能数据为历史验收记录；本次拉取没有重新执行 Jetson 推理测试。

This repository is the deployable implementation workspace for the four design
documents in the parent directory. It is intentionally separate from the old
StreamSpeech runtime.

## Current state

| Gate | State | Meaning |
|---|---|---|
| S0 / G0 | complete | Platform, official reference, quality, and latency baseline |
| S1 / G1 | complete | Single-process C++ capture, inference, and playback plane |
| S2 / G2 | complete with tolerance | TensorRT modules and official 320 ms READ/WRITE policy |
| S3 / G3 | complete | Deterministic long-process soak and resource bounds |

The model architecture is not modified. FP16 execution, TensorRT conversion,
explicit cache I/O, CUDA Graphs, and external C++ scheduling are allowed. INT8,
pruning, distillation, vocabulary changes, and replacement attention are out of
scope.

The production replay matches the official text tokens, READ/WRITE decisions,
and output lengths. Unit parity is 136/137 (99.27%): one unit differs in the
first WRITE, while the final 80-unit WRITE is exact. This isolated numerical
tie is an accepted TensorRT precision tolerance, not a claim of bitwise 137/137
parity.

The native runtime keeps TensorRT contexts and GPU buffers alive across
utterances. Audio and playback use preallocated bounded SPSC pools with handles,
so queue transfer does not copy payloads or write audio to storage. Run the
long-process acceptance check with:

```bash
python3 deploy/validate_native_threaded_runtime.py --utterances 180
```

RSS is the process-local leak guard. `cudaMemGetInfo()` is retained as a coarse
128 MiB device-wide guard because Jetson reports global unified-memory state,
which can move in either direction independently of this process.

### AGX acceptance result

The final AGX run on 2026-07-21 passed 180 utterances in real-time mode followed
by 180 utterances in no-wait stress mode:

| Metric | Real-time | Stress |
|---|---:|---:|
| Policy calls / WRITEs | 1260 / 360 | 1260 / 360 |
| Wall time | 716.059 s | 112.861 s |
| Maximum inference | 223.570 ms | 214.228 ms |
| Capture queue high-water | 1 | 16 (intentional no-wait input) |
| Playback queue high-water | 1 | 1 |
| RSS growth | 70.64 MiB | 0.85 MiB |
| Output samples | 7,891,200 | 7,891,200 |
| Output checksum | 8665362568369647491 | 8665362568369647491 |

There was exactly one step above the 200 ms target per utterance: the final
WRITE. Every step remained below the accepted 240 ms hard limit. The report is
stored on the target NVMe at `/opt/streamglass/tmp/soak/final-180.json` with
SHA-256 `746b6b02fe362bd93b6a80b74363db5ae48ec2593ba8bf0a7a0a85b3f3865bbf`.

### Full-duplex glasses and echo control

`deploy/run_realtime_glasses.sh` creates a session-scoped PipeWire WebRTC AEC
graph. Capture comes from the headset HFP microphone, translated audio enters
the AEC reference sink, and only the echo-cancelled source reaches the native
runtime. No PCM is written unless `STREAMGLASS_PLAYBACK_TAP` is explicitly set.

The ZH-ZCGF Bluetooth path on this AGX was physically calibrated to a 180 ms
playback-reference delay. This value is device-specific and lives in
`config/pipewire/streamglass-echo-cancel.args`. Recalibrate after replacing the
headset or Bluetooth adapter:

```bash
python3 deploy/validate_full_duplex_aec.py --seconds 15
```

The production input gate uses the official Silero VAD v6.2.1 core as a static
16 kHz TensorRT FP16 engine. An utterance starts after 192 ms above probability
0.65, then uses a lower 0.35 continuation threshold so quiet or slow syllables
remain in the same sentence. It keeps 192 ms of pre-roll and ends after 544 ms
of non-speech. The RMS detector remains compiled in as an explicit fallback
when `STREAMGLASS_SILERO_PRIMARY_ENGINE` is unset. These values can be tuned
without rebuilding:

```bash
STREAMGLASS_SILERO_PRIMARY_ENGINE=/opt/streamglass/engine-cache/<fingerprint>/silero_vad_16k_fp16.engine \
STREAMGLASS_SILERO_START_THRESHOLD=0.65 \
STREAMGLASS_SILERO_CONTINUE_THRESHOLD=0.35 \
STREAMGLASS_VAD_START_MS=192 \
STREAMGLASS_VAD_PREROLL_MS=192 \
STREAMGLASS_VAD_SILENCE_MS=544 \
  ./deploy/run_realtime_glasses.sh
```

The production utterance hard cap is 12.8 seconds. Override it only after
validating the deployed T2U profiles for the wider text and character bounds:

```bash
STREAMGLASS_UTTERANCE_HARD_CAP_MS=12800 \
  ./deploy/run_realtime_glasses.sh
```

The archived physical regression used the earlier 10-second hard cap and the
official sample repeated in memory to 15 seconds. It produced exactly the two
expected utterances, no third echo-triggered utterance, and a maximum input
backlog of 64 ms:

```bash
python3 deploy/validate_feedback_loop.py \
  --seconds 15 --expected-utterances 2 --post-playback-seconds 10
```

### Headless boot session

Promote the verified Silero experiment engine into the current platform cache
before installing the service:

```bash
./deploy/trt/promote_silero_vad_engine.sh
```

Install the system Bluetooth monitor and the user realtime service once:

```bash
sudo ./deploy/install_boot_services.sh
```

At each boot, the system service loads the BARROT driver, enables the `jetson`
lingering user manager for headless audio, releases the GDM Bluetooth-audio
owner, and starts the user D-Bus/PipeWire stack. It first checks for existing
glasses audio nodes, makes three bounded direct-connect attempts, then performs
a 12-second BR/EDR discovery and one post-scan connect. Only after both the
glasses input and output nodes remain stable does it start the user C++ service.
If the bounded sequence fails, the Bluetooth service exits and systemd retries
after 15 seconds. The same bounded recovery is used after a disconnect.

Control the C++ service independently with the user-level entrypoint:

```bash
./deploy/realtime_service.sh status
./deploy/realtime_service.sh logs
./deploy/realtime_service.sh stop
./deploy/realtime_service.sh restart
./deploy/realtime_service.sh enable
./deploy/realtime_service.sh disable
```

On this JetPack image, the vendor `rtk_btusb` package suppresses the in-tree
`btusb` module during normal module loading. The BARROT `33fa:0012` adapter
requires `btusb`, while `rtk_btusb` remains attached to the separate Realtek
`0bda:c822` radio. The Bluetooth service therefore loads `btusb` with
`--ignore-install` before selecting the BARROT controller by MAC address.
WirePlumber 0.4 also defaults to logind seat arbitration, which hides Bluetooth
audio from a lingering user before graphical login. The connection script
enables linger only for the headless user audio manager; it does not enable the
translation service. It places
`config/wireplumber/51-streamglass-bluez.lua` in the target user's configuration
to permit headless ownership and HFP/mSBC profile restoration, and uses a
`NoInputNoOutput` BlueZ agent so no desktop pairing dialog is required.

The realtime log is kept in RAM at `/dev/shm/streamglass-realtime.log`; it is
cleared by reboot and does not continuously write the eMMC.

## Repository layout

```text
config/                 Frozen model and runtime configuration
deploy/                 Provisioning, target audit, synchronization, and services
tests/                  Host-side unit tests for deployment tooling
third_party/            Frozen official Seamless Communication source snapshot
```

Large checkpoints and generated engines are never committed. Their expected
names, byte sizes, and SHA-256 values live in `config/models.manifest.json`.

## Target connection

Set the host, user, and port for your own Jetson. Historical relay addresses
are not required by the runtime or deployment artifacts.

```bash
ssh -p <SSH_PORT> <JETSON_USER>@<JETSON_HOST>
```

The source device was backed up through its existing USB serial console and a
temporary read-only USB storage image. No SSH key or Bluetooth pairing key is
included in this repository.

## First commands

```bash
python3 -m unittest discover -s tests -v
python3 deploy/model_manifest.py verify \
  --model-dir /path/to/seamless_streaming \
  --manifest config/models.manifest.json

python3 deploy/audit_platform.py --output config/platform.local.json
```

Use `deploy/sync_project.sh` and `deploy/sync_models.sh` only after the target
data mount has been provisioned. Engine build services remain disabled until an
official PyTorch reference run and module parity pass.

The S0 reference environment uses an NVIDIA PyTorch container. Keep both
Docker metadata and containerd image layers on the target NVMe before pulling
it:

```bash
STREAMGLASS_BOOTSTRAP_PROXY=http://<proxy-host>:3128 \
  ./deploy/configure_docker.sh
```

Omit `STREAMGLASS_BOOTSTRAP_PROXY` when the AGX has direct DNS and internet
access. Running the command without a proxy also removes the temporary Docker
proxy drop-in. The migration intentionally retains `/var/lib/containerd` until
the NVMe-backed runtime has been verified; cleanup is a separate operation.

The official reference image pull is resumable and retries interrupted proxy
connections:

```bash
./deploy/pull_reference_image.sh
```
