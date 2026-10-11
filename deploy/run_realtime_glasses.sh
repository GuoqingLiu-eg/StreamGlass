#!/usr/bin/env bash
# v1 glasses realtime session: pw-record | native-realtime | pw-play
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FP="${STREAMGLASS_PLATFORM_FINGERPRINT:-87d587047967766c63f3de7012a479fbd4f3c471ffea70efe8454017bf908709}"
BIN="${STREAMGLASS_RUNTIME_BIN:-/opt/streamglass/tmp/native-install/$FP/bin/streamglass-native-realtime}"
LANG_TARGET="${STREAMGLASS_RESTORE_TARGET_LANGUAGE:-${STREAMGLASS_TARGET_LANGUAGE:-eng}}"
RAW_SRC_NAME="bluez_input.41_42_84_69_36_F8.0"
RAW_SINK_NAME="bluez_output.41_42_84_69_36_F8.1"
AEC_ENABLED="${STREAMGLASS_AEC:-1}"
AEC_ARGS_FILE="$ROOT_DIR/config/pipewire/streamglass-echo-cancel.args"
AEC_HOST_PID_VALUE=""
AEC_FIFO=""
AEC_LOG=""
AEC_COMMAND_FD_OPEN=0
AUDIO_CAPTURE_FIFO=""
AUDIO_PLAYBACK_FIFO=""
AUDIO_CAPTURE_PID=""
AUDIO_RUNTIME_PID=""
AUDIO_PLAYBACK_PID=""
AUDIO_CAPTURE_FD_OPEN=0
AUDIO_PLAYBACK_FD_OPEN=0
AUDIO_RETRY_SECONDS="${STREAMGLASS_AUDIO_RETRY_SECONDS:-0.10}"
MAIN_BASHPID="$BASHPID"

[[ -x "$BIN" ]] || { echo "missing realtime binary: $BIN" >&2; exit 1; }
if [[ -n "${STREAMGLASS_SILERO_PRIMARY_ENGINE:-}" &&
      ! -s "$STREAMGLASS_SILERO_PRIMARY_ENGINE" ]]; then
    echo "missing Silero TensorRT engine: $STREAMGLASS_SILERO_PRIMARY_ENGINE" >&2
    exit 1
fi

if [ "$AEC_ENABLED" = "1" ]; then
    SRC_NAME="streamglass_aec_source"
    SINK_NAME="streamglass_aec_sink"
    HALF_DUPLEX_DEFAULT=0
else
    SRC_NAME="$RAW_SRC_NAME"
    SINK_NAME="$RAW_SINK_NAME"
    HALF_DUPLEX_DEFAULT=1
fi

node_exists() {
    local expected=$1
    pw-dump 2>/dev/null | python3 -c '
import json
import sys

expected = sys.argv[1]
payload = sys.stdin.read().lstrip()
if not payload:
    raise SystemExit(1)
try:
    objects, _ = json.JSONDecoder().raw_decode(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
if not isinstance(objects, list):
    raise SystemExit(1)

def properties(obj):
    if not isinstance(obj, dict):
        return {}
    info = obj.get("info")
    if not isinstance(info, dict):
        return {}
    props = info.get("props")
    return props if isinstance(props, dict) else {}

found = any(
    properties(obj).get("node.name") == expected
    for obj in objects
)
raise SystemExit(0 if found else 1)
' "$expected"
}

cleanup() {
    local pid=""
    # Asynchronous shell functions run in subshells. Only the top-level
    # supervisor owns the shared AEC and audio-bridge resources.
    if [ "$BASHPID" != "$MAIN_BASHPID" ]; then
        return
    fi
    for pid in "$AUDIO_CAPTURE_PID" "$AUDIO_RUNTIME_PID" \
               "$AUDIO_PLAYBACK_PID"; do
        if [ -n "$pid" ]; then
            # The background jobs are shell functions whose direct children
            # are pw-record, the native runtime/tee, and pw-play. Stop those
            # children first so waiting for the function cannot hang.
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$AUDIO_CAPTURE_PID" "$AUDIO_RUNTIME_PID" \
               "$AUDIO_PLAYBACK_PID"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [ "$AUDIO_CAPTURE_FD_OPEN" = "1" ]; then
        exec 7>&-
        AUDIO_CAPTURE_FD_OPEN=0
    fi
    if [ "$AUDIO_PLAYBACK_FD_OPEN" = "1" ]; then
        exec 8>&-
        AUDIO_PLAYBACK_FD_OPEN=0
    fi
    if [ -n "$AUDIO_CAPTURE_FIFO" ]; then
        rm -f "$AUDIO_CAPTURE_FIFO"
    fi
    if [ -n "$AUDIO_PLAYBACK_FIFO" ]; then
        rm -f "$AUDIO_PLAYBACK_FIFO"
    fi
    if [ "$AEC_COMMAND_FD_OPEN" = "1" ]; then
        exec 9>&-
        AEC_COMMAND_FD_OPEN=0
    fi
    if [ -n "$AEC_HOST_PID_VALUE" ]; then
        kill "$AEC_HOST_PID_VALUE" 2>/dev/null || true
        wait "$AEC_HOST_PID_VALUE" 2>/dev/null || true
    fi
    if [ -n "$AEC_FIFO" ]; then
        rm -f "$AEC_FIFO"
    fi
    if [ -n "$AEC_LOG" ]; then
        rm -f "$AEC_LOG"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_aec() {
    local aec_args=""

    [[ -f "$AEC_ARGS_FILE" ]] || {
        echo "missing AEC arguments: $AEC_ARGS_FILE" >&2
        exit 1
    }

    # pw-cli treats each newline as a separate command. Keep the checked-in
    # module arguments readable, then fold whitespace before sending one
    # complete load-module command to its control FIFO.
    aec_args=$(tr '\n\r\t' '   ' <"$AEC_ARGS_FILE" | tr -s ' ')
    [[ -n "${aec_args// /}" ]] || {
        echo "empty PipeWire AEC arguments: $AEC_ARGS_FILE" >&2
        exit 1
    }

    AEC_FIFO="${XDG_RUNTIME_DIR:-/tmp}/streamglass-aec-$$.fifo"
    AEC_LOG="${XDG_RUNTIME_DIR:-/tmp}/streamglass-aec-$$.log"
    umask 077
    mkfifo "$AEC_FIFO"
    # pw-cli prints every registry event to stdout. Discard that stream so a
    # long session cannot flood logs or block the AEC control process.
    pw-cli <"$AEC_FIFO" >/dev/null 2>"$AEC_LOG" &
    AEC_HOST_PID_VALUE=$!
    exec 9>"$AEC_FIFO"
    AEC_COMMAND_FD_OPEN=1
    rm -f "$AEC_FIFO"
    AEC_FIFO=""
    printf 'load-module libpipewire-module-echo-cancel %s\n' \
        "$aec_args" >&9

    for _ in $(seq 1 50); do
        if node_exists streamglass_aec_source && \
           node_exists streamglass_aec_sink; then
            echo "PipeWire WebRTC AEC ready (pid=$AEC_HOST_PID_VALUE)" >&2
            return
        fi
        if ! kill -0 "$AEC_HOST_PID_VALUE" 2>/dev/null; then
            echo "PipeWire AEC host exited during startup" >&2
            cat "$AEC_LOG" >&2
            exit 1
        fi
        sleep 0.1
    done

    echo "PipeWire AEC nodes did not appear" >&2
    cat "$AEC_LOG" >&2
    exit 1
}

active_profile() {
    pw-dump 2>/dev/null | python3 -c '
import json
import sys

payload = sys.stdin.read().lstrip()
if not payload:
    raise SystemExit(1)
try:
    objects, _ = json.JSONDecoder().raw_decode(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
if not isinstance(objects, list):
    raise SystemExit(1)
for obj in objects:
    if not isinstance(obj, dict):
        continue
    info = obj.get("info")
    if not isinstance(info, dict):
        continue
    props = info.get("props")
    if not isinstance(props, dict) or props.get("device.alias") != "ZH-ZCGF":
        continue
    params = info.get("params")
    if not isinstance(params, dict):
        continue
    profiles = params.get("Profile")
    if not isinstance(profiles, list):
        continue
    for profile in profiles:
        if isinstance(profile, dict) and profile.get("index") is not None:
            print(profile["index"])
            break
    break
'
}

DEV=$(pw-dump 2>/dev/null | python3 -c '
import json
import sys

payload = sys.stdin.read().lstrip()
if not payload:
    raise SystemExit(1)
try:
    objects, _ = json.JSONDecoder().raw_decode(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
if not isinstance(objects, list):
    raise SystemExit(1)
for obj in objects:
    if not isinstance(obj, dict):
        continue
    info = obj.get("info")
    if not isinstance(info, dict):
        continue
    props = info.get("props")
    if isinstance(props, dict) and props.get("device.alias") == "ZH-ZCGF":
        obj_id = obj.get("id")
        if obj_id is not None:
            print(obj_id)
        break
')
[ -n "$DEV" ] || { echo "眼镜未连接" >&2; exit 1; }

if [ "$(active_profile)" != "261" ]; then
    echo "切换到 HFP mSBC..." >&2
    wpctl set-profile "$DEV" 261
    sleep 5
fi

# 采集流健康探测：确认真的有数据再启动主链
for attempt in 1 2 3; do
    BYTES=$(
        { timeout 2 pw-record --target "$RAW_SRC_NAME" --rate 16000 \
              --channels 1 --format f32 - 2>/dev/null || true; } \
            | wc -c
    )
    if [ "$BYTES" -gt 32000 ]; then
        echo "采集探测 OK (${BYTES}B)" >&2
        break
    fi
    echo "采集探测失败(第${attempt}次, ${BYTES}B)，重置 profile..." >&2
    wpctl set-profile "$DEV" 0; sleep 2
    wpctl set-profile "$DEV" 261; sleep 5
    [ "$attempt" = "3" ] && { echo "采集流无法建立" >&2; exit 1; }
done

if [ "$AEC_ENABLED" = "1" ]; then
    start_aec
fi

echo "audio route: source=$SRC_NAME sink=$SINK_NAME aec=$AEC_ENABLED" >&2

run_capture() {
    local failures=0
    local status=0
    while true; do
        if pw-record --target "$SRC_NAME" --rate 16000 \
                --channels 1 --format f32 -; then
            status=0
        else
            status=$?
        fi
        failures=$((failures + 1))
        if [ "$failures" -eq 1 ] || ((failures % 10 == 0)); then
            printf '[audio-bridge] capture endpoint unavailable; retry=%d status=%d\n' \
                "$failures" "$status" >&2
        fi
        sleep "$AUDIO_RETRY_SECONDS"
    done
}

run_runtime() {
    STREAMGLASS_HALF_DUPLEX="${STREAMGLASS_HALF_DUPLEX:-$HALF_DUPLEX_DEFAULT}" \
    "$BIN" "/opt/streamglass/engine-cache/$FP" \
        "/opt/streamglass/tmp/runtime-assets/official-v1" "$LANG_TARGET"
}

run_playback() {
    local failures=0
    local status=0
    while true; do
        if pw-play --target "$SINK_NAME" --rate 16000 \
                --channels 1 --format f32 -; then
            status=0
        else
            status=$?
        fi
        failures=$((failures + 1))
        if [ "$failures" -eq 1 ] || ((failures % 10 == 0)); then
            printf '[audio-bridge] playback endpoint unavailable; retry=%d status=%d\n' \
                "$failures" "$status" >&2
        fi
        sleep "$AUDIO_RETRY_SECONDS"
    done
}

run_runtime_output() {
    # PCM tapping is opt-in because a real-time session can otherwise write
    # many gigabytes to disk. Set an NVMe path only for explicit debugging.
    if [ -n "${STREAMGLASS_PLAYBACK_TAP:-}" ]; then
        run_runtime | tee "$STREAMGLASS_PLAYBACK_TAP"
    else
        run_runtime
    fi
}

# Keep both ends of these RAM-backed FIFOs open in the supervisor. If a
# Bluetooth/PipeWire endpoint disappears, only pw-record or pw-play exits and
# is retried; the native process does not see EOF/SIGPIPE, so its TensorRT
# engines, decoder state allocation and control socket remain resident.
AUDIO_CAPTURE_FIFO="${XDG_RUNTIME_DIR:-/tmp}/streamglass-capture-$$.fifo"
AUDIO_PLAYBACK_FIFO="${XDG_RUNTIME_DIR:-/tmp}/streamglass-playback-$$.fifo"
umask 077
mkfifo "$AUDIO_CAPTURE_FIFO" "$AUDIO_PLAYBACK_FIFO"
exec 7<>"$AUDIO_CAPTURE_FIFO"
AUDIO_CAPTURE_FD_OPEN=1
exec 8<>"$AUDIO_PLAYBACK_FIFO"
AUDIO_PLAYBACK_FD_OPEN=1

run_capture >"$AUDIO_CAPTURE_FIFO" &
AUDIO_CAPTURE_PID=$!
run_runtime_output <"$AUDIO_CAPTURE_FIFO" >"$AUDIO_PLAYBACK_FIFO" &
AUDIO_RUNTIME_PID=$!
run_playback <"$AUDIO_PLAYBACK_FIFO" &
AUDIO_PLAYBACK_PID=$!

# The open descriptors above keep the channels alive, so the pathnames are no
# longer needed and cannot leave stale runtime files after a crash.
rm -f "$AUDIO_CAPTURE_FIFO" "$AUDIO_PLAYBACK_FIFO"
AUDIO_CAPTURE_FIFO=""
AUDIO_PLAYBACK_FIFO=""

runtime_status=0
if wait "$AUDIO_RUNTIME_PID"; then
    runtime_status=0
else
    runtime_status=$?
fi
AUDIO_RUNTIME_PID=""
printf '[audio-bridge] native runtime exited status=%d\n' \
    "$runtime_status" >&2
exit "$runtime_status"
