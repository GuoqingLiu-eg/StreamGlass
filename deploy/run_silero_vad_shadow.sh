#!/usr/bin/env bash
set -euo pipefail

UNIT="streamglass-silero-vad-shadow.service"
EXPERIMENT_ID="silero-vad-v6.2.1"
BUILD_DIR="${STREAMGLASS_SILERO_BUILD_DIR:-/opt/streamglass/tmp/experiments/$EXPERIMENT_ID/build-sm87}"
BIN="${STREAMGLASS_SILERO_SHADOW_BIN:-$BUILD_DIR/streamglass-silero-vad-shadow}"
ENGINE="${STREAMGLASS_SILERO_ENGINE:-/opt/streamglass/engine-cache/experiments/$EXPERIMENT_ID/silero_vad_16k_fp16.engine}"
SOURCE="${STREAMGLASS_SILERO_SHADOW_SOURCE:-streamglass_aec_source}"
LOG="/dev/shm/streamglass-silero-vad-shadow.log"

start_shadow() {
    [[ -x "$BIN" ]] || { echo "Missing shadow binary: $BIN" >&2; exit 1; }
    [[ -s "$ENGINE" ]] || { echo "Missing Silero engine: $ENGINE" >&2; exit 1; }
    systemctl --user stop "$UNIT" 2>/dev/null || true
    : >"$LOG"
    systemd-run --user --unit "$UNIT" --collect \
        --setenv=STREAMGLASS_VAD_START_RMS="${STREAMGLASS_VAD_START_RMS:-0.030}" \
        --setenv=STREAMGLASS_VAD_CONTINUE_RMS="${STREAMGLASS_VAD_CONTINUE_RMS:-0.016}" \
        --setenv=STREAMGLASS_VAD_START_MS="${STREAMGLASS_VAD_START_MS:-144}" \
        --setenv=STREAMGLASS_VAD_PREROLL_MS="${STREAMGLASS_VAD_PREROLL_MS:-192}" \
        --setenv=STREAMGLASS_VAD_SILENCE_MS="${STREAMGLASS_VAD_SILENCE_MS:-544}" \
        --setenv=STREAMGLASS_UTTERANCE_HARD_CAP_MS="${STREAMGLASS_UTTERANCE_HARD_CAP_MS:-12800}" \
        --setenv=STREAMGLASS_SILERO_THRESHOLD="${STREAMGLASS_SILERO_THRESHOLD:-0.5}" \
        /bin/bash -c \
        "exec pw-record --target '$SOURCE' --rate 16000 --channels 1 --format f32 - 2>>'$LOG' | exec '$BIN' '$ENGINE' 2>>'$LOG'"
    for _ in $(seq 1 50); do
        if grep -q '\[shadow-vad\] ready' "$LOG" 2>/dev/null; then
            echo "Silero shadow VAD is observing $SOURCE."
            echo "Log: $LOG"
            return
        fi
        sleep 0.1
    done
    systemctl --user status "$UNIT" --no-pager -n 20 || true
    cat "$LOG" >&2
    exit 1
}

case "${1:-}" in
    start)
        start_shadow
        ;;
    stop)
        systemctl --user stop "$UNIT" 2>/dev/null || true
        echo "Silero shadow VAD stopped."
        ;;
    status)
        systemctl --user status "$UNIT" --no-pager -n 30
        ;;
    logs)
        tail -f "$LOG"
        ;;
    snapshot)
        tail -n 80 "$LOG"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|snapshot}" >&2
        exit 2
        ;;
esac
