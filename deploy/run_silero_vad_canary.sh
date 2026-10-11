#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT_DIR/deploy/$(basename "${BASH_SOURCE[0]}")"
PRODUCTION_UNIT="streamglass-realtime.service"
CANARY_UNIT="streamglass-silero-vad-canary.service"
SHADOW_UNIT="streamglass-silero-vad-shadow.service"
EXPERIMENT_ID="silero-vad-v6.2.1"
BUILD_DIR="${STREAMGLASS_SILERO_BUILD_DIR:-/opt/streamglass/tmp/experiments/$EXPERIMENT_ID/build-sm87}"
BIN="${STREAMGLASS_SILERO_CANARY_BIN:-$BUILD_DIR/streamglass-native-realtime}"
ENGINE="${STREAMGLASS_SILERO_ENGINE:-/opt/streamglass/engine-cache/experiments/$EXPERIMENT_ID/silero_vad_16k_fp16.engine}"
ENV_FILE="${STREAMGLASS_REALTIME_ENV:-$HOME/.config/streamglass/realtime.env}"
RUNNER="$ROOT_DIR/deploy/run_realtime_glasses.sh"
LOG="/dev/shm/streamglass-silero-vad-canary.log"

production_is_enabled() {
    systemctl --user is-enabled --quiet "$PRODUCTION_UNIT"
}

wait_for_canary_unloaded() {
    local load_state
    for _ in $(seq 1 50); do
        load_state=$(systemctl --user show "$CANARY_UNIT" \
            --property=LoadState --value 2>/dev/null || true)
        if [[ -z "$load_state" || "$load_state" == "not-found" ]]; then
            return
        fi
        systemctl --user reset-failed "$CANARY_UNIT" 2>/dev/null || true
        sleep 0.1
    done
    echo "Timed out waiting for the previous canary unit to unload." >&2
    exit 1
}

start_production() {
    systemctl --user reset-failed "$PRODUCTION_UNIT" 2>/dev/null || true
    systemctl --user start "$PRODUCTION_UNIT"
    echo "Original RMS realtime service restored (still enabled)."
}

rollback_canary() {
    systemctl --user stop "$CANARY_UNIT" 2>/dev/null || true
    systemctl --user reset-failed "$CANARY_UNIT" 2>/dev/null || true
    start_production
}

run_canary() {
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    export STREAMGLASS_RUNTIME_BIN="$BIN"
    export STREAMGLASS_SILERO_PRIMARY_ENGINE="$ENGINE"
    export STREAMGLASS_SILERO_START_THRESHOLD="${STREAMGLASS_SILERO_START_THRESHOLD:-0.65}"
    export STREAMGLASS_SILERO_CONTINUE_THRESHOLD="${STREAMGLASS_SILERO_CONTINUE_THRESHOLD:-0.35}"
    export STREAMGLASS_VAD_START_MS="${STREAMGLASS_SILERO_START_MS:-192}"
    exec "$RUNNER" >>"$LOG" 2>&1
}

start_canary() {
    [[ -x "$BIN" ]] || { echo "Missing canary binary: $BIN" >&2; exit 1; }
    [[ -s "$ENGINE" ]] || { echo "Missing Silero engine: $ENGINE" >&2; exit 1; }
    [[ -r "$ENV_FILE" ]] || { echo "Missing realtime environment: $ENV_FILE" >&2; exit 1; }
    [[ -x "$RUNNER" ]] || { echo "Missing realtime runner: $RUNNER" >&2; exit 1; }
    if ! production_is_enabled; then
        echo "$PRODUCTION_UNIT is not enabled; refusing a canary that cannot recover on reboot." >&2
        exit 1
    fi

    systemctl --user stop "$SHADOW_UNIT" 2>/dev/null || true
    systemctl --user stop "$CANARY_UNIT" 2>/dev/null || true
    wait_for_canary_unloaded
    systemctl --user stop "$PRODUCTION_UNIT"
    systemctl --user reset-failed "$PRODUCTION_UNIT" 2>/dev/null || true
    : >"$LOG"

    if ! systemd-run --user --unit "$CANARY_UNIT" --collect \
        "$SELF" _run; then
        rollback_canary
        exit 1
    fi

    local timeout_seconds="${STREAMGLASS_SILERO_CANARY_START_TIMEOUT_SECONDS:-180}"
    local deadline=$((SECONDS + timeout_seconds))
    while (( SECONDS < deadline )); do
        if grep -q '\[realtime\] ready .*vad=silero-tensorrt' "$LOG" 2>/dev/null; then
            echo "Silero TensorRT is temporarily controlling live segmentation."
            echo "Production remains enabled but paused; reboot or rollback restores RMS VAD."
            echo "Log: $LOG"
            return
        fi
        if ! systemctl --user is-active --quiet "$CANARY_UNIT"; then
            echo "Silero canary exited before becoming ready." >&2
            tail -n 80 "$LOG" >&2 || true
            rollback_canary
            exit 1
        fi
        sleep 1
    done

    echo "Silero canary did not become ready within ${timeout_seconds}s." >&2
    tail -n 80 "$LOG" >&2 || true
    rollback_canary
    exit 1
}

show_status() {
    printf 'production enabled: '
    systemctl --user is-enabled "$PRODUCTION_UNIT" 2>/dev/null || true
    printf 'production active:  '
    systemctl --user is-active "$PRODUCTION_UNIT" 2>/dev/null || true
    printf 'canary active:      '
    systemctl --user is-active "$CANARY_UNIT" 2>/dev/null || true
    if [[ -r "$LOG" ]]; then
        tail -n 12 "$LOG"
    fi
}

case "${1:-}" in
    start)
        start_canary
        ;;
    rollback|stop)
        rollback_canary
        ;;
    status)
        show_status
        ;;
    logs)
        tail -f "$LOG"
        ;;
    snapshot)
        tail -n 100 "$LOG"
        ;;
    _run)
        run_canary
        ;;
    *)
        echo "Usage: $0 {start|rollback|status|logs|snapshot}" >&2
        exit 2
        ;;
esac
