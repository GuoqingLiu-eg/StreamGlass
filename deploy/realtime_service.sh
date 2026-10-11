#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT_SOURCE="$ROOT_DIR/deploy/systemd/streamglass-realtime.service"
WEB_UNIT_SOURCE="$ROOT_DIR/deploy/systemd/streamglass-web.service"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_TARGET="$UNIT_DIR/streamglass-realtime.service"
WEB_UNIT_TARGET="$UNIT_DIR/streamglass-web.service"
ENV_DIR="$HOME/.config/streamglass"
ENV_FILE="$ENV_DIR/realtime.env"
LOG_FILE="/dev/shm/streamglass-realtime.log"
ALIAS="${STREAMGLASS_BT_ALIAS:-ZH-ZCGF}"
FINGERPRINT="${STREAMGLASS_PLATFORM_FINGERPRINT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' "$ROOT_DIR/config/platform.lock.json")}"
SILERO_ENGINE="${STREAMGLASS_SILERO_PRIMARY_ENGINE:-/opt/streamglass/engine-cache/$FINGERPRINT/silero_vad_16k_fp16.engine}"
SILERO_START_THRESHOLD="${STREAMGLASS_SILERO_START_THRESHOLD:-0.65}"
SILERO_CONTINUE_THRESHOLD="${STREAMGLASS_SILERO_CONTINUE_THRESHOLD:-0.35}"

install_manual_unit() {
    [[ -s "$SILERO_ENGINE" ]] || {
        echo "Silero TensorRT engine is missing: $SILERO_ENGINE" >&2
        exit 1
    }
    install -d -m 0700 "$UNIT_DIR" "$ENV_DIR"
    install -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"
    install -m 0644 "$WEB_UNIT_SOURCE" "$WEB_UNIT_TARGET"
    cat >"$ENV_FILE" <<EOF
STREAMGLASS_PROJECT_DIR=$ROOT_DIR
STREAMGLASS_BT_ALIAS=$ALIAS
STREAMGLASS_TARGET_LANGUAGE=eng
STREAMGLASS_AEC=1
STREAMGLASS_HALF_DUPLEX=0
STREAMGLASS_SILERO_PRIMARY_ENGINE=$SILERO_ENGINE
STREAMGLASS_SILERO_START_THRESHOLD=$SILERO_START_THRESHOLD
STREAMGLASS_SILERO_CONTINUE_THRESHOLD=$SILERO_CONTINUE_THRESHOLD
STREAMGLASS_VAD_START_RMS=0.030
STREAMGLASS_VAD_CONTINUE_RMS=0.016
STREAMGLASS_VAD_START_MS=192
STREAMGLASS_VAD_PREROLL_MS=192
STREAMGLASS_VAD_SILENCE_MS=544
STREAMGLASS_UTTERANCE_HARD_CAP_MS=12800
STREAMGLASS_HFP_BUTTON_CONTROL=1
STREAMGLASS_HFP_BUTTON_RETRY_MS=3500
STREAMGLASS_AVRCP_CONTROL=0
EOF
    chmod 0600 "$ENV_FILE"
    systemctl --user daemon-reload
}

enable_runtime() {
    install_manual_unit
    systemctl --user enable streamglass-realtime.service streamglass-web.service
    systemctl --user start streamglass-web.service
    systemctl --user start --no-block streamglass-realtime.service
    echo "StreamGlass realtime and web services are enabled for headless startup."
}

disable_runtime() {
    systemctl --user disable --now \
        streamglass-realtime.service streamglass-web.service \
        2>/dev/null || true
    echo "StreamGlass realtime and web services are disabled."
}

start_runtime() {
    install_manual_unit
    systemctl --user start streamglass-web.service
    if ! wpctl status 2>/dev/null | grep -Fq "$ALIAS"; then
        echo "Bluetooth audio is not ready." >&2
        echo "Run: sudo $ROOT_DIR/deploy/connect_barrot_glasses.sh" >&2
        exit 1
    fi
    rm -f "$LOG_FILE"
    systemctl --user start --no-block streamglass-realtime.service
    for _ in $(seq 1 120); do
        if grep -q '\[realtime\] ready' "$LOG_FILE" 2>/dev/null; then
            echo "StreamGlass realtime service is ready."
            return 0
        fi
        sleep 1
    done
    systemctl --user status streamglass-realtime.service --no-pager -n 20 || true
    echo "Timed out waiting for the realtime engine." >&2
    exit 1
}

case "${1:-}" in
    install)
        install_manual_unit
        echo "StreamGlass realtime and web service units are installed."
        ;;
    start)
        start_runtime
        ;;
    stop)
        systemctl --user stop streamglass-realtime.service
        ;;
    restart)
        systemctl --user stop streamglass-realtime.service 2>/dev/null || true
        start_runtime
        ;;
    enable)
        enable_runtime
        ;;
    disable)
        disable_runtime
        ;;
    status)
        systemctl --user status streamglass-realtime.service --no-pager -n 30
        systemctl --user status streamglass-web.service --no-pager -n 20
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 {install|start|stop|restart|enable|disable|status|logs}" >&2
        exit 2
        ;;
esac
