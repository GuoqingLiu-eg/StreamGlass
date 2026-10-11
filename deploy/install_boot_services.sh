#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_USER="${STREAMGLASS_TARGET_USER:-jetson}"
SYSTEM_UNIT_SOURCE="$ROOT_DIR/deploy/systemd/streamglass-bluetooth.service"
SYSTEM_UNIT_TARGET="/etc/systemd/system/streamglass-bluetooth.service"
BOOTSTRAP_DIR="/etc/streamglass"
BOOTSTRAP_ENV="$BOOTSTRAP_DIR/bootstrap.env"

if [ "$(id -u)" -ne 0 ]; then
    echo "run this script with sudo" >&2
    exit 1
fi

target_uid=$(id -u "$TARGET_USER")
target_gid=$(id -g "$TARGET_USER")
target_home=$(getent passwd "$TARGET_USER" | cut -d: -f6)
runtime_dir="/run/user/$target_uid"

run_user() {
    setpriv --reuid="$target_uid" --regid="$target_gid" --init-groups env \
        "HOME=$target_home" \
        "XDG_RUNTIME_DIR=$runtime_dir" \
        "DBUS_SESSION_BUS_ADDRESS=unix:path=$runtime_dir/bus" "$@"
}

install -d -m 0755 "$BOOTSTRAP_DIR"
printf 'STREAMGLASS_PROJECT_DIR=%s\nSTREAMGLASS_TARGET_USER=%s\n' \
    "$ROOT_DIR" "$TARGET_USER" >"$BOOTSTRAP_ENV"
chmod 0644 "$BOOTSTRAP_ENV"
install -m 0644 "$SYSTEM_UNIT_SOURCE" "$SYSTEM_UNIT_TARGET"

loginctl enable-linger "$TARGET_USER"
systemctl start "user@${target_uid}.service"
for _ in $(seq 1 30); do
    [ -S "$runtime_dir/bus" ] && break
    sleep 1
done
[ -S "$runtime_dir/bus" ] || {
    echo "headless user D-Bus did not become ready" >&2
    exit 1
}

systemctl daemon-reload
systemctl enable streamglass-bluetooth.service
run_user "$ROOT_DIR/deploy/realtime_service.sh" install
# The system Bluetooth orchestrator starts realtime only after both raw HFP
# endpoints are stable. Keep the user unit installed but out of default.target.
run_user systemctl --user disable --now streamglass-realtime.service \
    2>/dev/null || true
run_user systemctl --user enable --now streamglass-web.service
systemctl restart --no-block streamglass-bluetooth.service

echo "StreamGlass boot services enabled:"
echo "  system: streamglass-bluetooth.service"
echo "  user:   streamglass-realtime.service (started after Bluetooth readiness)"
