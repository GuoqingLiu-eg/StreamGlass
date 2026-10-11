#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTROLLER="${STREAMGLASS_BT_CONTROLLER:-04:7F:0E:11:36:75}"
DEVICE="${STREAMGLASS_BT_DEVICE:-41:42:84:69:36:F8}"
ALIAS="${STREAMGLASS_BT_ALIAS:-ZH-ZCGF}"
TARGET_USER="${STREAMGLASS_TARGET_USER:-jetson}"
RETRY_SECONDS="${STREAMGLASS_BT_RETRY_SECONDS:-4}"
MONITOR_SECONDS="${STREAMGLASS_BT_MONITOR_SECONDS:-3}"
MONITOR="${STREAMGLASS_BT_MONITOR:-0}"
HFP_PROFILE="${STREAMGLASS_BT_HFP_PROFILE:-261}"
PW_DUMP_TIMEOUT_SECONDS="${STREAMGLASS_PW_DUMP_TIMEOUT_SECONDS:-3}"
DIRECT_CONNECT_ATTEMPTS="${STREAMGLASS_BT_DIRECT_CONNECT_ATTEMPTS:-3}"
CONNECT_TIMEOUT_SECONDS="${STREAMGLASS_BT_CONNECT_TIMEOUT_SECONDS:-12}"
DIRECT_ENDPOINT_WAIT_SECONDS="${STREAMGLASS_BT_DIRECT_ENDPOINT_WAIT_SECONDS:-4}"
BREDR_SCAN_SECONDS="${STREAMGLASS_BT_BREDR_SCAN_SECONDS:-12}"
ENDPOINT_WAIT_SECONDS="${STREAMGLASS_BT_ENDPOINT_WAIT_SECONDS:-20}"
RECOVERY_FAILURES="${STREAMGLASS_BT_RECOVERY_FAILURES:-2}"
STABLE_PROBES="${STREAMGLASS_BT_STABLE_PROBES:-2}"
PAUSE_PROBE_SECONDS="${STREAMGLASS_BT_PAUSE_PROBE_SECONDS:-15}"
PAUSE_PROBE_TIMEOUT_SECONDS="${STREAMGLASS_BT_PAUSE_PROBE_TIMEOUT_SECONDS:-2}"
device_token="${DEVICE//:/_}"
RAW_SOURCE="bluez_input.${device_token}.0"
RAW_SINK="bluez_output.${device_token}.1"
AEC_SOURCE="streamglass_aec_source"
AEC_SINK="streamglass_aec_sink"
AEC_CAPTURE="streamglass_aec_capture"
AEC_PLAYBACK="streamglass_aec_playback"

if [ "$(id -u)" -ne 0 ]; then
    echo "run this script with sudo" >&2
    exit 1
fi

target_uid=$(id -u "$TARGET_USER")
target_gid=$(id -g "$TARGET_USER")
target_group=$(id -gn "$TARGET_USER")
target_home=$(getent passwd "$TARGET_USER" | cut -d: -f6)
runtime_dir="/run/user/$target_uid"
control_socket="$runtime_dir/streamglass/control.sock"
restore_paused=0
restore_language=""
last_pause_probe_epoch=0
pause_probe_warning_emitted=0

run_user() {
    setpriv --reuid="$target_uid" --regid="$target_gid" --init-groups env \
        "HOME=$target_home" \
        "XDG_RUNTIME_DIR=$runtime_dir" \
        "DBUS_SESSION_BUS_ADDRESS=unix:path=$runtime_dir/bus" "$@"
}

controller_available() {
    timeout 5s bluetoothctl list 2>/dev/null | grep -Fq "$CONTROLLER"
}

wait_for_controller() {
    until controller_available; do
        printf '[barrot] waiting for USB controller %s; Ctrl-C to stop\n' \
            "$CONTROLLER"
        modprobe --ignore-install btusb 2>/dev/null || true
        rfkill unblock bluetooth 2>/dev/null || true
        systemctl start bluetooth.service
        sleep "$RETRY_SECONDS"
    done
}

user_bus_available() {
    [ -S "$runtime_dir/bus" ]
}

device_info() {
    printf 'select %s\ninfo %s\nquit\n' "$CONTROLLER" "$DEVICE" |
        timeout 5s bluetoothctl 2>/dev/null
}

device_connected() {
    device_info | grep -Fq "Connected: yes"
}

pipewire_dump() {
    run_user timeout "${PW_DUMP_TIMEOUT_SECONDS}s" pw-dump 2>/dev/null
}

audio_card_id() {
    pipewire_dump | python3 -c '
import json
import sys

alias = sys.argv[1]
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
    if not isinstance(props, dict):
        continue
    if props.get("device.alias") == alias:
        obj_id = obj.get("id")
        if obj_id is not None:
            print(obj_id)
        break
' "$ALIAS"
}

audio_endpoint_available() {
    pipewire_dump | python3 -c '
import json
import sys

expected = set(sys.argv[1:])
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

actual = {
    properties(obj).get("node.name")
    for obj in objects
}
raise SystemExit(0 if expected <= actual else 1)
' "$RAW_SOURCE" "$RAW_SINK"
}

aec_endpoint_available() {
    pipewire_dump | python3 -c '
import json
import sys

expected = set(sys.argv[1:])
payload = sys.stdin.read().lstrip()
if not payload:
    raise SystemExit(1)
try:
    objects, _ = json.JSONDecoder().raw_decode(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
if not isinstance(objects, list):
    raise SystemExit(1)

actual = set()
for obj in objects:
    if not isinstance(obj, dict):
        continue
    info = obj.get("info")
    if not isinstance(info, dict):
        continue
    props = info.get("props")
    if isinstance(props, dict):
        actual.add(props.get("node.name"))
raise SystemExit(0 if expected <= actual else 1)
' "$AEC_SOURCE" "$AEC_SINK" "$AEC_CAPTURE" "$AEC_PLAYBACK"
}

pipewire_graph_ready() {
    pipewire_dump | python3 -c '
import json
import sys

raw_source, raw_sink, aec_capture, aec_playback, aec_source, aec_sink = (
    sys.argv[1:]
)
payload = sys.stdin.read().lstrip()
if not payload:
    raise SystemExit(1)
try:
    objects, _ = json.JSONDecoder().raw_decode(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
if not isinstance(objects, list):
    raise SystemExit(1)

nodes = {}
links = set()
for obj in objects:
    if not isinstance(obj, dict):
        continue
    info = obj.get("info")
    if not isinstance(info, dict):
        continue
    props = info.get("props")
    if not isinstance(props, dict):
        props = {}
    if obj.get("type") == "PipeWire:Interface:Node":
        name = props.get("node.name")
        if name:
            nodes[name] = obj.get("id")
    elif obj.get("type") == "PipeWire:Interface:Link":
        output_node = info.get("output-node-id", props.get("link.output.node"))
        input_node = info.get("input-node-id", props.get("link.input.node"))
        if output_node is not None and input_node is not None:
            links.add((str(output_node), str(input_node)))

required = {
    raw_source, raw_sink, aec_capture, aec_playback, aec_source, aec_sink
}
if not required <= nodes.keys():
    raise SystemExit(1)

def linked(output_name, input_name):
    return (str(nodes[output_name]), str(nodes[input_name])) in links

capture_bound = linked(raw_source, aec_capture)
playback_bound = linked(aec_playback, raw_sink)
runtime_capture_bound = any(
    output_node == str(nodes[aec_source]) for output_node, _ in links
)
runtime_playback_bound = any(
    input_node == str(nodes[aec_sink]) for _, input_node in links
)
raise SystemExit(
    0 if capture_bound and playback_bound and
    runtime_capture_bound and runtime_playback_bound else 1
)
' "$RAW_SOURCE" "$RAW_SINK" "$AEC_CAPTURE" "$AEC_PLAYBACK" \
    "$AEC_SOURCE" "$AEC_SINK"
}

repair_aec_links() {
    local repairs=""
    local output_port=""
    local input_port=""
    local label=""

    repairs=$(pipewire_dump | python3 -c '
import json
import sys

raw_source, raw_sink, aec_capture, aec_playback = sys.argv[1:]
payload = sys.stdin.read().lstrip()
if not payload:
    raise SystemExit(1)
try:
    objects, _ = json.JSONDecoder().raw_decode(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
if not isinstance(objects, list):
    raise SystemExit(1)

nodes = {}
ports = {}
links = set()
for obj in objects:
    if not isinstance(obj, dict):
        continue
    info = obj.get("info")
    if not isinstance(info, dict):
        continue
    props = info.get("props")
    if not isinstance(props, dict):
        props = {}
    object_type = obj.get("type")
    if object_type == "PipeWire:Interface:Node":
        name = props.get("node.name")
        if name:
            nodes[name] = obj.get("id")
    elif object_type == "PipeWire:Interface:Port":
        node_id = props.get("node.id")
        direction = info.get("direction", props.get("port.direction"))
        if node_id is not None and direction:
            ports.setdefault((str(node_id), direction), []).append(obj.get("id"))
    elif object_type == "PipeWire:Interface:Link":
        output_node = info.get("output-node-id", props.get("link.output.node"))
        input_node = info.get("input-node-id", props.get("link.input.node"))
        if output_node is not None and input_node is not None:
            links.add((str(output_node), str(input_node)))

def first_port(node_name, directions):
    node_id = nodes.get(node_name)
    if node_id is None:
        return None
    for direction in directions:
        candidates = ports.get((str(node_id), direction), [])
        if candidates:
            return candidates[0]
    return None

for output_name, input_name, label in (
    (raw_source, aec_capture, "capture"),
    (aec_playback, raw_sink, "playback"),
):
    output_node = nodes.get(output_name)
    input_node = nodes.get(input_name)
    if output_node is None or input_node is None:
        continue
    if (str(output_node), str(input_node)) in links:
        continue
    output_port = first_port(output_name, ("output", "out"))
    input_port = first_port(input_name, ("input", "in"))
    if output_port is not None and input_port is not None:
        print(output_port, input_port, label)
' "$RAW_SOURCE" "$RAW_SINK" "$AEC_CAPTURE" "$AEC_PLAYBACK") || true

    [ -n "$repairs" ] || return 0
    while read -r output_port input_port label; do
        [ -n "$output_port" ] || continue
        printf '[barrot] repairing AEC %s link ports=%s->%s\n' \
            "$label" "$output_port" "$input_port"
        run_user timeout 3s pw-link -L "$output_port" "$input_port" \
            >/dev/null 2>&1 || true
    done <<<"$repairs"
}

wait_for_hot_rebind() {
    local attempt=0
    for attempt in $(seq 1 30); do
        pipewire_graph_ready && return 0
        if ((attempt % 5 == 0)); then
            repair_aec_links
        fi
        sleep 0.1
    done
    repair_aec_links
    sleep 0.2
    pipewire_graph_ready
}

runtime_state() {
    [ -S "$control_socket" ] || return 1
    run_user timeout 2s python3 - "$control_socket" <<'PY'
import json
import socket
import sys

path = sys.argv[1]
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.settimeout(1.0)
try:
    client.connect(path)
    client.sendall(b"status\n")
    payload = b""
    while b"\n" not in payload:
        chunk = client.recv(4096)
        if not chunk:
            break
        payload += chunk
finally:
    client.close()

status = json.loads(payload.decode("utf-8"))
paused = "1" if status.get("paused") is True else "0"
language = (
    status.get("requestedLanguage")
    if status.get("languagePending") is True
    else status.get("targetLanguage")
)
if language not in {
    "cmn", "eng", "fra", "deu", "spa", "ita",
    "jpn", "kor", "rus", "por", "arb", "vie",
}:
    language = ""
print(paused, language)
PY
}

update_runtime_restore_state() {
    local state=""
    local paused=""
    local language=""

    state=$(runtime_state 2>/dev/null || true)
    if [ -n "$state" ]; then
        read -r paused language <<<"$state"
        if [ "$paused" = "0" ] || [ "$paused" = "1" ]; then
            restore_paused="$paused"
        fi
        restore_language="$language"
    fi
}

capture_runtime_restore_state() {
    update_runtime_restore_state
    printf '[barrot] preserving runtime state: paused=%s language=%s\n' \
        "$restore_paused" "${restore_language:-default}"
}

controller_hci() {
    local address_file=""
    local address=""
    for address_file in /sys/class/bluetooth/hci*/address; do
        [ -r "$address_file" ] || continue
        address=$(tr -d '[:space:]' <"$address_file")
        if [ "${address^^}" = "${CONTROLLER^^}" ]; then
            basename "$(dirname "$address_file")"
            return 0
        fi
    done
    return 1
}

send_non_hfp_probe() {
    local hci=""
    command -v l2ping >/dev/null 2>&1 || return 2
    hci=$(controller_hci) || return 2
    timeout "$((PAUSE_PROBE_TIMEOUT_SECONDS + 1))s" \
        l2ping -i "$hci" -c 1 -t "$PAUSE_PROBE_TIMEOUT_SECONDS" \
        "$DEVICE" >/dev/null 2>&1
}

maybe_send_pause_probe() {
    local now=""

    now=$(date +%s)
    if ((now - last_pause_probe_epoch < PAUSE_PROBE_SECONDS)); then
        return 0
    fi
    last_pause_probe_epoch=$now

    [ "$restore_paused" = "1" ] || return 0
    device_connected || return 0

    # L2CAP echo is a control-plane packet, not HFP audio. Some headset
    # firmware does not answer echoes even though it receives them, so failure
    # is diagnostic only and must never tear down a healthy audio session.
    if send_non_hfp_probe; then
        if [ "$pause_probe_warning_emitted" = "1" ]; then
            echo "[barrot] non-HFP pause probe responding again"
        fi
        pause_probe_warning_emitted=0
    elif [ "$pause_probe_warning_emitted" = "0" ]; then
        echo "[barrot] non-HFP pause probe was not acknowledged; endpoint health remains authoritative"
        pause_probe_warning_emitted=1
    fi
}

wait_for_stable_endpoints() {
    local timeout_seconds="${1:-$ENDPOINT_WAIT_SECONDS}"
    local deadline=$((SECONDS + timeout_seconds))
    local stable=0
    while ((SECONDS < deadline)); do
        if controller_available && device_connected && audio_endpoint_available; then
            stable=$((stable + 1))
            if ((stable >= STABLE_PROBES)); then
                return 0
            fi
        else
            stable=0
        fi
        sleep 1
    done
    printf '[barrot] endpoints did not remain stable for %d probes within %ss\n' \
        "$STABLE_PROBES" "$timeout_seconds" >&2
    return 1
}

restart_realtime_session() {
    local restart_status=0
    local language="${restore_language:-}"
    local environment=("STREAMGLASS_START_PAUSED=$restore_paused")
    if [ -n "$language" ]; then
        environment+=("STREAMGLASS_RESTORE_TARGET_LANGUAGE=$language")
    fi

    run_user systemctl --user set-environment "${environment[@]}" || return 1
    run_user systemctl --user restart streamglass-realtime.service || \
        restart_status=$?
    run_user systemctl --user unset-environment \
        STREAMGLASS_START_PAUSED STREAMGLASS_RESTORE_TARGET_LANGUAGE || true

    if [ "$restart_status" -ne 0 ]; then
        printf '[barrot] realtime restart failed with status=%d\n' \
            "$restart_status" >&2
        return "$restart_status"
    fi
    printf '[barrot] rebuilt AEC, capture, playback and touch control (paused=%s language=%s)\n' \
        "$restore_paused" "${language:-default}"
}

resident_model_ready() {
    run_user systemctl --user is-active --quiet \
        streamglass-realtime.service || return 1
    runtime_state >/dev/null 2>&1 || return 1
    aec_endpoint_available
}

start_realtime_session() {
    if run_user systemctl --user is-active --quiet \
        streamglass-realtime.service; then
        echo "[barrot] realtime service is already active"
        return 0
    fi
    echo "[barrot] audio endpoints are ready; starting realtime service"
    run_user systemctl --user start streamglass-realtime.service
}

log_bluetoothctl_output() {
    local output="$1"
    local line=""
    [ -n "$output" ] || return 0
    while IFS= read -r line; do
        printf '[barrot][bluetoothctl] %s\n' "$line"
    done <<<"$output"
}

bluez_adapter_path() {
    local path=""
    local property=""
    local property_type=""
    local address=""

    while IFS= read -r path; do
        [[ "$path" =~ ^/org/bluez/hci[0-9]+$ ]] || continue
        property=$(busctl get-property org.bluez "$path" \
            org.bluez.Adapter1 Address 2>/dev/null || true)
        [ -n "$property" ] || continue
        read -r property_type address <<<"$property"
        address="${address#\"}"
        address="${address%\"}"
        if [ "$property_type" = "s" ] && \
            [ "${address^^}" = "${CONTROLLER^^}" ]; then
            printf '%s\n' "$path"
            return 0
        fi
    done < <(busctl tree org.bluez --list 2>/dev/null)
    return 1
}

bluez_device_path() {
    local adapter_path=""
    adapter_path=$(bluez_adapter_path) || return 1
    printf '%s/dev_%s\n' "$adapter_path" "$device_token"
}

connect_once() {
    local label="$1"
    local device_path=""
    local output=""
    local status=0

    printf '[barrot] direct connect %s: device=%s controller=%s timeout=%ss\n' \
        "$label" "$DEVICE" "$CONTROLLER" "$CONNECT_TIMEOUT_SECONDS"
    # Controller setup is synchronous and may exit immediately. The actual
    # connect must not share that session: feeding `quit` after `connect`
    # cancels bluetoothctl before BR/EDR paging has had time to finish.
    output=$(printf 'select %s\npower on\nagent NoInputNoOutput\ndefault-agent\npairable on\ntrust %s\nquit\n' \
        "$CONTROLLER" "$DEVICE" |
        timeout 8s bluetoothctl 2>&1) || status=$?
    log_bluetoothctl_output "$output"

    device_path=$(bluez_device_path || true)
    if [ -z "$device_path" ]; then
        printf '[barrot] cannot resolve BlueZ device path for controller=%s device=%s\n' \
            "$CONTROLLER" "$DEVICE" >&2
        return 1
    fi
    printf '[barrot] waiting for BlueZ Device1.Connect on %s\n' "$device_path"
    status=0
    output=$(timeout "${CONNECT_TIMEOUT_SECONDS}s" \
        busctl call org.bluez "$device_path" \
        org.bluez.Device1 Connect 2>&1) || status=$?
    log_bluetoothctl_output "$output"

    if device_connected; then
        printf '[barrot] direct connect %s reached BlueZ Connected=yes\n' \
            "$label"
        return 0
    fi
    printf '[barrot] direct connect %s failed status=%d\n' \
        "$label" "$status" >&2
    return 1
}

activate_hfp_profile() {
    local card_id=""

    card_id=$(audio_card_id || true)
    [ -n "$card_id" ] || return 1
    if run_user timeout 5s wpctl set-profile "$card_id" "$HFP_PROFILE" \
        >/dev/null 2>&1; then
        printf '[barrot] requested HFP profile=%s on PipeWire card=%s\n' \
            "$HFP_PROFILE" "$card_id"
        return 0
    fi
    printf '[barrot] failed to request HFP profile=%s on PipeWire card=%s\n' \
        "$HFP_PROFILE" "$card_id" >&2
    return 1
}

wait_for_audio_endpoints() {
    local timeout_seconds="$1"
    local deadline=$((SECONDS + timeout_seconds))
    local next_profile_attempt=0

    printf '[barrot] waiting up to %ss for source=%s sink=%s\n' \
        "$timeout_seconds" "$RAW_SOURCE" "$RAW_SINK"
    while ((SECONDS < deadline)); do
        if device_connected && audio_endpoint_available; then
            echo "[barrot] BlueZ input/output endpoints are ready"
            return 0
        fi
        if device_connected && ((SECONDS >= next_profile_attempt)); then
            activate_hfp_profile || true
            next_profile_attempt=$((SECONDS + 3))
        fi
        sleep 1
    done
    printf '[barrot] audio endpoints were not ready within %ss\n' \
        "$timeout_seconds" >&2
    return 1
}

bredr_discovery() {
    local output=""
    local status=0

    printf '[barrot] direct paging exhausted; starting BR/EDR discovery for %ss\n' \
        "$BREDR_SCAN_SECONDS"
    output=$(
        {
            printf 'select %s\npower on\nscan bredr\n' "$CONTROLLER"
            sleep "$BREDR_SCAN_SECONDS"
            printf 'scan off\nquit\n'
        } | timeout "$((BREDR_SCAN_SECONDS + 8))s" bluetoothctl 2>&1
    ) || status=$?
    log_bluetoothctl_output "$output"

    # Stop discovery defensively even if the interactive scan timed out.
    printf 'select %s\nscan off\nquit\n' "$CONTROLLER" |
        timeout 5s bluetoothctl >/dev/null 2>&1 || true
    if [ "$status" -ne 0 ]; then
        printf '[barrot] BR/EDR discovery ended with status=%d; trying the post-scan connect anyway\n' \
            "$status" >&2
    else
        echo "[barrot] BR/EDR discovery stopped"
    fi
}

connect_until_ready() {
    local attempt=0

    wait_for_controller
    if device_connected && audio_endpoint_available; then
        echo "[barrot] existing BlueZ input/output endpoints are healthy"
        return 0
    fi

    for attempt in $(seq 1 "$DIRECT_CONNECT_ATTEMPTS"); do
        if connect_once "$attempt/$DIRECT_CONNECT_ATTEMPTS"; then
            if wait_for_audio_endpoints "$DIRECT_ENDPOINT_WAIT_SECONDS"; then
                return 0
            fi
        fi
        if ((attempt < DIRECT_CONNECT_ATTEMPTS)); then
            sleep "$RETRY_SECONDS"
        fi
    done

    bredr_discovery
    if ! connect_once "post-scan"; then
        echo "[barrot] post-scan direct connect failed; yielding to systemd retry" >&2
        return 1
    fi
    if wait_for_audio_endpoints "$ENDPOINT_WAIT_SECONDS"; then
        return 0
    fi
    echo "[barrot] bounded Bluetooth recovery exhausted; yielding to systemd retry" >&2
    return 1
}

echo "[barrot] loading btusb for the 33fa:0012 adapter"
modprobe --ignore-install btusb
rfkill unblock bluetooth 2>/dev/null || true
systemctl start bluetooth.service

wireplumber_config_changed=0
wireplumber_config_target="$target_home/.config/wireplumber/bluetooth.lua.d/51-streamglass-bluez.lua"
install -d -o "$TARGET_USER" -g "$target_group" -m 0700 \
    "$target_home/.config/wireplumber/bluetooth.lua.d"
if ! cmp -s "$ROOT_DIR/config/wireplumber/51-streamglass-bluez.lua" \
    "$wireplumber_config_target"; then
    install -o "$TARGET_USER" -g "$target_group" -m 0644 \
        "$ROOT_DIR/config/wireplumber/51-streamglass-bluez.lua" \
        "$wireplumber_config_target"
    wireplumber_config_changed=1
fi

# A lingering user manager keeps D-Bus and PipeWire alive after SSH exits. The
# translation unit is enabled separately by install_boot_services.sh.
loginctl enable-linger "$TARGET_USER"
systemctl start "user@${target_uid}.service"
for _ in $(seq 1 30); do
    user_bus_available && break
    sleep 1
done
if ! user_bus_available; then
    echo "[barrot] headless user D-Bus did not become ready" >&2
    exit 1
fi

# GDM owns the active seat before desktop login and can capture the Bluetooth
# audio profiles. Release only its WirePlumber instance for this headless run.
if gdm_uid=$(id -u gdm 2>/dev/null); then
    gdm_runtime="/run/user/$gdm_uid"
    if [ -S "$gdm_runtime/bus" ]; then
        runuser -u gdm -- env \
            "XDG_RUNTIME_DIR=$gdm_runtime" \
            "DBUS_SESSION_BUS_ADDRESS=unix:path=$gdm_runtime/bus" \
            systemctl --user stop wireplumber.service >/dev/null 2>&1 || true
    fi
fi

run_user systemctl --user start \
    pipewire.service pipewire-pulse.service wireplumber.service
if [ "$wireplumber_config_changed" = "1" ]; then
    run_user systemctl --user restart wireplumber.service
fi
run_user systemctl --user is-active --quiet \
    pipewire.service pipewire-pulse.service wireplumber.service

if ! connect_until_ready; then
    echo "[barrot] startup recovery failed; exiting for systemd backoff" >&2
    exit 1
fi
if ! wait_for_stable_endpoints "$ENDPOINT_WAIT_SECONDS"; then
    echo "[barrot] startup endpoints were unstable; exiting for systemd backoff" >&2
    exit 1
fi

printf '[barrot] headless ready: %s owns source=%s and sink=%s in the %s PipeWire session\n' \
    "$ALIAS" "$RAW_SOURCE" "$RAW_SINK" "$TARGET_USER"

if [ "$MONITOR" != "1" ]; then
    exit 0
fi

start_realtime_session
printf '[barrot] monitoring the controller and full-duplex audio endpoints\n'
consecutive_failures=0
while sleep "$MONITOR_SECONDS"; do
    update_runtime_restore_state
    maybe_send_pause_probe
    if controller_available && device_connected && audio_endpoint_available; then
        consecutive_failures=0
        if resident_model_ready && ! pipewire_graph_ready; then
            if wait_for_hot_rebind; then
                echo "[barrot] repaired resident PipeWire graph without reloading the model"
            else
                echo "[barrot] resident model is ready, but the audio graph has not rebound yet; keeping engines resident" >&2
            fi
        fi
        continue
    fi
    consecutive_failures=$((consecutive_failures + 1))
    if ((consecutive_failures < RECOVERY_FAILURES)); then
        printf '[barrot] endpoint health probe failed (%d/%d); waiting for confirmation\n' \
            "$consecutive_failures" "$RECOVERY_FAILURES"
        continue
    fi

    capture_runtime_restore_state
    printf '[barrot] endpoint lost; restoring the headless Bluetooth session\n'
    if ! connect_until_ready; then
        echo "[barrot] bounded reconnect failed; exiting for systemd backoff" >&2
        exit 1
    fi
    if ! wait_for_stable_endpoints "$ENDPOINT_WAIT_SECONDS"; then
        echo "[barrot] recovered endpoints remained unstable; exiting for systemd backoff" >&2
        exit 1
    fi
    # Normal recovery is a hot PipeWire rebind. The model process and all
    # TensorRT engines stay resident, so translation is available as soon as
    # the raw HFP nodes return. A full service restart is only a cold fallback
    # when the model process or its AEC virtual nodes have actually died.
    if resident_model_ready; then
        if wait_for_hot_rebind; then
            echo "[barrot] hot rebound complete; resident model session preserved"
        else
            echo "[barrot] raw endpoints recovered, but AEC is still rebinding; resident model session preserved" >&2
        fi
    else
        echo "[barrot] resident session is unavailable; using cold restart fallback"
        restart_realtime_session || true
    fi
    consecutive_failures=0
    printf '[barrot] endpoint restored and realtime session rebound\n'
done
