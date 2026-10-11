#!/usr/bin/env bash
set -u -o pipefail

DEVICE="${STREAMGLASS_BT_DEVICE:-41:42:84:69:36:F8}"
POLL_SECONDS="${STREAMGLASS_AUDIO_WAIT_SECONDS:-2}"
WAIT_TIMEOUT_SECONDS="${STREAMGLASS_AUDIO_WAIT_TIMEOUT_SECONDS:-45}"
PW_DUMP_TIMEOUT_SECONDS="${STREAMGLASS_PW_DUMP_TIMEOUT_SECONDS:-3}"
device_token="${DEVICE//:/_}"
RAW_SOURCE="bluez_input.${device_token}.0"
RAW_SINK="bluez_output.${device_token}.1"
announced=0

audio_endpoints_ready() {
    timeout "${PW_DUMP_TIMEOUT_SECONDS}s" pw-dump 2>/dev/null | python3 -c '
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

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
    if audio_endpoints_ready; then
        printf '[audio] PipeWire source=%s sink=%s are ready\n' \
            "$RAW_SOURCE" "$RAW_SINK"
        exit 0
    fi

    if [ "$announced" -eq 0 ]; then
        printf '[audio] waiting for PipeWire source=%s sink=%s\n' \
            "$RAW_SOURCE" "$RAW_SINK"
        announced=1
    fi
    sleep "$POLL_SECONDS"
done

printf '[audio] timed out after %ss waiting for PipeWire source=%s sink=%s\n' \
    "$WAIT_TIMEOUT_SECONDS" "$RAW_SOURCE" "$RAW_SINK" >&2
exit 1
