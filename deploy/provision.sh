#!/usr/bin/env bash
set -euo pipefail

MODE="check"
DATA_DEVICE=""
DATA_MOUNT="/mnt/streamglass-data"
INSTALL_PACKAGES=0
DISABLE_GUI=0

PACKAGES=(
  build-essential
  cmake
  ninja-build
  pkg-config
  git
  rsync
  jq
  libasound2-dev
  libpipewire-0.3-dev
  libspa-0.2-dev
  pipewire
  wireplumber
  bluez
  libbluetooth-dev
  libsamplerate0-dev
  libspeexdsp-dev
  libsndfile1-dev
  libssl-dev
  chrony
  python3-venv
  python3-pip
)

usage() {
  cat <<'EOF'
Usage: sudo ./deploy/provision.sh [options]

Options:
  --check                 Report readiness without changing the target (default)
  --apply                 Apply power, directory, mount, and service prerequisites
  --install-packages      Run apt-get for missing packages (requires a package source)
  --data-device PATH      Existing ext4 partition to mount at /mnt/streamglass-data
  --data-mount PATH       Override the data mount point
  --disable-gui           Switch the next boot to multi-user.target
  -h, --help              Show this help

The script never formats a block device.
EOF
}

while (($#)); do
  case "$1" in
    --check) MODE="check" ;;
    --apply) MODE="apply" ;;
    --install-packages) INSTALL_PACKAGES=1 ;;
    --data-device)
      shift
      DATA_DEVICE="${1:?missing value for --data-device}"
      ;;
    --data-mount)
      shift
      DATA_MOUNT="${1:?missing value for --data-mount}"
      ;;
    --disable-gui) DISABLE_GUI=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$MODE" == "apply" && "$(id -u)" -ne 0 ]]; then
  echo "--apply must run as root" >&2
  exit 1
fi

missing_packages=()
for package in "${PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${db:Status-Abbrev}' "$package" 2>/dev/null | grep -q '^ii'; then
    missing_packages+=("$package")
  fi
done

echo "target=$(hostname)"
echo "mode=$MODE"
echo "missing_packages=${missing_packages[*]:-none}"
nvpmodel -q 2>/dev/null || true

if [[ "$MODE" == "check" ]]; then
  if ((${#missing_packages[@]})); then
    exit 3
  fi
  exit 0
fi

if ((INSTALL_PACKAGES)) && ((${#missing_packages[@]})); then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing_packages[@]}"
fi

if [[ -n "$DATA_DEVICE" ]]; then
  if [[ ! -b "$DATA_DEVICE" ]]; then
    echo "not a block device: $DATA_DEVICE" >&2
    exit 1
  fi
  fs_type="$(blkid -s TYPE -o value "$DATA_DEVICE")"
  if [[ "$fs_type" != "ext4" ]]; then
    echo "$DATA_DEVICE uses $fs_type; expected an existing ext4 filesystem" >&2
    exit 1
  fi
  fs_uuid="$(blkid -s UUID -o value "$DATA_DEVICE")"
  install -d -m 0755 "$DATA_MOUNT"
  if ! grep -q "UUID=$fs_uuid" /etc/fstab; then
    printf 'UUID=%s %s ext4 defaults,noatime 0 2\n' "$fs_uuid" "$DATA_MOUNT" >> /etc/fstab
  fi
  mountpoint -q "$DATA_MOUNT" || mount "$DATA_MOUNT"
fi

if ! getent group streamglass >/dev/null; then
  groupadd --system streamglass
fi
if ! id streamglass >/dev/null 2>&1; then
  useradd --system --gid streamglass --home-dir /opt/streamglass --shell /usr/sbin/nologin streamglass
fi

if mountpoint -q "$DATA_MOUNT"; then
  install -d -o root -g streamglass -m 2775 "$DATA_MOUNT/streamglass"
  if [[ -e /opt/streamglass && ! -L /opt/streamglass ]]; then
    echo "/opt/streamglass exists and is not a symlink; refusing to replace it" >&2
    exit 1
  fi
  ln -sfn "$DATA_MOUNT/streamglass" /opt/streamglass
else
  install -d -o root -g streamglass -m 2775 /opt/streamglass
fi

for directory in releases models onnx engine-cache config golden logs staging cache tmp; do
  install -d -o root -g streamglass -m 2775 "/opt/streamglass/$directory"
done

nvpmodel -m 0
jetson_clocks

if ((DISABLE_GUI)); then
  systemctl set-default multi-user.target
fi

echo "provisioning complete; reboot is not required unless the default target changed"
