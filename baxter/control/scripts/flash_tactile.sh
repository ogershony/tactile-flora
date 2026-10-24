#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# flash_tactile.sh -- build tactile-flora's teensy_fsr firmware (fingertip FSRs,
# read by bridge/tactile.py) and flash it to the Teensy 4.0.
#
#   ./scripts/flash_tactile.sh             # compile + upload
#   ./scripts/flash_tactile.sh --compile   # compile only (no Teensy needed)
#
# First run installs arduino-cli + PJRC's Teensy core under ~/.local / ~/.arduino15
# (no sudo). Uploading as non-root needs PJRC's udev rule, once:
#   sudo cp config/00-teensy.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger
# The sketch comes from the tactile-flora package installed in .venv-lerobot.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_VERSION=1.1.1
TEENSY_INDEX=https://www.pjrc.com/teensy/package_teensy_index.json
FQBN=teensy:avr:teensy40
export PATH="$HOME/.local/bin:$PATH"

SKETCH="$("$HERE/.venv-lerobot/bin/python" -c 'import os, tactile_flora as t
print(os.path.join(os.path.dirname(t.__file__), "baxter", "firmware", "teensy_fsr"))')"

if ! command -v arduino-cli >/dev/null; then
  echo "installing arduino-cli $CLI_VERSION to ~/.local/bin ..."
  mkdir -p "$HOME/.local/bin"
  curl -fsSL "https://downloads.arduino.cc/arduino-cli/arduino-cli_${CLI_VERSION}_Linux_64bit.tar.gz" \
    | tar -xz -C "$HOME/.local/bin" arduino-cli
fi
if ! arduino-cli core list --additional-urls "$TEENSY_INDEX" | grep -q '^teensy:avr'; then
  echo "installing the Teensy core (PJRC) ..."
  arduino-cli core update-index --additional-urls "$TEENSY_INDEX"
  arduino-cli core install teensy:avr --additional-urls "$TEENSY_INDEX"
fi

BUILD="$HERE/logs/teensy_fsr_build"
mkdir -p "$BUILD"
arduino-cli compile --fqbn "$FQBN" --additional-urls "$TEENSY_INDEX" --output-dir "$BUILD" "$SKETCH"
[[ "${1:-}" == "--compile" ]] && { echo "compiled: $BUILD"; exit 0; }

# the Teensy as arduino-cli's teensy discovery sees it (running a sketch, or in its bootloader)
PORT="$(arduino-cli board list --format json --additional-urls "$TEENSY_INDEX" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for p in d.get("detected_ports", d if isinstance(d, list) else []):
    port = p.get("port", {})
    if port.get("protocol") == "teensy" or any(b.get("fqbn", "").startswith("teensy:") for b in p.get("matching_boards", [])):
        print(port["address"], port.get("protocol", "")); break')"
if [[ -z "$PORT" ]]; then
  echo "no Teensy found: plug it in (data cable), or press its button to enter the bootloader" >&2
  exit 1
fi
read -r ADDR PROTO <<<"$PORT"
echo "uploading to $ADDR ($PROTO) ..."
arduino-cli upload --fqbn "$FQBN" --additional-urls "$TEENSY_INDEX" -p "$ADDR" -l "$PROTO" \
  --input-dir "$BUILD" "$SKETCH"
echo "flashed. check: .venv-lerobot/bin/python bridge/tactile.py --check 5"
