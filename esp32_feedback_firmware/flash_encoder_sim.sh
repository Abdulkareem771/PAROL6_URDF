#!/bin/bash
# Build and flash the MT6816 encoder PWM simulator to ESP32.
# Usage: ./flash_encoder_sim.sh [/dev/ttyUSBx]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=${1:-/dev/ttyUSB0}

echo "=========================================="
echo "  MT6816 Encoder Simulator - Build & Flash"
echo "=========================================="
echo "Port: $PORT"
echo ""

if [ ! -e "$PORT" ]; then
    echo "Port $PORT not found. Available:"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  None"
    exit 1
fi

sudo chmod 666 "$PORT" 2>/dev/null || true

docker run -it --rm \
  --device="$PORT" \
  --privileged \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  parol6-ultimate:latest \
  bash -c "
    echo '==> Loading ESP-IDF...'
    . /opt/esp-idf/export.sh

    echo '==> Setting target...'
    idf.py set-target esp32

    echo '==> Building...'
    idf.py build

    echo '==> Flashing + monitor (Ctrl+] to exit)...'
    idf.py -p $PORT flash monitor
  "
