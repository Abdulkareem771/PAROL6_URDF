#!/bin/bash
# Builds and flashes the QuadTimer PWM diagnostic to Teensy 4.1
# using PlatformIO inside the parol6-ultimate container.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  QuadTimer Diagnostic Build & Flash"
echo "=========================================="

docker run -it --rm \
  --privileged \
  --device=/dev/bus/usb \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  parol6-ultimate:latest \
  bash -c "
    echo '==> Building diagnostic firmware...'
    pio run

    echo '==> Press the RESET button on the Teensy now to flash...'
    teensy_loader_cli --mcu=TEENSY41 -w .pio/build/teensy41/firmware.hex

    echo '==> Done! Open serial monitor at 115200 baud.'
  "
