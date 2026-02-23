#!/bin/bash
set -e

echo "=========================================="
echo "  Teensy 4.1 Build & Flash (parol6_firmware)"
echo "=========================================="

echo "Starting Docker container for build and flash..."
docker run -it --rm \
  --device=/dev/bus/usb \
  --privileged \
  -v $(pwd):/workspace \
  -w /workspace/parol6_firmware \
  parol6-ultimate:latest \
  bash -c "
    echo '==> Building firmware with PlatformIO...'
    pio run
    
    echo '==> Flashing Teensy...'
    # Use -w to wait for device reprogramming (button press might be needed initially)
    teensy_loader_cli --mcu=TEENSY41 -w .pio/build/teensy41/firmware.hex
  "

echo "Done.".................................................................................................
