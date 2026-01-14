#!/bin/bash
# Quick build and flash script for ESP32 benchmark firmware
# Run from host machine

set -e

PORT=${1:-/dev/ttyUSB0}

echo "=========================================="
echo "  ESP32 Benchmark - Build & Flash"
echo "=========================================="
echo "Port: $PORT"
echo ""

# Check if port exists
if [ ! -e "$PORT" ]; then
    echo "❌ Port $PORT not found!"
    echo "Available ports:"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  None found"
    exit 1
fi

# Give permissions
sudo chmod 666 $PORT 2>/dev/null || true

echo "Starting Docker container..."
docker run -it --rm \
  --device=$PORT \
  --privileged \
  -v $(pwd):/workspace \
  -w /workspace/esp32_benchmark_idf \
  parol6-ultimate:latest \
  bash -c "
    echo '==> Loading ESP-IDF environment...'
    . /opt/esp-idf/export.sh
    
    echo '==> Setting target to esp32...'
    idf.py set-target esp32
    
    echo '==> Building firmware...'
    idf.py build
    
    echo '==> Flashing to ESP32...'
    idf.py -p $PORT flash monitor
  "

echo ""
echo "Flash complete! Press Ctrl+] to exit monitor."
