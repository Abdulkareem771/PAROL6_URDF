#!/bin/bash
# Download Vision Library Wheels
# Run this ONCE on a machine with good internet
# Then share the wheels/ folder with teammates

set -e

echo "=========================================="
echo "  Downloading Vision Library Wheels"
echo "=========================================="
echo ""

# Navigate to workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create wheels directory
mkdir -p wheels

echo "[1/3] Creating temporary venv..."
python3 -m venv temp_venv
source temp_venv/bin/activate

echo "✓ Temporary venv created"
echo ""

echo "[2/3] Downloading wheels..."
echo "  Platform: manylinux2014_x86_64"
echo "  Python: 3.10"
echo "  This may take 10-15 minutes (downloading ~2GB)..."
echo ""

pip download \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --dest wheels/ \
    ultralytics \
    opencv-python \
    scipy \
    torch \
    torchvision \
    pyyaml

echo ""
echo "✓ Wheels downloaded"
echo ""

echo "[3/3] Cleaning up..."
deactivate
rm -rf temp_venv
echo "✓ Cleanup complete"
echo ""

# Show what was downloaded
echo "=========================================="
echo "  Download Complete!"
echo "=========================================="
echo ""
echo "Wheels saved to: $(pwd)/wheels/"
echo ""
ls -lh wheels/*.whl | wc -l | xargs echo "Total packages:"
du -sh wheels/ | awk '{print "Total size: "$1}'
echo ""
echo "Next steps:"
echo "  1. Share wheels/ folder with teammates"
echo "  2. Run: ./setup_vision_env.sh (uses wheels for offline install)"
echo ""
echo "To share:"
echo "  tar -czf vision_wheels.tar.gz wheels/"
echo "  # Share vision_wheels.tar.gz via Google Drive/USB"
echo ""
