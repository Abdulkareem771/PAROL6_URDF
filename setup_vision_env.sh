#!/bin/bash
# Vision Environment Setup Script
# Installs Necessary Vision Libraries (YOLO, OpenCV, PyTorch)

set -e

VENV_DIR="venv_vision"
# Look for local Linux wheels
WHEELS_DIR="wheels_linux_py310"

echo "=========================================="
echo "  PAROL6 Vision Environment Setup"
echo "=========================================="

echo "Checking installation mode..."
if [ -d "$WHEELS_DIR" ] && [ "$(ls -A $WHEELS_DIR/*.whl 2>/dev/null)" ]; then
    echo "✓ Found local wheels in '$WHEELS_DIR'"
    MODE="OFFLINE"
else
    echo "⚠️  No local wheels folder found."
    echo "   Will download libraries from Internet (PyPI)."
    MODE="ONLINE"
fi

# Create Venv
if [ -d "$VENV_DIR" ]; then
    echo "Recreating venv..."
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

echo ""
echo "Installing NECESSARY libraries..."
echo "Target: Ultralytics (YOLO), OpenCV, PyTorch, SciPy"

if [ "$MODE" = "OFFLINE" ]; then
    echo "From: Local Wheels"
    pip install --no-index --find-links="$WHEELS_DIR" \
        ultralytics opencv-python scipy torch torchvision pyyaml
else
    echo "From: PyPI (Online)"
    pip install ultralytics opencv-python scipy torch torchvision pyyaml
fi

pip freeze > requirements_vision.txt
echo ""
echo "✅ Setup Complete. Venv is ready at: $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Read: VISION_ENV_STRATEGY.md"
echo "  2. Read: PARALLEL_WORK_GUIDE.md"
echo "  3. Start coding vision nodes!"
echo ""
