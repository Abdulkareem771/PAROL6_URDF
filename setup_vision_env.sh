#!/bin/bash
# Vision Environment Setup Script (Wheels-Based)
# Installs YOLO and vision libraries from pre-downloaded wheels
# Works OFFLINE if wheels/ folder exists

set -e  # Exit on error

echo "=========================================="
echo "  PAROL6 Vision Environment Setup"
echo "=========================================="
echo ""

VENV_DIR="venv_vision"
WHEELS_DIR="wheels"
WORKSPACE_DIR="$(pwd)"

# Check if running inside Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
else
    echo "⚠️  Not inside Docker - this should be run in parol6-ultimate container"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi
echo ""

# Step 1: Check for wheels
echo "[1/5] Checking for wheels..."
if [ -d "$WHEELS_DIR" ] && [ "$(ls -A $WHEELS_DIR/*.whl 2>/dev/null)" ]; then
    echo "✓ Wheels found - will install offline"
    USE_WHEELS=true
else
    echo "⚠️  No wheels found - will download from PyPI (requires internet)"
    echo "   To use offline install:"
    echo "   1. Run: ./download_vision_wheels.sh"
    echo "   2. Or get wheels/ folder from colleague"
    echo ""
    USE_WHEELS=false
fi
echo ""

# Step 2: Create/recreate venv
echo "[2/5] Setting up virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  venv_vision already exists. Remove it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing venv."
        exit 0
    fi
fi

python3 -m venv "$VENV_DIR"
echo "✓ Virtual environment created"
echo ""

# Step 3: Activate and upgrade pip
echo "[3/5] Activating venv and upgrading pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"
echo ""

# Step 4: Install vision libraries
echo "[4/5] Installing vision libraries..."
if [ "$USE_WHEELS" = true ]; then
    echo "  Mode: OFFLINE (from wheels/)"
    echo "  This should take 1-2 minutes..."
    echo ""
    
    pip install --no-index --find-links="$WHEELS_DIR" \
        ultralytics \
        opencv-python \
        scipy \
        torch \
        torchvision \
        pyyaml
else
    echo "  Mode: ONLINE (from PyPI)"
    echo "  This may take several minutes (downloading PyTorch ~900MB)..."
    echo ""
    
    pip install \
        ultralytics \
        opencv-python \
        scipy \
        torch \
        torchvision \
        pyyaml
fi

echo ""
echo "✓ Vision libraries installed"
echo ""

# Step 5: Save requirements
echo "[5/5] Saving requirements..."
pip freeze > requirements_vision.txt
echo "✓ Requirements saved to requirements_vision.txt"
echo ""

# Deactivate
deactivate

echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Installation summary:"
echo "  Virtual environment: $WORKSPACE_DIR/$VENV_DIR"
echo "  Python version: $(python3 --version | awk '{print $2}')"
if [ "$USE_WHEELS" = true ]; then
    echo "  Source: Offline (wheels/)"
else
    echo "  Source: Online (PyPI)"
fi
echo ""
echo "To activate the environment:"
echo "  source venv_vision/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To test YOLO:"
echo "  source venv_vision/bin/activate"
echo "  python3 -c 'from ultralytics import YOLO; print(\"YOLO ready!\")'"
echo ""
echo "Next steps:"
echo "  1. Read: VISION_ENV_STRATEGY.md"
echo "  2. Read: PARALLEL_WORK_GUIDE.md"
echo "  3. Start coding vision nodes!"
echo ""
