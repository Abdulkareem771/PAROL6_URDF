#!/bin/bash
# Download Vision Library Wheels (Linux / Docker Compatible)
# Run this on ANY machine with internet (Linux, Mac, or WSL)
# It downloads wheels specifically for:
# - OS: Linux (manylinux2014_x86_64)
# - Python: 3.10 (ROS 2 Humble standard)


set -e

echo "Select download mode:"
echo "1) CPU"
echo "2) GPU (CUDA 12.6)"
echo "3) GPU (CUDA 12.8)"
echo "4) GPU (CUDA 13.0)"
read -p "Enter 1 or 2 or 3 or 4: " choice

echo "=========================================="
echo "  Downloading Vision Wheels (Linux/ROS 2 version)"
echo "=========================================="

# Create wheels directory
WHEEL_DIR="wheels_linux_py310"
mkdir -p "$WHEEL_DIR"

echo
echo "Downloading base packages..."
pip3 download -d "$WHEEL_DIR" \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --implementation cp \
    --abi cp310 \
    ultralytics \
    opencv-python \
    scipy \
    pyyaml

echo "✅ Base packages downloaded."

if [ "$choice" == "1" ]; then
    echo
    echo "Downloading CPU-only PyTorch..."
    pip3 download -d "$WHEEL_DIR" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

elif [ "$choice" == "2" ]; then
    echo
    echo "Downloading GPU PyTorch (CUDA 12.6)..."
    pip3 download -d "$WHEEL_DIR" \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        torch torchvision

elif [ "$choice" == "3" ]; then
    echo
    echo "Downloading GPU PyTorch (CUDA 12.8)..."
    pip3 download -d "$WHEEL_DIR" \
        torch torchvision

elif [ "$choice" == "4" ]; then
    echo
    echo "Downloading GPU PyTorch (CUDA 13.0)..."
    pip3 download -d "$WHEEL_DIR" \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch torchvision
else
    echo "❌ Invalid choice"
    exit 1
fi

echo
echo "✅ All packages downloaded into ./$WHEEL_DIR"
echo

echo "[1/2] Downloading necessary libraries..."
echo "  Target: Linux x86_64, Python 3.10"
echo "  Libraries: YOLO, OpenCV, PyTorch, SciPy"


echo ""
echo "[2/2] Download Complete"
echo "  Location: $(pwd)"
echo "  Size: $(du -sh . | cut -f1)"
echo ""
echo "Next Step:"
echo "  Copy this 'wheels_linux_py310' folder to your robot/Docker workspace."
