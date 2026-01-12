#!/bin/bash
# Download Vision Library Wheels (Linux / Docker Compatible)
# Run this on ANY machine with internet (Linux, Mac, or WSL)
# It downloads wheels specifically for:
# - OS: Linux (manylinux2014_x86_64)
# - Python: 3.10 (ROS 2 Humble standard)

set -e

echo "=========================================="
echo "  Downloading Vision Wheels (Linux/ROS 2 version)"
echo "=========================================="

# Create wheels directory
mkdir -p wheels_linux_py310
cd wheels_linux_py310

echo "[1/2] Downloading necessary libraries..."
echo "  Target: Linux x86_64, Python 3.10"
echo "  Libraries: YOLO, OpenCV, PyTorch, SciPy"

# Download exact binary wheels for Linux
pip download \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --implementation cp \
    --abi cp310 \
    ultralytics \
    opencv-python \
    scipy \
    torch \
    torchvision \
    pyyaml

echo ""
echo "[2/2] Download Complete"
echo "  Location: $(pwd)"
echo "  Size: $(du -sh . | cut -f1)"
echo ""
echo "Next Step:"
echo "  Copy this 'wheels_linux_py310' folder to your robot/Docker workspace."
