#!/bin/bash
# Download Vision Library Wheels (Linux / Docker Compatible)
# Target:
#   OS: manylinux2014_x86_64
#   Python: 3.10 (ROS 2 Humble)

set -e

echo "Select download mode:"
echo "1) CPU"
echo "2) GPU (CUDA 12.6)"
echo "3) GPU (CUDA 12.8)"
echo "4) GPU (CUDA 13.0)"
echo "To exit, Enter any other key."
read -p "Enter choice: " choice

echo "=========================================="
echo "  Downloading Vision Wheels (Linux / Py3.10)"
echo "=========================================="

WHEEL_DIR="wheels_linux_py310"
mkdir -p "$WHEEL_DIR"

COMMON_FLAGS="\
--only-binary=:all: \
--platform manylinux2014_x86_64 \
--python-version 3.10 \
--implementation cp \
--abi cp310"

# -------------------------------------------------
# Step 1: Download PyTorch FIRST (fixed versions)
# -------------------------------------------------

TORCH_PKGS="torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1"

if [ "$choice" == "1" ]; then
    echo "Downloading CPU-only PyTorch..."
    pip3 download -d "$WHEEL_DIR" "$COMMON_FLAGS" \
        $TORCH_PKGS --index-url https://download.pytorch.org/whl/cpu
        

elif [ "$choice" == "2" ]; then
    echo "Downloading GPU PyTorch (CUDA 12.6)..."
    pip3 download -d "$WHEEL_DIR" "$COMMON_FLAGS" \
        $TORCH_PKGS --index-url https://download.pytorch.org/whl/cu126
    

elif [ "$choice" == "3" ]; then
    echo "Downloading GPU PyTorch (CUDA 12.8)..."
    pip3 download -d "$WHEEL_DIR" "$COMMON_FLAGS" \
        $TORCH_PKGS --index-url https://download.pytorch.org/whl/cu128

elif [ "$choice" == "4" ]; then
    echo "Downloading GPU PyTorch (CUDA 13.0)..."
    pip3 download -d "$WHEEL_DIR" "$COMMON_FLAGS" \
        $TORCH_PKGS --index-url https://download.pytorch.org/whl/cu130

else
    echo "Eixting.."
    exit 1
fi

echo "✅ PyTorch downloaded."

# ---------------------------------------------
# Step 2: Download YOLO + Vision Dependencies
# ---------------------------------------------

echo "Downloading YOLO and vision libraries..."

pip3 download -d "$WHEEL_DIR" $COMMON_FLAGS \
    pyyaml \
    opencv-python \
    scipy \
    ultralytics
    
echo
echo "✅ All packages downloaded into ./$WHEEL_DIR"
echo "Directory size:"
du -sh "$WHEEL_DIR"
echo
echo "Next Step:"
echo "Copy 'wheels_linux_py310' to your robot or Docker workspace."
