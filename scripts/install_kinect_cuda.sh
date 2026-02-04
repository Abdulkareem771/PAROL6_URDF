#!/bin/bash
# Install Xbox Kinect v2 Drivers with CUDA Acceleration
# AUTO-FIX version for CUDA 11.5
# Run as root

set -e

# ==========================================
# 1. AUTO-DETECT & FIX CUDA PATH
# ==========================================
echo "🔍 Checking CUDA environment..."

# If nvcc is NOT in the path, try to find it and add it
if ! command -v nvcc &> /dev/null; then
    if [ -f "/usr/local/cuda-11.5/bin/nvcc" ]; then
        echo "   -> Found CUDA 11.5 at /usr/local/cuda-11.5"
        export PATH=/usr/local/cuda-11.5/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:$LD_LIBRARY_PATH
    elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
        echo "   -> Found generic CUDA at /usr/local/cuda"
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    else
        echo "❌ Error: Could not find 'nvcc' automatically."
        echo "   Please verify CUDA 11.5 is installed in /usr/local/cuda-11.5"
        exit 1
    fi
fi

# Print final confirmation of what version we are using
NVCC_PATH=$(which nvcc)
echo "✅ Using CUDA Compiler: $NVCC_PATH"
echo "   Version: $(nvcc --version | grep release | awk '{print $5,$6}')"
echo "=========================================="

echo "Installing dependencies..."
# NOTE: Removed 'nvidia-cuda-toolkit' to avoid conflicts
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libusb-1.0-0-dev \
    libturbojpeg0-dev \
    libopenni2-dev \
    git \
    ros-humble-image-view \
    ros-humble-compressed-image-transport \
    ros-humble-compressed-depth-image-transport \
    ros-humble-image-pipeline

###############
# LIBFREENECT2
###############

echo "Building libfreenect2 (WITH CUDA 11.5)..."

cd /tmp
rm -rf libfreenect2
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2

mkdir build && cd build

# We use the detected path to ensure CMake finds the right libs
CUDA_ROOT=$(dirname $(dirname $(which nvcc)))

cmake .. \
  -DENABLE_OPENGL=OFF \
  -DENABLE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
  -DENABLE_OPENCL=OFF \
  -DENABLE_TLS=ON

# Verify CMake found it
if grep -q "CUDA_FOUND:TRUE" CMakeCache.txt; then
    echo "✅ CMake successfully linked against CUDA at $CUDA_ROOT"
else
    echo "⚠️  WARNING: CMake did not find CUDA."
    exit 1
fi

make -j$(nproc)
make install
ldconfig

# Install udev rules
mkdir -p /etc/udev/rules.d/
cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/

################
# ROS2 BRIDGE
################

echo "Building ROS2 driver from AryanRai/kinect2_ros2_cuda..."

mkdir -p /opt/kinect_ws/src
cd /opt/kinect_ws/src

rm -rf kinect2_ros2
# Using the specific branch for acceleration
git clone -b cuda-acceleration https://github.com/AryanRai/kinect2_ros2_cuda.git kinect2_ros2

cd /opt/kinect_ws
source /opt/ros/humble/setup.bash

# Build the bridge
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Add setup to bashrc if not present
if ! grep -q "kinect_ws" /root/.bashrc; then
    echo "source /opt/kinect_ws/install/setup.bash" >> /root/.bashrc
fi

# Ensure the paths are persistent for future sessions too
if ! grep -q "cuda-11.5" /root/.bashrc; then
    echo "export PATH=/usr/local/cuda-11.5/bin:\$PATH" >> /root/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc
fi

echo ""
echo "✅ INSTALL COMPLETE"
echo "Run with:"
echo "ros2 run kinect2_bridge kinect2_bridge --ros-args -p depth_method:=cuda -p reg_method:=cuda"