#!/bin/bash
# Install Xbox Kinect v2 Drivers with CUDA Acceleration
# FIXED for path: /usr/lib/nvidia-cuda-toolkit/bin/nvcc
# Run as root

set -e

# ==========================================
# 1. SET CORRECT CUDA PATHS
# ==========================================
echo "🔍 Setting CUDA environment variables..."

# 1. Add the "bin" folder you found to the system PATH
export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"

# 2. Add the library path (usually one level up in "lib64" or "lib")
export LD_LIBRARY_PATH="/usr/lib/nvidia-cuda-toolkit/lib64:$LD_LIBRARY_PATH"

# 3. Verify it works immediately
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: 'nvcc' is still not reachable."
    echo "   Double check if the path is '/usr/lib/nvidia-cuda-toolkit/bin' or '/usr/lib/nvidia-toolkit/bin'"
    exit 1
fi

NVCC_PATH=$(which nvcc)
echo "✅ CUDA Compiler Found: $NVCC_PATH"
echo "   Version: $(nvcc --version | grep release | awk '{print $5,$6}')"
echo "=========================================="

echo "Installing dependencies..."
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

echo "Building libfreenect2 (WITH CUDA)..."

cd /tmp
rm -rf libfreenect2
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2

mkdir build && cd build

# Calculate root folder: /usr/lib/nvidia-cuda-toolkit
CUDA_ROOT=$(dirname $(dirname "$NVCC_PATH"))

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
git clone -b cuda-acceleration https://github.com/AryanRai/kinect2_ros2_cuda.git kinect2_ros2

cd /opt/kinect_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Add setup to bashrc if not present
if ! grep -q "kinect_ws" /root/.bashrc; then
    echo "source /opt/kinect_ws/install/setup.bash" >> /root/.bashrc
fi

# Make the path permanent for future sessions
if ! grep -q "nvidia-cuda-toolkit" /root/.bashrc; then
    echo "export PATH=/usr/lib/nvidia-cuda-toolkit/bin:\$PATH" >> /root/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc
fi

echo ""
echo "✅ INSTALL COMPLETE"
echo "Run with:"
echo "ros2 run kinect2_bridge kinect2_bridge --ros-args -p depth_method:=cuda -p reg_method:=cuda"