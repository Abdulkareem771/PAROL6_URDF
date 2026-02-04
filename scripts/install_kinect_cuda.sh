#!/bin/bash
# Install Xbox Kinect v2 Drivers with CUDA Acceleration
# optimized for systems with EXISTING CUDA 11.5
# Run as root

set -e

# 1. PRE-FLIGHT CHECK: Verify CUDA 11.5 is visible
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: 'nvcc' not found in PATH."
    echo "   Please export your CUDA path before running this script."
    echo "   Example: export PATH=/usr/local/cuda-11.5/bin:\$PATH"
    exit 1
fi

echo "✅ Found CUDA Compiler: $(nvcc --version | grep release | awk '{print $5,$6}')"

echo "Installing dependencies..."
# NOTE: removed 'nvidia-cuda-toolkit' to avoid conflict with your existing CUDA 11.5
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

# We explicitly point CMake to your CUDA compiler to be safe
cmake .. \
  -DENABLE_OPENGL=OFF \
  -DENABLE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=$(dirname $(dirname $(which nvcc))) \
  -DENABLE_OPENCL=OFF \
  -DENABLE_TLS=ON

# Verify CMake found it
if grep -q "CUDA_FOUND:TRUE" CMakeCache.txt; then
    echo "✅ CMake successfully linked against your CUDA 11.5."
else
    echo "⚠️  WARNING: CMake did not find CUDA."
    echo "   You may need to manually pass -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.5"
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

echo ""
echo "✅ INSTALL COMPLETE"
echo "Run with:"
echo "ros2 run kinect2_bridge kinect2_bridge --ros-args -p depth_method:=cuda -p reg_method:=cuda"
