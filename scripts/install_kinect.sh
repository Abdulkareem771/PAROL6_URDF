#!/bin/bash
# Install Xbox Kinect v2 CPU-ONLY Drivers and ROS2 Bridge
# Run inside container as root

set -e

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

echo "Building libfreenect2 (CPU ONLY - NO OpenGL)..."

cd /tmp
rm -rf libfreenect2
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2

mkdir build && cd build

cmake .. \
  -DENABLE_OPENGL=OFF \
  -DENABLE_CUDA=OFF \
  -DENABLE_OPENCL=OFF \
  -DENABLE_TLS=ON

make -j$(nproc)
make install
ldconfig

# udev rules
mkdir -p /etc/udev/rules.d/
cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/

################
# ROS2 BRIDGE
################

echo "Building ROS2 driver..."

mkdir -p /opt/kinect_ws/src
cd /opt/kinect_ws/src

rm -rf kinect2_ros2
git clone https://github.com/krepa098/kinect2_ros2.git

cd /opt/kinect_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install

echo "source /opt/kinect_ws/install/setup.bash" >> /root/.bashrc

echo ""
echo "✅ INSTALL COMPLETE"
