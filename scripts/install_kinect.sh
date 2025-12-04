#!/bin/bash
# Install Xbox Kinect v2 Drivers and ROS 2 Bridge
# Run this script inside the container as root

set -e

echo "Installing dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libusb-1.0-0-dev \
    libturbojpeg0-dev \
    libglfw3-dev \
    libopenni2-dev \
    git \
    ros-humble-image-view

echo "Building libfreenect2 driver..."
cd /tmp
rm -rf libfreenect2
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
mkdir build && cd build
cmake .. -Dfreenect2_camera=ON -DENABLE_CXX11=ON -DTurboJPEG_INCLUDE_DIRS=/usr/include -DTurboJPEG_LIBRARIES=/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0
make -j$(nproc)
make install
ldconfig
mkdir -p /etc/udev/rules.d/
cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/

echo "Building ROS 2 package..."
mkdir -p /opt/kinect_ws/src
cd /opt/kinect_ws/src
git clone https://github.com/krepa098/kinect2_ros2.git
cd /opt/kinect_ws
source /opt/ros/humble/setup.bash
colcon build

echo "Adding to bashrc..."
echo "source /opt/kinect_ws/install/setup.bash" >> /root/.bashrc
# Attempt to add to user bashrc if it exists
if [ -f /home/kareem/.bashrc ]; then
    echo "source /opt/kinect_ws/install/setup.bash" >> /home/kareem/.bashrc
fi

echo "Installation Complete!"
