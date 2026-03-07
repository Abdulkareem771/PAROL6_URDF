#!/bin/bash
# Install Kinect v2 ROS2 Bridge
# Prerequisites: libfreenect2 must already be installed

set -e

echo "🚀 Installing Kinect v2 ROS2 Bridge..."
echo ""

# Install udev rules for Kinect device access
echo "📋 Installing udev rules..."
if [ -f /tmp/libfreenect2/platform/linux/udev/90-kinect2.rules ]; then
    mkdir -p /etc/udev/rules.d/
    cp /tmp/libfreenect2/platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
    echo "✅ Udev rules installed"
else
    echo "⚠️  Warning: Could not find udev rules. You may need to manually add them."
fi

echo ""
echo "🔧 Building Kinect ROS2 workspace..."

# Create workspace
mkdir -p /opt/kinect_ws/src
cd /opt/kinect_ws/src

# Clone ROS2 bridge
rm -rf kinect2_ros2
git clone -b cuda-acceleration https://github.com/AryanRai/kinect2_ros2_cuda.git kinect2_ros2

# Build workspace
cd /opt/kinect_ws
source /opt/ros/humble/setup.bash

echo ""
echo "🔨 Building ROS2 packages..."
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Add to bashrc
if ! grep -q "kinect_ws" /root/.bashrc; then
    echo "source /opt/kinect_ws/install/setup.bash" >> /root/.bashrc
    echo "✅ Added kinect_ws to .bashrc"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Kinect ROS2 Bridge Installation Complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "📝 To use Kinect:"
echo "   1. Connect your Kinect v2 sensor"
echo "   2. Source the workspace: source /opt/kinect_ws/install/setup.bash"
echo "   3. Run: ros2 run kinect2_bridge kinect2_bridge"
echo ""
echo "📝 For CPU processing (default):"
echo "   ros2 run kinect2_bridge kinect2_bridge"
echo ""
echo "Note: This build uses CPU processing (libfreenect2 was built without CUDA)"
