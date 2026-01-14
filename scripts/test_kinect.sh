#!/bin/bash
# Quick test script for Xbox Kinect v2
# This script launches the camera and opens a preview window

set -e

echo "Starting Kinect v2 camera..."
echo "Make sure the camera is plugged into a USB 3.0 port!"
echo ""

# Source the workspace
source /opt/kinect_ws/install/setup.bash

# Launch camera in background
echo "Launching camera node..."
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml &
CAMERA_PID=$!

# Wait for camera to initialize
echo "Waiting for camera to initialize..."
sleep 5

# Check if topics are being published
echo "Checking for camera topics..."
ros2 topic list | grep kinect2

echo ""
echo "Camera is running!"
echo "To preview the video, run in another terminal:"
echo "  docker exec -it parol6_dev bash"
echo "  source /opt/kinect_ws/install/setup.bash"
echo "  ros2 run image_view image_view --ros-args --remap /image:=/kinect2/qhd/image_color"
echo ""
echo "Press Ctrl+C to stop the camera"

# Wait for user interrupt
wait $CAMERA_PID
