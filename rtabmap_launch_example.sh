#!/bin/bash
# RTABMap Launch Script for Calibrated Kinect2
# This script launches RTABMap with your calibrated camera

echo "==================================================="
echo "RTABMap Launch for Kinect2 (Calibrated)"
echo "==================================================="
echo ""
echo "Make sure kinect2_bridge is already running!"
echo "If not, run in another terminal:"
echo "  ros2 launch kinect2_bridge kinect2_bridge_launch.yaml fps_limit:=5"
echo ""
echo "Starting RTABMap in 3 seconds..."
sleep 3

# Source ROS2
source /opt/ros/humble/setup.bash
source /opt/kinect_ws/install/setup.bash

# Launch RTABMap with Kinect2 topics
ros2 launch rtabmap_launch rtabmap.launch.py \
    rgb_topic:=/kinect2/qhd/image_color_rect \
    depth_topic:=/kinect2/qhd/image_depth_rect \
    camera_info_topic:=/kinect2/qhd/camera_info \
    frame_id:=kinect2_rgb_optical_frame \
    approx_sync:=true \
    wait_imu_to_init:=false \
    rgbd_sync:=true \
    queue_size:=30 \
    subscribe_scan:=false \
    subscribe_odom_info:=false \
    visual_odometry:=true \
    rtabmap_args:="--delete_db_on_start"

# Notes:
# - Uses QHD resolution (960x540) for balance of quality and speed
# - visual_odometry:=true enables camera-based tracking (no wheel odometry needed)
# - approx_sync:=true allows RGB and depth to be slightly out of sync (more robust)
# - delete_db_on_start clears previous map (remove this to continue mapping)
