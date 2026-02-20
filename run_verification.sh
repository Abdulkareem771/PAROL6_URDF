#!/bin/bash
set -e

# Source ROS 2 (assuming Humble, adjust if needed)
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
elif [ -f /opt/ros/foxy/setup.bash ]; then
    source /opt/ros/foxy/setup.bash
elif [ -f /opt/ros/iron/setup.bash ]; then
    source /opt/ros/iron/setup.bash
else
    echo "Could not find ROS 2 setup script!"
    exit 1
fi

echo "Building workspace..."
colcon build --packages-select parol6_msgs parol6_vision

# Source the local workspace
if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo "Build failed to produce install/setup.bash"
    exit 1
fi

echo "Starting TF Publisher..."
# Publish static transform: base_link -> camera_rgb_optical_frame
ros2 run tf2_ros static_transform_publisher --frame-id base_link --child-frame-id camera_rgb_optical_frame --x 0.5 --y 0.0 --z 0.6 --qx 0.5 --qy -0.5 --qz 0.5 --qw 0.5 &
TF_PID=$!
echo "TF Publisher started (PID $TF_PID)"

echo "Starting Depth Matcher..."
ros2 run parol6_vision depth_matcher --ros-args -p target_frame:=base_link &
MATCHER_PID=$!
echo "Depth Matcher started (PID $MATCHER_PID)"

echo "Waiting 5 seconds for nodes to initialize..."
sleep 5

echo "Running Verification Script..."
python3 parol6_vision/scripts/verify_depth_matcher.py || echo "Verification failed!"

echo "Cleaning up..."
kill $TF_PID || true
kill $MATCHER_PID || true
echo "Done."
