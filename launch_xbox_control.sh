#!/bin/bash
echo "Starting Xbox Controller Integration..."

# Source ROS 2
source /opt/ros/humble/setup.bash
source /home/ros2_ws/install/setup.bash

# Check controller
echo "Checking controller..."
ls -la /dev/input/js* || echo "No controller found - check USB connection"

# Launch everything
ros2 launch parol6_control xbox_control.launch.py
