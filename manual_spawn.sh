#!/bin/bash
# Manually spawn the robot if it's missing
# Run this while Gazebo is open

echo "Spawning robot manually..."

docker exec parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  ros2 run gazebo_ros spawn_entity.py \
    -topic robot_description \
    -entity parol6_manual \
    -x 0.0 -y 0.0 -z 0.0
"
