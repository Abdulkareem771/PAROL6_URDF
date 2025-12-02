#!/bin/bash
# Debug script to check why robot isn't appearing

echo "Checking package installation..."
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 pkg prefix parol6"

echo "Checking mesh files in install directory..."
docker exec parol6_dev bash -c "ls -R /workspace/install/parol6/share/parol6/meshes"

echo "Checking Gazebo logs..."
docker exec parol6_dev bash -c "cat /root/.ros/log/latest/launch.log | grep -i error"
