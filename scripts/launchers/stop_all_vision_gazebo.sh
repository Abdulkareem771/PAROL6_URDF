#!/usr/bin/env bash
set -euo pipefail

# Stop launch processes inside the container first.
docker exec parol6_dev bash -lc "pkill -15 -f 'ros2 launch parol6 ignition.launch.py|ros2 launch parol6_moveit_config demo.launch.py|ros2 launch parol6_vision vision_moveit.launch.py|rviz2|move_group|ign gazebo|ros_gz_bridge|depth_matcher|red_line_detector|path_generator|moveit_controller' || true"
sleep 2
docker exec parol6_dev bash -lc "pkill -9 -f 'rviz2|move_group|ign gazebo|ros_gz_bridge|depth_matcher|red_line_detector|path_generator|moveit_controller' || true"

echo "Stopped Gazebo + MoveIt + Vision processes inside parol6_dev."
