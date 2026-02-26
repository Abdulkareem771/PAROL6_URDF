#!/usr/bin/env bash
set -euo pipefail

# Publishes a conservative reachable test path in base_link frame.
# Use this to validate /vision/welding_path -> moveit_controller -> Gazebo.

docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && \
ros2 topic pub --once /vision/welding_path nav_msgs/msg/Path \"{\
header: {frame_id: 'base_link'}, \
poses: [\
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.34, y: -0.08, z: 0.30}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}},\
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.38, y: -0.03, z: 0.31}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}},\
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.42, y:  0.00, z: 0.32}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}},\
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.46, y:  0.03, z: 0.31}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}},\
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.50, y:  0.06, z: 0.30}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}\
]}\""
