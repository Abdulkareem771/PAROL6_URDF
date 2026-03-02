#!/usr/bin/env bash
set -euo pipefail

# Publishes a conservative reachable test path in base_link frame.
# Use this to validate /vision/welding_path -> moveit_controller -> Gazebo.
#
# EE at home state (all joints=0): x=0.200, y=0.000, z=0.334
# Quaternion at home: x=0.707, y=0.000, z=-0.707, w=0.000
#
# Weld points: small sweep in Y direction from home EE position.
# Approach (Z+0.05) will be: x=0.200, y=*, z=0.384 — reachable from home.

docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && \
ros2 topic pub --once /vision/welding_path nav_msgs/msg/Path \"{
header: {frame_id: 'base_link'},
poses: [
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y: -0.08, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y: -0.04, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.00, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.04, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
  {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.08, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}}
]}\""
