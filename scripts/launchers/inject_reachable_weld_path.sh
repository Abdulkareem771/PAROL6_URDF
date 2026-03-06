#!/usr/bin/env bash
set -eo pipefail

# Publishes a conservative reachable test path in base_link frame.
# Use this to validate /vision/welding_path -> moveit_controller -> Gazebo.

if [ -f /.dockerenv ]; then
    # We are inside the container
    cd /workspace
    source install/setup.bash
    ros2 topic pub --once /vision/welding_path nav_msgs/msg/Path "{
    header: {frame_id: 'base_link'},
    poses: [
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y: -0.08, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y: -0.04, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.00, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.04, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.08, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}}
    ]}"
else
    # We are on the host
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"
    
    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh
    
    docker exec -i parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 topic pub --once /vision/welding_path nav_msgs/msg/Path \"{
    header: {frame_id: 'base_link'},
    poses: [
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y: -0.08, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y: -0.04, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.00, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.04, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}},
      {header: {frame_id: 'base_link'}, pose: {position: {x: 0.20, y:  0.08, z: 0.33}, orientation: {x: 0.707, y: 0.0, z: -0.707, w: 0.0}}}
    ]}\""
fi
