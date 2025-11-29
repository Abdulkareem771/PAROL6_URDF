#!/bin/bash
echo "🎮 Testing Xbox Controller Integration..."

# Terminal 1: Start joy node
docker exec -d parol6_dev bash -c "
  source /opt/ros/humble/setup.bash
  ros2 run joy joy_node
"

# Terminal 2: Start controller
docker exec -it parol6_dev bash -c "
  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash
  python3 /workspace/xbox_controller_node.py
"
