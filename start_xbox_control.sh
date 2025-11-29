#!/bin/bash
echo "🎮 Starting Xbox Controller for PAROL6..."

# Terminal 1: Start joy node
gnome-terminal -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && ros2 run joy joy_node'; exec bash"

# Terminal 2: Start controller node  
gnome-terminal -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && python3 /workspace/xbox_controller_node.py'; exec bash"

# Terminal 3: Monitor
gnome-terminal -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && ros2 topic echo /joint_commands'; exec bash"

echo "✅ Xbox controller started in 3 terminals"
echo "Move sticks to control robot!"
