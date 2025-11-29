#!/bin/bash
echo "🎮 Starting Xbox Controller for PAROL6..."

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo "❌ ERROR: parol6_dev container is not running!"
    echo "Please start the simulation first: ./start_ignition.sh"
    exit 1
fi

# Terminal 1: Start joy node
gnome-terminal --title="Joy Node" -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && ros2 run joy joy_node'; exec bash"

# Terminal 2: Start trajectory controller node  
gnome-terminal --title="Xbox Controller" -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && python3 /workspace/xbox_trajectory_controller.py'; exec bash"

# Terminal 3: Monitor
gnome-terminal --title="Monitor" -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && ros2 topic echo /parol6_arm_controller/joint_trajectory'; exec bash"

echo "✅ Xbox controller started in 3 terminals"
echo "Move sticks to control robot!"
