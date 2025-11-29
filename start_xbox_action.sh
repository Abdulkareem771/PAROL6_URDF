#!/bin/bash
echo "🏭 Starting Industrial Xbox Controller..."

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo "❌ ERROR: parol6_dev container is not running!"
    echo "Please start the simulation first: ./start_ignition.sh"
    exit 1
fi

# Kill any existing controllers
docker exec parol6_dev bash -c "pkill -f xbox_action_controller; pkill -f xbox_industrial_controller; pkill -f joy_node"

# Start joy node
gnome-terminal --title="Joy Node" -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && ros2 run joy joy_node'; exec bash"

sleep 2

# Start industrial controller
gnome-terminal --title="Xbox Controller" -- bash -c "docker exec -it parol6_dev bash -c 'source /opt/ros/humble/setup.bash && python3 /workspace/xbox_industrial_controller.py'; exec bash"

echo ""
echo "✅ Industrial Controller Active!"
echo ""
echo "🎮 Controls:"
echo "  Left Stick:  Base & Shoulder"
echo "  Right Stick: Elbow & Wrist Pitch"  
echo "  Triggers:    Wrist Roll (LT/RT)"
echo "  A Button:    Reset to Zero"
echo "  B Button:    Home Position"
echo ""
