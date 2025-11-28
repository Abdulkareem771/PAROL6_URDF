#!/bin/bash

echo "🎮 Starting PAROL6 Joystick Control Bridge"
echo "================================================"

# Check if Docker container is running
if ! docker ps | grep -q "parol6_dev"; then
    echo "❌ Docker container 'parol6_dev' is not running!"
    echo "Please start the container first: ./start_ignition.sh"
    exit 1
fi

echo "📡 Starting Joystick Bridge inside Docker container..."
echo "🌐 Web interface will be available at: http://localhost:5000"
echo "📱 Mobile access: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "🎮 Features:"
echo "  - Virtual joystick for intuitive control"
echo "  - Mode switching between joystick and joint control"
echo "  - Emergency stop button"
echo "  - Real-time status monitoring"
echo ""
echo "Press Ctrl+C to stop the server"

# Run the joystick bridge with proper ROS 2 environment
docker exec -it parol6_dev bash -c "
source /opt/ros/humble/setup.bash
cd /workspace
python3 mobile_bridge_joystick.py
"
