#!/bin/bash
# Working Mobile Control

echo "🤖 Starting Mobile Control..."

# Stop any existing services
pkill -f "ros2 run rosbridge" || true
pkill -f "ros2 run mobile_control" || true

# Install the mobile control package
docker exec parol6_dev bash -c "
  source /opt/ros/humble/setup.bash
  cd /workspace/mobile_control
  pip3 install -e . > /dev/null 2>&1
"

# Start ROS bridge
echo "Starting ROS Bridge..."
docker exec -d parol6_dev bash -c "
  source /opt/ros/humble/setup.bash
  ros2 run rosbridge_server rosbridge_websocket --port 9090
"

# Start mobile bridge
echo "Starting Mobile Bridge..."
docker exec -d parol6_dev bash -c "
  source /opt/ros/humble/setup.bash
  ros2 run mobile_control mobile_bridge
"

# Ensure web server is running
echo "Starting Web Server..."
docker exec -d parol6_dev bash -c "
  cd /workspace/mobile_web
  python3 -m http.server 8080
"

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "✅ Mobile Control Started!"
echo "📱 Simple Control: http://$IP:8080/simple_control.html"
echo "🎮 Full Control: http://$IP:8080/control.html"
echo "🔧 ROS Bridge: ws://$IP:9090"
echo ""
echo "💡 Try the Simple Control first to test basic movement!"
