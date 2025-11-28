#!/bin/bash
# Stop Mobile ROS Bridge

echo "🛑 Stopping Mobile ROS Bridge..."

# Stop mobile bridge processes
docker exec parol6_dev bash -c "
  pkill -f 'ros2 launch mobile_control' || true
  pkill -f 'python3 -m http.server' || true
  pkill -f 'rosbridge_websocket' || true
  pkill -f 'web_video_server' || true
"

echo "✅ Mobile ROS Bridge stopped"
echo "📱 Main simulation continues running"
