#!/bin/bash
# Stop Mobile Control Only

echo "🛑 Stopping Mobile ROS Control..."

# Stop mobile bridge processes only
docker exec parol6_dev bash -c "
  pkill -f 'ros2 launch mobile_control' || true
  pkill -f 'python3 -m http.server' || true
  pkill -f 'rosbridge_websocket' || true
  pkill -f 'web_video_server' || true
" > /dev/null 2>&1

echo "✅ Mobile ROS Control stopped"
echo "📱 Main simulation continues running"
echo "🌐 To restart mobile: ./start_mobile.sh"
