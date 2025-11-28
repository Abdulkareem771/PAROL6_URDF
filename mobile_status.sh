#!/bin/bash
# Check Mobile Control Status

echo "📱 Mobile ROS Control Status"

if docker ps | grep -q parol6_dev; then
    echo "✅ Main container: RUNNING"
    
    # Check mobile processes
    echo ""
    echo "Mobile Services:"
    docker exec parol6_dev bash -c "
      echo '- ROS Bridge: \$(pgrep -f rosbridge_websocket > /dev/null && echo \"🟢 RUNNING\" || echo \"🔴 STOPPED\")'
      echo '- Web Server: \$(pgrep -f \"python3 -m http.server\" > /dev/null && echo \"🟢 RUNNING\" || echo \"🔴 STOPPED\")'
      echo '- Mobile Bridge: \$(pgrep -f \"mobile_bridge\" > /dev/null && echo \"🟢 RUNNING\" || echo \"🔴 STOPPED\")'
    "
    
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
    echo ""
    echo "🌐 Access URLs:"
    echo "   Web Interface: http://${IP_ADDRESS}:8080"
    echo "   ROS Bridge: ws://${IP_ADDRESS}:9090"
else
    echo "❌ Main container: STOPPED"
    echo "   Start with: ./start_ignition.sh"
fi
