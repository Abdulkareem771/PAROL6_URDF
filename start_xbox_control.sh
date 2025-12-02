#!/bin/bash
# Xbox Direct Control Launcher for PAROL6

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        PAROL6 - Xbox Controller (Direct Control)           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo -e "${YELLOW}⚠️  Container 'parol6_dev' is not running!${NC}"
    echo -e "${YELLOW}Please start it first with: ./start_ignition.sh${NC}"
    exit 1
fi

echo -e "${GREEN}🎮 Starting Xbox Controller...${NC}"
echo ""
echo -e "${YELLOW}How to Use:${NC}"
echo "  • Left Stick  → Move Base (L1) and Shoulder (L2)"
echo "  • Right Stick → Move Elbow (L3) and Wrist Pitch (L4)"
echo "  • D-Pad Up/Down → Wrist Yaw (L5)"
echo "  • Triggers (LT/RT) → Wrist Roll (L6)"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Create a temporary script inside the container
docker exec parol6_dev bash -c 'cat > /tmp/start_xbox.sh << "INNEREOF"
#!/bin/bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Start joy_node in background
ros2 run joy joy_node --ros-args -p device_id:=0 -p deadzone:=0.1 &
JOY_PID=$!

# Give joy_node time to start
sleep 1

# Start xbox control
python3 /workspace/xbox_direct_control.py

# Cleanup
kill $JOY_PID 2>/dev/null || true
INNEREOF
chmod +x /tmp/start_xbox.sh
/tmp/start_xbox.sh
'

echo ""
echo -e "${GREEN}✓ Xbox Control stopped${NC}"
