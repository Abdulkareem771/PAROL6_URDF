#!/bin/bash
# Launch Xbox Controller with MoveIt Servo
# This script starts the joy node and the servo bridge

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       PAROL6 - Xbox Controller with MoveIt Servo            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo -e "${RED}✗ Container 'parol6_dev' is not running${NC}"
    echo -e "${YELLOW}Please start it first with: ./start_ignition.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Container is running${NC}"
echo ""

echo -e "${BLUE}Starting Xbox controller with MoveIt Servo...${NC}"
echo ""
echo -e "${YELLOW}Instructions:${NC}"
echo "  • Left Stick:  X/Y linear motion"
echo "  • Right Stick: Z linear motion (up/down)"
echo "  • D-Pad:       Angular motion (pitch/yaw)"
echo "  • Triggers:    Roll motion (L2/R2)"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop${NC}"
echo ""

# Launch joy node and servo bridge in the container
docker exec -it parol6_dev bash -c "
source /opt/ros/humble/setup.bash && \
source /workspace/install/setup.bash && \
ros2 launch parol6_moveit_config servo_with_joy.launch.py
"

echo ""
echo -e "${GREEN}✓ Xbox controller stopped${NC}"
