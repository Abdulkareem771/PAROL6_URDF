#!/bin/bash
# PAROL6 - Add MoveIt to Running Gazebo
# Run this AFTER Gazebo has started

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Adding MoveIt + RViz to Gazebo                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo -e "${RED}✗ Container 'parol6_dev' is not running${NC}"
    echo "Please start Gazebo first with: ./start.sh"
    exit 1
fi

echo -e "${GREEN}✓ Container is running${NC}"
echo ""

# Check if Gazebo/Ignition is running
echo -e "${BLUE}Checking if simulation is running...${NC}"
if docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 node list 2>/dev/null | grep -E '(gazebo|ign)'" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Simulation is running${NC}"
else
    echo -e "${YELLOW}⚠️  Simulation might not be fully started yet${NC}"
    echo "Wait a few more seconds and try again"
fi
echo ""

# Enable X11 for RViz
echo -e "${YELLOW}Enabling X11 access for RViz...${NC}"
xhost +local:docker > /dev/null 2>&1

# Launch MoveIt + RViz
echo -e "${BLUE}Launching MoveIt + RViz...${NC}"
echo -e "${YELLOW}⚠️  RViz will open in a new window${NC}"
echo -e "${YELLOW}⚠️  Keep this terminal open!${NC}"
echo ""

docker exec -it parol6_dev bash -c "
  export DISPLAY=$DISPLAY && \
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  ros2 launch parol6 Movit_RViz_launch.py
"

echo ""
echo -e "${YELLOW}MoveIt/RViz closed${NC}"

# Reset X11
xhost -local:docker > /dev/null 2>&1
