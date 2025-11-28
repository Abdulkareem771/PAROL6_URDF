#!/bin/bash
# PAROL6 - Status Check
# Shows current system status

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              PAROL6 System Status                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check container
echo -e "${BLUE}Container Status:${NC}"
if docker ps | grep -q parol6_dev; then
    echo -e "  ${GREEN}✓ Running${NC}"
    CONTAINER_RUNNING=true
else
    echo -e "  ${RED}✗ Not running${NC}"
    CONTAINER_RUNNING=false
fi
echo ""

if [ "$CONTAINER_RUNNING" = true ]; then
    # Check controllers
    echo -e "${BLUE}Controllers:${NC}"
    docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 control list_controllers 2>/dev/null" || echo -e "  ${YELLOW}⚠️  Unable to check controllers${NC}"
    echo ""
    
    # Check topics
    echo -e "${BLUE}Key Topics:${NC}"
    docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 topic list 2>/dev/null | grep -E '(joint_states|joint_trajectory|move_group)'" || echo -e "  ${YELLOW}⚠️  Unable to check topics${NC}"
    echo ""
    
    # Check nodes
    echo -e "${BLUE}Running Nodes:${NC}"
    docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 node list 2>/dev/null" || echo -e "  ${YELLOW}⚠️  Unable to check nodes${NC}"
    echo ""
else
    echo -e "${YELLOW}Start the system with: ./start.sh${NC}"
fi

echo -e "${BLUE}Quick Commands:${NC}"
echo "  Start:  ./start.sh"
echo "  Stop:   ./stop.sh"
echo "  Shell:  docker exec -it parol6_dev bash"
echo ""
