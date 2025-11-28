#!/bin/bash
# Fix missing dependencies in the CURRENT running container
# Run this if you see "No module named 'moveit_configs_utils'"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Fixing dependencies in running container...${NC}"

if ! docker ps | grep -q parol6_dev; then
    echo -e "${YELLOW}Container not running. Start it first with ./start.sh${NC}"
    exit 1
fi

docker exec parol6_dev bash -c "apt-get update && apt-get install -y ros-humble-moveit-configs-utils ros-humble-moveit-ros-move-group ros-humble-moveit-ros-visualization"

echo -e "${GREEN}✓ Dependencies installed! You can now run ./add_moveit.sh${NC}"
