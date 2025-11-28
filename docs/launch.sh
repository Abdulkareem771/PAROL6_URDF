#!/bin/bash

# PAROL6 Quick Launch Script
# Provides easy access to common launch commands

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "PAROL6 Robot Quick Launch"
echo -e "==========================================${NC}"
echo ""
echo "Select an option:"
echo ""
echo "1) Launch MoveIt Demo (RViz only, no Gazebo)"
echo "2) Launch Gazebo Simulation"
echo "3) Launch Gazebo + MoveIt"
echo "4) Enter Docker Container (Interactive Shell)"
echo "5) Run Tests"
echo "6) Rebuild Workspace"
echo "7) Exit"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo -e "${GREEN}Launching MoveIt Demo...${NC}"
        docker exec -it parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py"
        ;;
    2)
        echo -e "${GREEN}Launching Gazebo Simulation...${NC}"
        docker exec -it parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch parol6 gazebo.launch.py"
        ;;
    3)
        echo -e "${YELLOW}Starting Gazebo in background...${NC}"
        docker exec -d parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch parol6 gazebo.launch.py"
        sleep 5
        echo -e "${GREEN}Launching MoveIt...${NC}"
        docker exec -it parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch parol6 Movit_RViz_launch.py"
        ;;
    4)
        echo -e "${GREEN}Entering container...${NC}"
        echo -e "${YELLOW}Remember to source the workspace:${NC}"
        echo "  source /opt/ros/humble/setup.bash"
        echo "  source /workspace/install/setup.bash"
        echo ""
        docker exec -it parol6_dev bash
        ;;
    5)
        echo -e "${GREEN}Running tests...${NC}"
        ./test_setup.sh
        ;;
    6)
        echo -e "${GREEN}Rebuilding workspace...${NC}"
        docker exec -it parol6_dev bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install"
        echo -e "${GREEN}Build complete!${NC}"
        ;;
    7)
        echo -e "${YELLOW}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac
