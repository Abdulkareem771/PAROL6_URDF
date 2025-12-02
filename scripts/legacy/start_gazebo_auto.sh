#!/bin/bash
# PAROL6 - Auto Gazebo with Controller Manager

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PAROL6 - Gazebo Classic (Auto Mode)                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Enable X11
xhost +local:docker > /dev/null 2>&1

echo -e "${GREEN}🚀 Starting PAROL6 with Gazebo Classic (Auto)...${NC}"
echo ""

# Start container and run everything automatically
docker run -it --rm \
  --name parol6_dev \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e QT_X11_NO_MITSHM=1 \
  -v /home/kareem/Desktop/PAROL6_URDF:/workspace \
  --device /dev/dri \
  parol6-ultimate:latest \
  bash -c "
    echo 'Building workspace...'
    source /opt/ros/humble/setup.bash
    cd /workspace
    colcon build --symlink-install
    
    echo 'Starting controller manager...'
    ros2 run controller_manager ros2_control_node --use-sim-time &
    CONTROLLER_PID=\$!
    sleep 2
    
    echo 'Launching Gazebo...'
    source /workspace/install/setup.bash
    ros2 launch parol6 gazebo.launch.py
    
    echo 'Cleaning up...'
    kill \$CONTROLLER_PID
  "

# Cleanup
xhost -local:docker > /dev/null 2>&1
echo -e "${GREEN}✓ Session ended${NC}"
