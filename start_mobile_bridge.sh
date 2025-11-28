#!/bin/bash
# Mobile ROS Bridge - Run alongside existing Ignition setup

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Mobile ROS Bridge (Add-on)                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if main container is running
if ! docker ps | grep -q parol6_dev; then
    echo -e "${RED}❌ Main PAROL6 container not running${NC}"
    echo -e "${YELLOW}Start it first: ./start_ignition.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Main PAROL6 container is running${NC}"

# Install mobile tools in the existing container
echo -e "${YELLOW}Installing mobile ROS tools...${NC}"
docker exec parol6_dev bash -c "
  apt-get update && \
  apt-get install -y ros-humble-rosbridge-suite ros-humble-web-video-server python3-pip && \
  pip3 install tornado
" > /dev/null 2>&1

echo -e "${GREEN}✅ Mobile tools installed${NC}"

# Build mobile_control package
echo -e "${YELLOW}Building mobile control package...${NC}"
docker exec parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  cd /workspace && \
  colcon build --packages-select mobile_control --symlink-install
" > /dev/null 2>&1

echo -e "${GREEN}✅ Mobile control package built${NC}"

# Start mobile bridge in the existing container
echo -e "${YELLOW}Starting Mobile ROS Bridge...${NC}"
docker exec -d parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  ros2 launch mobile_control mobile_control.launch.py
"

# Start web server
echo -e "${YELLOW}Starting web interface...${NC}"
docker exec -d parol6_dev bash -c "
  cd /workspace/mobile_web && \
  python3 -m http.server 8080
"

echo ""
echo -e "${GREEN}🎉 Mobile ROS Bridge Started!${NC}"
echo ""
echo -e "${BLUE}📱 Access from phone:${NC}"
echo -e "   Web Interface: ${GREEN}http://$(hostname -I | awk '{print $1}'):8080${NC}"
echo -e "   ROS Bridge: ${GREEN}ws://$(hostname -I | awk '{print $1}'):9090${NC}"
echo ""
echo -e "${YELLOW}⚠️  Keep this terminal open or run in background with:${NC}"
echo -e "   ${GREEN}./start_mobile_bridge.sh &${NC}"
echo ""
echo -e "${YELLOW}🛑 To stop mobile bridge:${NC}"
echo -e "   ${GREEN}./stop_mobile_bridge.sh${NC}"
