#!/bin/bash
# Install Gazebo Classic in the container
# This makes the existing launch files work

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Installing Gazebo Classic                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo -e "${RED}✗ Container 'parol6_dev' is not running${NC}"
    echo "Starting container..."
    docker run -d --rm \
      --name parol6_dev \
      --env="DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --volume="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):/workspace" \
      parol6-ultimate:latest \
      tail -f /dev/null
    sleep 2
fi

echo -e "${GREEN}✓ Container is running${NC}"
echo ""

echo -e "${BLUE}Installing Gazebo Classic packages...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"
echo ""

docker exec parol6_dev bash -c "
  apt update && \
  apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    gazebo
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          ✓ Gazebo Classic Installed!                        ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Now you can run:${NC}"
    echo "  ./start.sh"
    echo ""
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi
