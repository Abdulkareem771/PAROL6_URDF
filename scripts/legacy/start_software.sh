#!/bin/bash
# PAROL6 - One-Click Launcher (FORCED SOFTWARE RENDERING)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PAROL6 Robot - Software Rendering Launcher             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Reset X11 permissions
xhost +local:root > /dev/null 2>&1

# Check if container is already running
if docker ps | grep -q parol6_dev; then
    echo -e "${YELLOW}⚠️  Container 'parol6_dev' is already running${NC}"
    read -p "Stop it and start fresh? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Stopping existing container...${NC}"
        docker stop parol6_dev
        sleep 2
    else
        echo -e "${RED}Please stop the existing container first:${NC}"
        echo "  docker stop parol6_dev"
        exit 1
    fi
fi

echo -e "${GREEN}🚀 Starting PAROL6 Robot System (Software Rendering)...${NC}"
echo ""

# Start container with SOFTWARE RENDERING
echo -e "${BLUE}[1/3]${NC} Starting Docker container..."
docker run -d --rm \
  --name parol6_dev \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --env="LIBGL_ALWAYS_SOFTWARE=1" \
  --env="GALLIUM_DRIVER=softpipe" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):/workspace" \
  parol6-robot:latest \
  tail -f /dev/null

sleep 3
echo -e "${GREEN}✓ Container started${NC}"
echo ""

# Build workspace
echo -e "${BLUE}[2/3]${NC} Building workspace..."
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install" > /tmp/parol6_build.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed. Check /tmp/parol6_build.log${NC}"
    docker stop parol6_dev
    exit 1
fi
echo ""

# Launch Gazebo
echo -e "${BLUE}[3/3]${NC} Launching Gazebo simulation..."
echo -e "${YELLOW}⚠️  Starting with SOFTWARE RENDERING (may be slower)${NC}"
sleep 5

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    STARTING GAZEBO...                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"

# Run Gazebo with software rendering
docker exec -it parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  export GAZEBO_MODEL_PATH=/workspace/install/parol6/share:/workspace/install/share:\$GAZEBO_MODEL_PATH && \
  export LIBGL_ALWAYS_SOFTWARE=1 && \
  echo 'Using software rendering...' && \
  ros2 launch parol6 gazebo.launch.py
"

# Cleanup
echo ""
echo -e "${YELLOW}Gazebo closed. Stopping container...${NC}"
docker stop parol6_dev
echo -e "${GREEN}✓ System stopped${NC}"
