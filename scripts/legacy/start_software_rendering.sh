#!/bin/bash
# Fallback launcher using Software Rendering
# Use this if ./start.sh fails with GPU errors

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PAROL6 - Software Rendering Mode (Fallback)             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop existing container
if docker ps | grep -q parol6_dev; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop parol6_dev
    sleep 2
fi

echo -e "${GREEN}🚀 Starting in Software Rendering Mode...${NC}"

# Start container without GPU flags
docker run -d --rm \
  --name parol6_dev \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):/workspace" \
  parol6-ultimate:latest \
  tail -f /dev/null

sleep 2
echo -e "${GREEN}✓ Container started${NC}"

# Build workspace
echo -e "${BLUE}Building workspace...${NC}"
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install" > /tmp/parol6_build.log 2>&1

# Launch with LIBGL_ALWAYS_SOFTWARE=1
echo -e "${BLUE}Launching Gazebo (Software Rendering)...${NC}"
docker exec -it parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:/workspace/install/parol6/share:/workspace/install/share && \
  export LIBGL_ALWAYS_SOFTWARE=1 && \
  ros2 launch parol6 gazebo.launch.py
"

docker stop parol6_dev
