#!/usr/bin/env bash
# PAROL6 - Real Robot Launcher (No Gazebo)
# Connects to Real Hardware or Virtual Socat Ports

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PAROL6 Robot - Real/Virtual Hardware Launcher          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Enable X11 access for Docker (RViz)
echo -e "${YELLOW}Enabling X11 access...${NC}"
xhost +local:docker > /dev/null 2>&1

# Container Name (Different from simulation to avoid conflict)
CONTAINER_NAME="parol6_dev"

# Check if container is already running
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo -e "${YELLOW}⚠️  Container '$CONTAINER_NAME' already exists${NC}"
    echo -e "${YELLOW} stopping and removing it to ensure fresh mount...${NC}"
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

echo -e "${GREEN}🚀 Starting PAROL6 Real Robot Control...${NC}"
echo ""

# Start container
# Note: We mount /dev so we can access /dev/ttyUSB0 or /dev/pts/X
echo -e "${BLUE}[1/3]${NC} Starting Docker container..."
docker run -d --rm \
  --name $CONTAINER_NAME \
  --network host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -v /dev:/dev \
  -v "$(pwd)":/workspace \
  parol6-ultimate:latest \
  tail -f /dev/null

sleep 2
echo -e "${GREEN}✓ Container started${NC}"
echo ""

# Install libserial-dev from local cache (no download needed!)
echo -e "${BLUE}[2/4]${NC} Installing libserial-dev from cache..."
if [ -d ".docker_cache" ] && [ "$(ls -A .docker_cache/*.deb 2>/dev/null)" ]; then
  docker exec $CONTAINER_NAME bash -c "dpkg -i /workspace/.docker_cache/*.deb 2>/dev/null; apt-get install -f -y > /dev/null 2>&1; exit 0"
  echo -e "${GREEN}✓ Installed from cache (no download)${NC}"
else
  echo -e "${YELLOW}⚠️  Cache not found, downloading...${NC}"
  docker exec $CONTAINER_NAME bash -c "apt-get update > /dev/null 2>&1 && apt-get install -y libserial-dev > /dev/null 2>&1"
  echo -e "${GREEN}✓ Downloaded and installed${NC}"
fi
echo ""

# Install ROS 2 controllers (required for ros2_control)
echo -e "${BLUE}[3/4]${NC} Installing ROS 2 controllers..."
docker exec $CONTAINER_NAME bash -c "apt-get install -y ros-humble-ros2-controllers ros-humble-ros2-control ros-humble-ros2controlcli > /dev/null 2>&1 || exit 0"
echo -e "${GREEN}✓ Controllers installed${NC}"
echo ""

# Build workspace
echo -e "${BLUE}[4/4]${NC} Building workspace..."
# Clean workspace inside Docker to avoid pollution/permission issues
docker exec $CONTAINER_NAME bash -c "rm -rf /workspace/build /workspace/install /workspace/log"

# Using --packages-select to avoid building unrelated packages (like microros) found in the dir
docker exec $CONTAINER_NAME bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install --packages-select parol6 parol6_hardware parol6_moveit_config" > /tmp/parol6_real_build.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed. Dumping log:${NC}"
    cat /tmp/parol6_real_build.log
    docker stop $CONTAINER_NAME
    exit 1
fi
echo ""

# Launch Real Robot
echo -e "${BLUE}[4/4]${NC} Launching Real Robot Driver + MoveIt + RViz..."
echo -e "${YELLOW}⚠️  Keep this terminal open!${NC}"
echo ""

docker exec -it $CONTAINER_NAME bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  ros2 launch parol6_hardware real_robot.launch.py
"

# Cleanup
echo ""
echo -e "${YELLOW}Stopping container...${NC}"
docker stop $CONTAINER_NAME
echo -e "${GREEN}✓ System stopped${NC}"

# Reset X11
xhost -local:docker > /dev/null 2>&1
