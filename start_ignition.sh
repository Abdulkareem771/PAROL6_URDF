#!/bin/bash
# PAROL6 - Ignition Gazebo Launcher

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PAROL6 Robot - Ignition Gazebo Launcher                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Enable X11 access for Docker
echo -e "${YELLOW}Enabling X11 access...${NC}"
xhost +local:docker > /dev/null 2>&1

# Create X11 authentication file for Docker
XAUTH=/tmp/.docker.xauth
# Get the real user's Xauthority file (handle sudo case)
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    USER_UID=$(getent passwd "$SUDO_USER" | cut -d: -f3)
    
    # Check common locations
    if [ -f "/run/user/$USER_UID/gdm/Xauthority" ]; then
        XAUTH_SOURCE="/run/user/$USER_UID/gdm/Xauthority"
    elif [ -f "$USER_HOME/.Xauthority" ]; then
        XAUTH_SOURCE="$USER_HOME/.Xauthority"
    else
        # Try to find it from the user's environment if possible, or fallback
        XAUTH_SOURCE=""
        echo -e "${YELLOW}⚠️  Could not find .Xauthority file. X11 forwarding might fail.${NC}"
    fi
else
    XAUTH_SOURCE=${XAUTHORITY:-"$HOME/.Xauthority"}
fi

# Create/Update the auth file
touch $XAUTH
if [ -n "$XAUTH_SOURCE" ] && [ -f "$XAUTH_SOURCE" ]; then
    xauth -f "$XAUTH_SOURCE" nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
else
    echo -e "${YELLOW}⚠️  No Xauthority source found. Creating empty auth file.${NC}"
fi
chmod 644 $XAUTH

# Check if container is already running
if docker ps -a | grep -q parol6_dev; then
    echo -e "${YELLOW}⚠️  Container 'parol6_dev' already exists${NC}"
    read -p "Stop it and start fresh? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Stopping/Removing existing container...${NC}"
        docker stop parol6_dev >/dev/null 2>&1 || true
        docker rm parol6_dev >/dev/null 2>&1 || true
        sleep 2
    else
        echo -e "${RED}Please stop the existing container first:${NC}"
        echo "  docker stop parol6_dev && docker rm parol6_dev"
        exit 1
    fi
fi

echo -e "${GREEN}🚀 Starting PAROL6 with Ignition Gazebo...${NC}"
echo ""

# Start container with enhanced X11 support
echo -e "${BLUE}[1/3]${NC} Starting Docker container..."
docker run -d --rm \
  --name parol6_dev \
  --network host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -v /dev:/dev \
  -v "$(pwd)":/workspace \
  --device /dev/dri \
  --group-add video \
  --shm-size=512m \
  parol6-ultimate:latest \
  tail -f /dev/null

sleep 2
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

# Launch Ignition Gazebo
echo -e "${BLUE}[3/3]${NC} Launching Ignition Gazebo..."
echo -e "${YELLOW}⚠️  Ignition will open in a new window${NC}"
echo -e "${YELLOW}⚠️  Keep this terminal open!${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              STARTING IGNITION GAZEBO...                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}After Ignition loads:${NC}"
echo "  1. Open a NEW terminal"
echo "  2. Run: ./add_moveit.sh"
echo ""
echo -e "${YELLOW}To stop: Press Ctrl+C or run ./stop.sh${NC}"
echo ""

# Run Ignition in foreground
docker exec -it parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  export IGN_GAZEBO_RESOURCE_PATH=/workspace/install/parol6/share:\$IGN_GAZEBO_RESOURCE_PATH && \
  ros2 launch parol6 ignition.launch.py
"

# When closed, clean up
echo ""
echo -e "${YELLOW}Ignition closed. Stopping container...${NC}"
docker stop parol6_dev
echo -e "${GREEN}✓ System stopped${NC}"

# Reset X11 access
xhost -local:docker > /dev/null 2>&1