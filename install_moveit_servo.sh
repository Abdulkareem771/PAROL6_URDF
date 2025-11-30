#!/bin/bash
# Install MoveIt Servo into existing Docker image (with GPG fix)
# This modifies the parol6-ultimate:latest image to add the missing packages

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Installing MoveIt Servo into parol6-ultimate Image       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop any existing container
if docker ps -a | grep -q parol6_servo_install; then
    echo -e "${YELLOW}Removing existing temporary container...${NC}"
    docker stop parol6_servo_install 2>/dev/null || true
    docker rm parol6_servo_install 2>/dev/null || true
fi

echo -e "${BLUE}[1/5]${NC} Creating temporary container from parol6-ultimate:latest..."
docker run -d --name parol6_servo_install parol6-ultimate:latest tail -f /dev/null

echo -e "${GREEN}✓ Container created${NC}"
echo ""

echo -e "${BLUE}[2/5]${NC} Fixing APT sources configuration..."
docker exec parol6_servo_install bash -c "
    # Remove duplicate or conflicting sources
    rm -f /etc/apt/sources.list.d/ros2-latest.list
    rm -f /etc/apt/sources.list.d/ros2.list
    
    # Re-add ROS 2 repository properly
    apt-get update 2>/dev/null || true
    apt-get install -y curl gnupg lsb-release 2>/dev/null || true
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
    echo \"deb http://packages.ros.org/ros2/ubuntu \$(lsb_release -sc) main\" > /etc/apt/sources.list.d/ros2.list
"

echo -e "${GREEN}✓ APT sources fixed${NC}"
echo ""

echo -e "${BLUE}[3/5]${NC} Updating package lists..."
docker exec parol6_servo_install bash -c "apt-get update"

echo -e "${GREEN}✓ Package lists updated${NC}"
echo ""

echo -e "${BLUE}[4/5]${NC} Installing MoveIt Servo and dependencies..."
echo -e "${YELLOW}This may take 2-3 minutes...${NC}"
docker exec parol6_servo_install bash -c "
    apt-get install -y \
        ros-humble-moveit-servo \
        ros-humble-joy \
        ros-humble-joint-state-publisher \
    && rm -rf /var/lib/apt/lists/*
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Packages installed successfully${NC}"
else
    echo -e "${RED}✗ Installation failed${NC}"
    echo -e "${YELLOW}Checking if packages are already installed...${NC}"
    
    # Check if packages exist
    docker exec parol6_servo_install bash -c "
        dpkg -l | grep -E 'moveit-servo|ros-humble-joy|joint-state-publisher' || echo 'Packages not found'
    "
    
    docker stop parol6_servo_install
    docker rm parol6_servo_install
    exit 1
fi
echo ""

echo -e "${BLUE}[5/5]${NC} Committing changes to image..."
docker commit parol6_servo_install parol6-ultimate:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image updated successfully${NC}"
else
    echo -e "${RED}✗ Commit failed${NC}"
    docker stop parol6_servo_install
    docker rm parol6_servo_install
    exit 1
fi
echo ""

echo -e "${BLUE}Cleaning up temporary container..."
docker stop parol6_servo_install
docker rm parol6_servo_install
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✓ Installation Complete!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}MoveIt Servo and Xbox controller support are now permanently installed!${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Start the simulation: ./start_ignition.sh"
echo "  2. In a new terminal: ./start_xbox_servo.sh"
echo ""
echo -e "${YELLOW}To share with colleagues:${NC}"
echo "  docker save parol6-ultimate:latest | gzip > parol6-ultimate-with-servo.tar.gz"
echo ""
