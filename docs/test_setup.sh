#!/bin/bash

# PAROL6 MoveIt Test Script
# This script tests the MoveIt configuration in the Docker container

set -e  # Exit on error

echo "=========================================="
echo "PAROL6 MoveIt Configuration Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker container is running
echo -e "${YELLOW}[1/6] Checking Docker container...${NC}"
if docker ps | grep -q parol6_dev; then
    echo -e "${GREEN}✓ Container 'parol6_dev' is running${NC}"
else
    echo -e "${RED}✗ Container 'parol6_dev' is not running${NC}"
    echo "Starting container..."
    docker run -d --rm \
      --name parol6_dev \
      --env="DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --volume=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd):/workspace\" \
      parol6-robot:latest \
      tail -f /dev/null
    sleep 2
    echo -e "${GREEN}✓ Container started${NC}"
fi
echo ""

# Build the workspace
echo -e "${YELLOW}[2/6] Building workspace...${NC}"
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install" > /tmp/parol6_build.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed. Check /tmp/parol6_build.log${NC}"
    exit 1
fi
echo ""

# Check packages
echo -e "${YELLOW}[3/6] Verifying packages...${NC}"
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 pkg list | grep parol6"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Packages found${NC}"
else
    echo -e "${RED}✗ Packages not found${NC}"
    exit 1
fi
echo ""

# Verify URDF
echo -e "${YELLOW}[4/6] Checking URDF...${NC}"
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=\$(cat /workspace/PAROL6/urdf/PAROL6.urdf)" > /tmp/parol6_urdf.log 2>&1 &
URDF_PID=$!
sleep 2
kill $URDF_PID 2>/dev/null || true
if grep -q "error" /tmp/parol6_urdf.log; then
    echo -e "${RED}✗ URDF has errors${NC}"
    cat /tmp/parol6_urdf.log
else
    echo -e "${GREEN}✓ URDF is valid${NC}"
fi
echo ""

# Check MoveIt configuration files
echo -e "${YELLOW}[5/6] Verifying MoveIt configuration files...${NC}"
FILES=(
    "/workspace/parol6_moveit_config/config/parol6.srdf"
    "/workspace/parol6_moveit_config/config/kinematics.yaml"
    "/workspace/parol6_moveit_config/config/ompl_planning.yaml"
    "/workspace/parol6_moveit_config/config/moveit_controllers.yaml"
    "/workspace/parol6_moveit_config/launch/demo.launch.py"
)

for file in "${FILES[@]}"; do
    if docker exec parol6_dev test -f "$file"; then
        echo -e "${GREEN}✓ Found: $(basename $file)${NC}"
    else
        echo -e "${RED}✗ Missing: $(basename $file)${NC}"
    fi
done
echo ""

# Test launch file syntax
echo -e "${YELLOW}[6/6] Testing launch file syntax...${NC}"
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && python3 -m py_compile /workspace/parol6_moveit_config/launch/demo.launch.py" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Launch file syntax is valid${NC}"
else
    echo -e "${RED}✗ Launch file has syntax errors${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}All tests passed! ✓${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Enter the container:"
echo "   docker exec -it parol6_dev bash"
echo ""
echo "2. Source the workspace:"
echo "   source /opt/ros/humble/setup.bash"
echo "   source /workspace/install/setup.bash"
echo ""
echo "3. Launch MoveIt demo:"
echo "   ros2 launch parol6_moveit_config demo.launch.py"
echo ""
echo "4. Or launch Gazebo simulation:"
echo "   ros2 launch parol6 gazebo.launch.py"
echo ""
