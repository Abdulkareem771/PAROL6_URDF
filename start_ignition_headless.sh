#!/bin/bash
# PAROL6 - Ignition Gazebo Headless Launcher (Server Only)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PAROL6 - Ignition Gazebo (Headless Server)               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

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

echo -e "${GREEN}🚀 Starting PAROL6 with Ignition (Headless)...${NC}"
echo ""

# Start container
echo -e "${BLUE}[1/3]${NC} Starting Docker container..."
docker run -d --rm \
  --name parol6_dev \
  --network host \
  -v /home/kareem/Desktop/PAROL6_URDF:/workspace \
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

# Launch Ignition Headless
echo -e "${BLUE}[3/3]${NC} Launching Ignition Gazebo (Server Only)..."
echo -e "${YELLOW}⚠️  Running in HEADLESS mode (no GUI)${NC}"
echo -e "${YELLOW}⚠️  Use RViz for visualization${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         STARTING IGNITION SERVER (HEADLESS)...              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}After server starts:${NC}"
echo "  1. Open a NEW terminal"
echo "  2. Run: ./add_moveit.sh"
echo "  3. Use RViz to visualize and control"
echo ""
echo -e "${YELLOW}To stop: Press Ctrl+C or run ./stop.sh${NC}"
echo ""

# Run Ignition headless (server only, no GUI) using a launcher script
docker exec parol6_dev bash -c "cat > /tmp/launch_ign.sh << 'EOFSCRIPT'
#!/bin/bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export IGN_GAZEBO_RESOURCE_PATH=/workspace/install/parol6/share:\$IGN_GAZEBO_RESOURCE_PATH

# Create a simple launch file for robot_state_publisher
cat > /tmp/rsp_launch.py << 'LAUNCHEOF'
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    urdf_file = '/workspace/install/parol6/share/parol6/urdf/PAROL6.urdf'
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True
            }]
        )
    ])
LAUNCHEOF

# Start robot_state_publisher using launch file
echo \"Starting robot_state_publisher...\"
ros2 launch /tmp/rsp_launch.py &
RSP_PID=\$!
echo \"robot_state_publisher started with PID \$RSP_PID\"
sleep 3

# Verify robot_description topic is published
echo \"Checking robot_description topic...\"
if ros2 topic list | grep -q robot_description; then
  echo \"✓ robot_description topic is available\"
else
  echo \"✗ robot_description topic not found!\"
  echo \"Available topics:\"
  ros2 topic list
  exit 1
fi

# Start Ignition server in background
echo \"Starting Ignition server...\"
ign gazebo -r -s empty.sdf &
IGN_PID=\$!
echo \"Ignition server started with PID \$IGN_PID\"

# Wait for server to be ready
sleep 5

# Spawn robot
echo \"Spawning robot...\"
ros2 run ros_ign_gazebo create -name parol6 -topic robot_description -z 0.5

# Wait a bit for robot to spawn
sleep 3

# Load controllers
echo \"Loading controllers...\"
ros2 run controller_manager spawner joint_state_broadcaster -c /controller_manager &
sleep 2
ros2 run controller_manager spawner parol6_arm_controller -c /controller_manager &
sleep 2

# Keep script running
echo \"\"
echo \"═══════════════════════════════════════════\"
echo \"✓ System ready!\"
echo \"  - Ignition server: PID \$IGN_PID\"
echo \"  - robot_state_publisher: PID \$RSP_PID\"
echo \"  - Robot spawned and controllers loaded\"
echo \"\"
echo \"Press Ctrl+C to stop.\"
echo \"═══════════════════════════════════════════\"
wait \$IGN_PID
EOFSCRIPT
chmod +x /tmp/launch_ign.sh
/tmp/launch_ign.sh
"

# When closed, clean up
echo ""
echo -e "${YELLOW}Server closed. Stopping container...${NC}"
docker stop parol6_dev
echo -e "${GREEN}✓ System stopped${NC}"
