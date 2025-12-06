#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting Interactive ESP32 & Micro-ROS Installation...${NC}"

# 1. Install Dependencies
echo -e "${BLUE}[1/4] Installing dependencies...${NC}"
apt-get update
apt-get install -y git wget flex bison gperf python3 python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0

# 2. Install ESP-IDF
echo -e "${BLUE}[2/4] Installing ESP-IDF v5.1...${NC}"
mkdir -p /opt/esp-idf
export IDF_PATH=/opt/esp-idf
if [ ! -d "$IDF_PATH/.git" ]; then
    git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git $IDF_PATH
fi
$IDF_PATH/install.sh all

# 3. Build Micro-ROS Agent
echo -e "${BLUE}[3/4] Building Micro-ROS Agent...${NC}"
mkdir -p /microros_ws/src
cd /microros_ws
if [ ! -d "src/micro_ros_setup" ]; then
    git clone -b humble https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup
fi
source /opt/ros/humble/setup.bash
apt-get update && rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build

source install/setup.bash
ros2 run micro_ros_setup create_agent_ws.sh
ros2 run micro_ros_setup build_agent.sh

# 4. Setup Environment
echo -e "${BLUE}[4/4] Setting up environment...${NC}"
echo "source /opt/esp-idf/export.sh" >> ~/.bashrc
echo "source /microros_ws/install/setup.bash" >> ~/.bashrc

echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}Please run 'source ~/.bashrc' to apply changes.${NC}"
