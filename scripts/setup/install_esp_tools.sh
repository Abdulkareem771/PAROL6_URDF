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

# 2. Install ESP-IDF (Robust Resume)
echo -e "${BLUE}[2/4] Installing ESP-IDF v5.1...${NC}"
mkdir -p /opt/esp-idf
export IDF_PATH=/opt/esp-idf

if [ -d "$IDF_PATH/.git" ]; then
    echo -e "${BLUE}Detected existing ESP-IDF. Attempting to resume/update...${NC}"
    cd $IDF_PATH
    # Try to repair/update the repo if it was interrupted
    git fetch --all || true
    git reset --hard origin/v5.1 || true
    git submodule update --init --recursive || true
else
    echo -e "${BLUE}Cloning ESP-IDF...${NC}"
    # Remove empty or broken dir if it exists but no .git
    if [ -d "$IDF_PATH" ]; then rm -rf $IDF_PATH; fi
    git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git $IDF_PATH
fi

echo -e "${BLUE}Running ESP-IDF install tool (this will verify/resume tool downloads)...${NC}"
$IDF_PATH/install.sh all

echo -e "${BLUE}Installing Python dependencies for micro-ROS build...${NC}"
# Source the export script to activate the python environment
. $IDF_PATH/export.sh
# Install required packages
pip3 install catkin_pkg lark-parser colcon-common-extensions empy==3.3.4


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
if [ ! -d "agent_ws" ]; then
    ros2 run micro_ros_setup create_agent_ws.sh
fi
ros2 run micro_ros_setup build_agent.sh

# 4. Setup Environment
echo -e "${BLUE}[4/4] Setting up environment...${NC}"
# Check if already added to avoid duplicates
if ! grep -q "esp-idf/export.sh" ~/.bashrc; then
    echo "source /opt/esp-idf/export.sh" >> ~/.bashrc
fi
if ! grep -q "microros_ws/install/setup.bash" ~/.bashrc; then
    echo "source /microros_ws/install/setup.bash" >> ~/.bashrc
fi

echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}Please run 'source ~/.bashrc' to apply changes.${NC}"
