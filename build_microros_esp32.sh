#!/bin/bash
# Build Micro-ROS ESP32 project inside Docker container

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Micro-ROS ESP32 Build Script                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if container is running
if ! docker ps | grep -q parol6_dev; then
    echo -e "${RED}✗ Container 'parol6_dev' is not running${NC}"
    echo -e "${YELLOW}Please start the container first:${NC}"
    echo "  ./start_ignition.sh"
    exit 1
fi

echo -e "${GREEN}✓ Container 'parol6_dev' is running${NC}"
echo ""

# Check if ESP-IDF is installed
echo -e "${BLUE}[1/4]${NC} Checking ESP-IDF installation..."
if docker exec parol6_dev test -f /opt/esp-idf/export.sh; then
    echo -e "${GREEN}✓ ESP-IDF found at /opt/esp-idf${NC}"
else
    echo -e "${YELLOW}⚠ ESP-IDF not found at /opt/esp-idf${NC}"
    echo -e "${YELLOW}Installing ESP-IDF...${NC}"
    docker exec parol6_dev bash -c "bash /workspace/scripts/setup/install_esp_tools.sh" || {
        echo -e "${RED}✗ Failed to install ESP-IDF${NC}"
        exit 1
    }
fi
echo ""

# Install dependencies
echo -e "${BLUE}[2/4]${NC} Installing Python dependencies..."
docker exec parol6_dev bash -c "
    . /opt/esp-idf/export.sh && \
    pip3 install catkin_pkg lark-parser colcon-common-extensions
"
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Set target (default: esp32)
TARGET=${1:-esp32}
echo -e "${BLUE}[3/4]${NC} Setting target to: ${TARGET}"

# Build the project
echo -e "${BLUE}[4/4]${NC} Building Micro-ROS ESP32 project..."
echo -e "${YELLOW}This may take several minutes...${NC}"
echo ""

docker exec -it parol6_dev bash -c "
    cd /workspace/microros_esp32 && \
    . /opt/esp-idf/export.sh && \
    idf.py set-target ${TARGET} && \
    idf.py build
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✓ Build Successful!                            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Configure (optional): docker exec -it parol6_dev bash -c 'cd /workspace/microros_esp32 && . /opt/esp-idf/export.sh && idf.py menuconfig'"
    echo "  2. Flash: docker exec -it parol6_dev bash -c 'cd /workspace/microros_esp32 && . /opt/esp-idf/export.sh && idf.py -p /dev/ttyUSB0 flash'"
    echo "  3. Monitor: docker exec -it parol6_dev bash -c 'cd /workspace/microros_esp32 && . /opt/esp-idf/export.sh && idf.py monitor'"
else
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ✗ Build Failed                                 ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

