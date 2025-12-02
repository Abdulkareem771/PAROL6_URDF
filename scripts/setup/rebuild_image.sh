#!/bin/bash
# Rebuild Docker image with Gazebo Classic included

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Rebuilding Docker Image with Gazebo Classic             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop any running containers
if docker ps | grep -q parol6_dev; then
    echo -e "${YELLOW}Stopping running container...${NC}"
    docker stop parol6_dev
    sleep 2
fi

echo -e "${BLUE}Building new Docker image...${NC}"
echo -e "${YELLOW}This will take 5-10 minutes (one-time only)${NC}"
echo ""

# Get the root directory of the repository
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

docker build -t parol6-ultimate:latest -f Dockerfile .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          ✓ Docker Image Rebuilt Successfully!               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Gazebo Classic is now permanently installed in the image!${NC}"
    echo ""
    echo -e "${GREEN}You can now run:${NC}"
    echo "  ./start.sh"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
