#!/bin/bash
# PAROL6 - Stop Script
# Cleanly stops all running components

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Stopping PAROL6 Robot System...${NC}"

if docker ps | grep -q parol6_dev; then
    docker stop parol6_dev
    echo -e "${GREEN}✓ Container stopped${NC}"
else
    echo -e "${RED}✗ Container 'parol6_dev' is not running${NC}"
fi

echo ""
echo -e "${GREEN}System stopped.${NC}"
echo "To start again, run: ./start.sh"
