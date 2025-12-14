#!/bin/bash


# Start container directly
echo -e "${BLUE}[1/3]${NC} Starting Docker container..."
docker run -d --rm \
  --gpus all \
  --name parol6_dev \
  --network host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -v /dev:/dev \
  -v "$(pwd)":/workspace \
  --shm-size=512m \
  parol6-ultimate:latest \
  tail -f /dev/null

sleep 2
echo -e "${GREEN}✓ Container started${NC}"
echo ""

# Enter to container
docker exec -it parol6_dev bash


# Stop container
echo -e "${BLUE}[3/3]${NC} Stopping Docker container..."
docker stop parol6_dev
sleep 1
echo -e "${GREEN}✓ Container stopped${NC}"
echo ""