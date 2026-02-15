#!/bin/bash
# Unified Container Management Script
# Single persistent container for ALL robot operations

set -e
GPU_FLAG=""
if docker info | grep -i nvidia >/dev/null 2>&1; then
    GPU_FLAG="--gpus all"
    echo "[INFO] NVIDIA runtime detected — enabling GPU"
else
    echo "[INFO] No NVIDIA runtime detected — running CPU only"
fi

CONTAINER_NAME="parol6_dev"
IMAGE_NAME="parol6-ultimate:latest"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     PAROL6 Unified Container Manager                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if container exists
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo -e "${BLUE}[INFO]${NC} Container '$CONTAINER_NAME' exists"
    
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo -e "${GREEN}[✓]${NC} Container is already running"
    else
        echo -e "${YELLOW}[!]${NC} Starting existing container..."
        docker start "$CONTAINER_NAME"
        echo -e "${GREEN}[✓]${NC} Container started"
    fi
else
    echo -e "${BLUE}[INFO]${NC} Creating new container '$CONTAINER_NAME'..."
    
    # Enable X11
    xhost +local:root >/dev/null 2>&1
    
    # Detect ESP32 port
    ESP_PORT=""
    for port in /dev/ttyUSB0 /dev/ttyACM0 /dev/ttyUSB1; do
        if [ -e "$port" ]; then
            ESP_PORT="$port"
            echo -e "${GREEN}[✓]${NC} Detected ESP32 at $ESP_PORT"
            break
        fi
    done
    
    docker run -d --name $CONTAINER_NAME \
    --network host \
    --privileged \
    $GPU_FLAG \
    -e DISPLAY=$DISPLAY \
    -e PATH="/usr/bin:$PATH" \
    -e LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/nvidia-cuda-toolkit/lib64:/host-cuda-libs:$LD_LIBRARY_PATH" \
    -e CUDA_HOME="/usr/lib/nvidia-cuda-toolkit" \
    -e CUDA_PATH="/usr/lib/nvidia-cuda-toolkit" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --env QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -v $(pwd):/workspace \
    -v /dev:/dev \
    -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
    -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
    -v /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1:ro \
    -v /usr/bin/nvcc:/usr/bin/nvcc:ro \
    -v /usr/lib/nvidia-cuda-toolkit:/usr/lib/nvidia-cuda-toolkit:ro \
    -v /usr/lib/nvidia:/usr/lib/nvidia:ro \
    -v /usr/lib/cuda:/usr/lib/cuda:ro \
    -v /usr/lib/x86_64-linux-gnu/libcudart.so:/host-cuda-libs/libcudart.so:ro \
    -v /usr/lib/x86_64-linux-gnu/libcudart.so.11.0:/host-cuda-libs/libcudart.so.11.0:ro \
    -v /usr/lib/x86_64-linux-gnu/libcudadevrt.a:/host-cuda-libs/libcudadevrt.a:ro \
    -v /usr/share/cmake-3.22/Modules:/host-cmake:ro \
    -w /workspace \
    --shm-size=512m \
    $IMAGE_NAME \
    tail -f /dev/null

    echo -e "${GREEN}[✓]${NC} Container created and started"
fi

echo ""
echo "║  Container Status: READY                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Quick Commands:"
echo "  ${GREEN}Enter shell:${NC}        docker exec -it $CONTAINER_NAME bash"
echo "  ${GREEN}Run simulation:${NC}     ./run_robot.sh sim"
echo "  ${GREEN}Run real robot:${NC}     ./run_robot.sh real"
echo "  ${GREEN}Stop container:${NC}     docker stop $CONTAINER_NAME"
echo ""

# Activate the environment
#source ultralytics_cpu_env/bin/activate
