#!/bin/bash
# Unified Container Management Script
# Single persistent container for ALL robot operations

set -e
GPU_ARGS=()
if docker info | grep -i nvidia >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_ARGS+=("--gpus" "all")
    echo "[INFO] NVIDIA runtime detected — enabling GPU"
    
    # Add optional GPU/CUDA mounts if they exist
    # Note: resolve symlinks so Docker gets real files, not dangling symlinks.
    # Skip paths where host type (file) would conflict with container type (dir) or vice versa.
    for path in /etc/OpenCL/vendors \
                /usr/bin/nvidia-smi \
                /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
                /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1 \
                /usr/lib/nvidia-cuda-toolkit \
                /usr/lib/nvidia \
                /usr/lib/cuda; do
        if [ -e "$path" ]; then
            real_path=$(realpath "$path")
            GPU_ARGS+=("-v" "$real_path:$path:ro")
        fi
    done

    # /usr/bin/nvcc is a file on host but a directory in the image — skip direct mount,
    # the container's own nvcc is sufficient since nvidia-cuda-toolkit is mounted above.

    for path in /usr/lib/x86_64-linux-gnu/libcudart.so \
                /usr/lib/x86_64-linux-gnu/libcudart.so.11.0 \
                /usr/lib/x86_64-linux-gnu/libcudadevrt.a; do
        if [ -e "$path" ]; then
            # Resolve symlinks — Docker cannot bind-mount a symlink as a file.
            # The container image has /host-cuda-libs/<name>/ as a directory,
            # so we mount the real file *inside* that directory.
            real_path=$(realpath "$path")
            dest_dir="/host-cuda-libs/$(basename $path)"
            GPU_ARGS+=("-v" "$real_path:$dest_dir/$(basename $real_path):ro")
        fi
    done
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
    "${GPU_ARGS[@]}" \
    -e DISPLAY=$DISPLAY \
    -e PATH="/usr/bin:$PATH" \
    -e LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/nvidia-cuda-toolkit/lib64:/host-cuda-libs:$LD_LIBRARY_PATH" \
    -e CUDA_HOME="/usr/lib/nvidia-cuda-toolkit" \
    -e CUDA_PATH="/usr/lib/nvidia-cuda-toolkit" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --env QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v $(pwd):/workspace \
    -v $HOME:/host_home:ro \
    -v /dev:/dev \
    -w /workspace \
    --shm-size=512m \
    $IMAGE_NAME \
    tail -f /dev/null

    echo -e "${GREEN}[✓]${NC} Container created and started"
fi

# Install libserial-dev from local cache (no download needed!)
echo -e "${BLUE}[2/4]${NC} Installing libserial-dev from cache..."
if [ -d ".docker_cache" ] && [ "$(ls -A .docker_cache/*.deb 2>/dev/null)" ]; then
  docker exec $CONTAINER_NAME bash -c "dpkg -i /workspace/.docker_cache/*.deb 2>/dev/null; apt-get install -f -y > /dev/null 2>&1; exit 0"
  echo -e "${GREEN}✓ Installed from cache (no download)${NC}"
else
  echo -e "${YELLOW}⚠️  Cache not found, downloading...${NC}"
  docker exec $CONTAINER_NAME bash -c "apt-get update > /dev/null 2>&1 && apt-get install -y libserial-dev > /dev/null 2>&1"
  echo -e "${GREEN}✓ Downloaded and installed${NC}"
fi
echo ""

# Install ROS 2 controllers (required for ros2_control)
echo -e "${BLUE}[3/4]${NC} Installing ROS 2 controllers..."
docker exec $CONTAINER_NAME bash -c "apt-get install -y ros-humble-ros2-controllers ros-humble-ros2-control ros-humble-ros2controlcli > /dev/null 2>&1 || exit 0"
echo -e "${GREEN}✓ Controllers installed${NC}"
echo ""

# Build workspace
echo -e "${BLUE}[4/4]${NC} Building workspace..."
# Clean workspace inside Docker to avoid pollution/permission issues
docker exec $CONTAINER_NAME bash -c "rm -rf /workspace/build /workspace/install /workspace/log"

# Using --packages-select to avoid building unrelated packages (like microros) found in the dir
docker exec $CONTAINER_NAME bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install --packages-select parol6 parol6_hardware parol6_moveit_config parol6_driver parol6_msgs parol6_vision" > /tmp/parol6_real_build.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed. Dumping log:${NC}"
    cat /tmp/parol6_real_build.log
    docker stop $CONTAINER_NAME
    exit 1
fi
echo ""


# Always refresh xauth token so RViz/GUI apps work in all terminals
echo -e "${BLUE}[INFO]${NC} Refreshing X11 auth token for GUI support..."
xauth nlist $DISPLAY 2>/dev/null | sed 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 2>/dev/null || true
docker cp /tmp/.docker.xauth parol6_dev:/tmp/.docker.xauth 2>/dev/null && \
    echo -e "${GREEN}[✓]${NC} X11 auth token injected — RViz will work in all terminals" || \
    echo -e "${YELLOW}[!]${NC} Could not inject X11 auth (GUI may not work)"

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
