#!/bin/bash
set -e

# Defaults
MODE="real"
CONTAINER_NAME="parol6_dev"
IMAGE_NAME="parol6-ultimate:latest"

# Help Menu
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./bringup.sh [mode]"
    echo "Modes:"
    echo "  real  : Run with Real Robot Driver (Default)"
    echo "  sim   : Run with Ignition Gazebo Simulation"
    echo "  fake  : Run with Fake Hardware (MoveIt only)"
    exit 0
fi

# Parse Mode
if [[ -n "$1" ]]; then
    MODE="$1"
fi

if [[ "$MODE" != "real" && "$MODE" != "sim" && "$MODE" != "fake" ]]; then
    echo "Error: Invalid mode '$MODE'. Use 'real', 'sim', or 'fake'."
    exit 1
fi

echo "=================================================="
echo "   PAROL6 Unified Bringup | Mode: $MODE"
echo "=================================================="

# 1. Host Setup
xhost +local:root
echo "X11 access granted."

# 2. Cleanup Old Containers (Aggressive cleanup to prevent conflicts)
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker rm -f $CONTAINER_NAME > /dev/null
fi

# 3. Docker Run Command
# We mount /dev for Serial/USB access (Real) and GPU acceleration (Sim)
# We mount the current directory to /workspace so edits on host reflect in container

DOCKER_ARGS=(
    --name "$CONTAINER_NAME"
    --privileged
    --net=host
    --env="DISPLAY"
    --env="QT_X11_NO_MITSHM=1"
    --env="XAUTHORITY=/tmp/.docker.xauth"
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
    --volume="$PWD:/workspace"
    --volume="/dev:/dev"
    --device="/dev/dri:/dev/dri"
    --group-add="video"
    --workdir="/workspace"
    -dt "$IMAGE_NAME" 
    /bin/bash
)

echo "Starting Container..."
docker run "${DOCKER_ARGS[@]}" > /dev/null
echo "Container '$CONTAINER_NAME' is running."

# 4. Build Workspace
# Using --packages-select to keep it fast and clean
echo "Building Workspace..."
PACKAGES="parol6 parol6_driver parol6_moveit_config"

# Clean build artifacts inside container to avoid host pollution
docker exec "$CONTAINER_NAME" bash -c "rm -rf /workspace/build /workspace/install /workspace/log"

# Build
docker exec "$CONTAINER_NAME" bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select $PACKAGES"
echo "Build Complete."

# 5. Launch
echo "Launching Unified System (Mode: $MODE)..."
docker exec -it "$CONTAINER_NAME" bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch parol6_driver unified_bringup.launch.py mode:=$MODE"
