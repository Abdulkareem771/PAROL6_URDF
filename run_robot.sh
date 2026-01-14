#!/bin/bash
# Unified Robot Launcher
# Run simulation, real hardware, or fake mode from ONE container

MODE=${1:-help}

CONTAINER_NAME="parol6_dev"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ "$MODE" == "help" ] || [ "$MODE" == "-h" ] || [ "$MODE" == "--help" ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     PAROL6 Unified Robot Launcher                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: ./run_robot.sh [MODE]"
    echo ""
    echo "Modes:"
    echo "  ${GREEN}sim${NC}   - Ignition Gazebo simulation (with physics)"
    echo "  ${GREEN}real${NC}  - Real hardware (ESP32 + motors)"
    echo "  ${GREEN}fake${NC}  - Fake mode (no hardware, visualization only)"
    echo ""
    echo "Examples:"
    echo "  ./run_robot.sh sim    # Launch with Gazebo"
    echo "  ./run_robot.sh real   # Connect to real robot"
    echo "  ./run_robot.sh fake   # Visualization without hardware"
    echo ""
    exit 0
fi

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${RED}[ERROR]${NC} Container '$CONTAINER_NAME' is not running!"
    echo "Run: ./start_container.sh first"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Launching robot in ${GREEN}$MODE${NC} mode..."

# Execute appropriate launch command
case "$MODE" in
    sim|simulation)
        docker exec -it $CONTAINER_NAME bash -c "
            cd /workspace &&
            source /opt/ros/humble/setup.bash &&
            colcon build --symlink-install &&
            source install/setup.bash &&
            ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=sim
        "
        ;;
    
    real|hardware)
        docker exec -it $CONTAINER_NAME bash -c "
            cd /workspace &&
            source /opt/ros/humble/setup.bash &&
            colcon build --symlink-install &&
            source install/setup.bash &&
            ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=real
        "
        ;;
    
    fake)
        docker exec -it $CONTAINER_NAME bash -c "
            cd /workspace &&
            source /opt/ros/humble/setup.bash &&
            colcon build --symlink-install &&
            source install/setup.bash &&
            ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=fake
        "
        ;;
    
    *)
        echo -e "${RED}[ERROR]${NC} Unknown mode: $MODE"
        echo "Run: ./run_robot.sh help"
        exit 1
        ;;
esac
