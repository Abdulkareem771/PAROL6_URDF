#!/usr/bin/env bash
set -eo pipefail

SERIAL_PORT="${PAROL6_SERIAL_PORT:-/dev/ttyACM0}"
BAUD_RATE="${PAROL6_BAUD_RATE:-115200}"

if [ -f /.dockerenv ]; then
    # We are inside the container
    cd /workspace
    source install/setup.bash

    echo "Starting ros2_control hardware bringup on ${SERIAL_PORT}..."
    ros2 launch parol6_hardware real_robot.launch.py \
        serial_port:="${SERIAL_PORT}" \
        baud_rate:="${BAUD_RATE}" \
        allow_spoofing:=false &
    HW_PID=$!

    trap "echo 'Stopping hardware bringup...'; kill ${HW_PID}; exit" SIGINT SIGTERM

    echo "Waiting for controller_manager to become available..."
    until ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers"; do
        sleep 0.5
    done
    # Give controllers one extra second to fully activate after the service appears
    sleep 1
    echo "Controller manager ready — starting MoveIt + RViz..."
    ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false

    kill ${HW_PID} || true
else
    # We are on the host
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"
    
    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh
    
    docker exec -i parol6_dev bash -lc "cd /workspace && source install/setup.bash && PAROL6_SERIAL_PORT='${SERIAL_PORT}' PAROL6_BAUD_RATE='${BAUD_RATE}' ./scripts/launchers/launch_moveit_real_hw.sh"
fi
