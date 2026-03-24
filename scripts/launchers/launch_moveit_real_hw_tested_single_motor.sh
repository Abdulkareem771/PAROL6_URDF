#!/usr/bin/env bash
set -eo pipefail

if [ -z "${PAROL6_SERIAL_PORT}" ] || [ ! -e "${PAROL6_SERIAL_PORT}" ]; then
    DETECTED_PORT=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -n 1 || true)
    if [ ! -z "$DETECTED_PORT" ]; then
        SERIAL_PORT="$DETECTED_PORT"
        echo "Auto-detected STM32 on port: $SERIAL_PORT"
    else
        # Fallback if nothing connected
        SERIAL_PORT="/dev/ttyACM0"
    fi
else
    SERIAL_PORT="${PAROL6_SERIAL_PORT}"
fi

BAUD_RATE="${PAROL6_BAUD_RATE:-115200}"

if [ -f /.dockerenv ]; then
    cd /workspace
    source install/setup.bash
    ros2 launch parol6_hardware real_robot_tested_single_motor.launch.py \
        serial_port:="${SERIAL_PORT}" \
        baud_rate:="${BAUD_RATE}"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"

    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh

    # CRITICAL: Trap SIGINT/SIGTERM from the GUI's STOP button and forward it 
    # to kill the orphaned ros2 processes INSIDE the container before exiting.
    trap 'echo "[LAUNCH] Propagating kill signal to container..."; docker exec parol6_dev pkill -INT -f "ros2" || true; exit 0' SIGINT SIGTERM

    docker exec -i parol6_dev bash -lc "cd /workspace && source install/setup.bash && PAROL6_SERIAL_PORT='${SERIAL_PORT}' PAROL6_BAUD_RATE='${BAUD_RATE}' ./scripts/launchers/launch_moveit_real_hw_tested_single_motor.sh"
fi
