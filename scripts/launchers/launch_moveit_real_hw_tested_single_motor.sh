#!/usr/bin/env bash
set -eo pipefail

SERIAL_PORT="${PAROL6_SERIAL_PORT:-/dev/ttyACM0}"
BAUD_RATE="${PAROL6_BAUD_RATE:-115200}"

if [ -f /.dockerenv ]; then
    cd /workspace
    source install/setup.bash
    ros2 launch /workspace/parol6_hardware/launch/real_robot_tested_single_motor.launch.py \
        serial_port:="${SERIAL_PORT}" \
        baud_rate:="${BAUD_RATE}"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"

    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh

    docker exec -i parol6_dev bash -lc "cd /workspace && source install/setup.bash && PAROL6_SERIAL_PORT='${SERIAL_PORT}' PAROL6_BAUD_RATE='${BAUD_RATE}' ./scripts/launchers/launch_moveit_real_hw_tested_single_motor.sh"
fi
