#!/usr/bin/env bash
set -eo pipefail

SERIAL_PORT="${PAROL6_SERIAL_PORT:-}"
BAUD_RATE="${PAROL6_BAUD_RATE:-115200}"

if [ -f /.dockerenv ]; then
    # ── Inside container ────────────────────────────────────────────────────
    cd /workspace
    source install/setup.bash

    # Auto-detect serial port if not explicitly set
    if [ -z "${SERIAL_PORT}" ]; then
        for candidate in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyUSB0 /dev/ttyUSB1; do
            if [ -e "${candidate}" ]; then
                SERIAL_PORT="${candidate}"
                echo "Auto-detected serial port: ${SERIAL_PORT}"
                break
            fi
        done
    fi

    if [ -z "${SERIAL_PORT}" ]; then
        echo "WARNING: No serial device found. Falling back to allow_spoofing=true."
        SERIAL_PORT="/dev/ttyACM0"
        SPOOFING="true"
    fi
    SPOOFING="${SPOOFING:-false}"

    echo "Starting ros2_control hardware bringup on ${SERIAL_PORT} (allow_spoofing=${SPOOFING})..."
    ros2 launch parol6_hardware real_robot.launch.py \
        serial_port:="${SERIAL_PORT}" \
        baud_rate:="${BAUD_RATE}" \
        allow_spoofing:=${SPOOFING} &
    HW_PID=$!

    trap "echo 'Stopping hardware bringup...'; kill ${HW_PID} 2>/dev/null; exit" SIGINT SIGTERM

    echo "Waiting for controller_manager to become available (timeout 20s)..."
    WAIT=0
    until ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers"; do
        sleep 0.5
        WAIT=$((WAIT+1))
        if [ $WAIT -ge 40 ]; then
            echo "ERROR: controller_manager did not start within 20 seconds."
            kill ${HW_PID} 2>/dev/null
            exit 1
        fi
        if ! kill -0 ${HW_PID} 2>/dev/null; then
            echo "ERROR: ros2_control_node has died. Aborting."
            echo "  Tip: Set PAROL6_SERIAL_PORT=/dev/ttyACM1 if port auto-detection picked wrong device."
            exit 1
        fi
    done

    echo "Waiting for controllers to become active (timeout 20s)..."
    WAIT=0
    until ros2 control list_controllers 2>/dev/null | grep -q "joint_state_broadcaster.*active" && \
          ros2 control list_controllers 2>/dev/null | grep -q "parol6_arm_controller.*active"; do
        sleep 0.5
        WAIT=$((WAIT+1))
        if [ $WAIT -ge 40 ]; then
            echo "ERROR: controllers did not become active within 20 seconds."
            ros2 control list_controllers 2>/dev/null || true
            kill ${HW_PID} 2>/dev/null
            exit 1
        fi
        if ! kill -0 ${HW_PID} 2>/dev/null; then
            echo "ERROR: ros2_control_node exited while waiting for controllers."
            exit 1
        fi
    done

    echo "Controller manager ready — starting MoveIt + RViz..."
    ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false

    kill ${HW_PID} || true

else
    # ── On host — re-run inside the container ──────────────────────────────
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"

    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh

    docker exec -i parol6_dev bash -lc \
        "cd /workspace && source install/setup.bash && \
         PAROL6_SERIAL_PORT='${SERIAL_PORT}' PAROL6_BAUD_RATE='${BAUD_RATE}' \
         ./scripts/launchers/launch_moveit_real_hw.sh"
fi
