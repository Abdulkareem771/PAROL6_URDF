#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

xhost +local:root >/dev/null 2>&1 || true
xhost +local:docker >/dev/null 2>&1 || true
./start_container.sh

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/launchers/$TS"
mkdir -p "$LOG_DIR"

echo "Starting full pipeline. Logs: $LOG_DIR"

# 1) Gazebo
docker exec parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 launch parol6 ignition.launch.py" \
  >"$LOG_DIR/gazebo.log" 2>&1 &
GAZEBO_PID=$!
echo "Gazebo PID: $GAZEBO_PID"

sleep 12

# 2) MoveIt (external controllers -> Gazebo)
docker exec parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false" \
  >"$LOG_DIR/moveit.log" 2>&1 &
MOVEIT_PID=$!
echo "MoveIt PID: $MOVEIT_PID"

sleep 8

# 3) Vision pipeline + bag + moveit_controller
docker exec parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 launch parol6_vision vision_moveit.launch.py" \
  >"$LOG_DIR/vision.log" 2>&1 &
VISION_PID=$!
echo "Vision PID: $VISION_PID"

echo
echo "Started all 3 components:"
echo "  Gazebo : $GAZEBO_PID"
echo "  MoveIt : $MOVEIT_PID"
echo "  Vision : $VISION_PID"
echo
echo "Use this to stop them:"
echo "  ./scripts/launchers/stop_all_vision_gazebo.sh"
echo
echo "Use this to watch logs:"
echo "  tail -f $LOG_DIR/gazebo.log"
echo "  tail -f $LOG_DIR/moveit.log"
echo "  tail -f $LOG_DIR/vision.log"
