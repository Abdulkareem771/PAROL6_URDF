#!/usr/bin/env bash
# inject_path.sh
# ============================================================
# PAROL6 Full Stack Launcher + Manual Path Injection
#
# Usage:
#   ./scripts/inject_path.sh          → full launch + vision + inject test path
#   ./scripts/inject_path.sh gazebo   → start Gazebo only
#   ./scripts/inject_path.sh moveit   → start MoveIt/RViz only
#   ./scripts/inject_path.sh vision   → start vision pipeline only
#   ./scripts/inject_path.sh inject   → inject test path + trigger (stack must be running)
#   ./scripts/inject_path.sh kill     → kill all ROS/Gazebo processes in container
# ============================================================

set -e

DOCKER="docker exec parol6_dev"
ROS="$DOCKER bash -lc"
DCMD="source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[inject_path]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ─────────────────────────────────────────────
# Check Docker container is running
# ─────────────────────────────────────────────
check_container() {
  STATUS=$(docker ps --filter name=parol6_dev --format "{{.Status}}" 2>/dev/null)
  if [ -z "$STATUS" ]; then
    error "Container 'parol6_dev' is not running. Start it with: ./start_container.sh"
  fi
  log "Container: $STATUS"
}

# ─────────────────────────────────────────────
# KILL — stop all ROS/Gazebo processes
# ─────────────────────────────────────────────
do_kill() {
  log "Killing all ROS/Gazebo/vision processes in container..."
  $DOCKER bash -c "pkill -9 -f 'ign|gazebo|rviz|move_group|robot_state|controller_manager|red_line_detector|depth_matcher|path_generator|moveit_controller|ros2_bag' 2>/dev/null; echo done" || true
  sleep 2
  log "Done."
}

# ─────────────────────────────────────────────
# GAZEBO — launch ignition simulation
# ─────────────────────────────────────────────
do_gazebo() {
  log "Starting Gazebo (background) → log: /tmp/gazebo.log"
  $DOCKER bash -lc "$DCMD && ros2 launch parol6 ignition.launch.py > /tmp/gazebo.log 2>&1" &

  log "Waiting for controllers to activate..."
  for i in $(seq 1 30); do
    sleep 2
    if $DOCKER bash -lc "$DCMD && ros2 control list_controllers 2>/dev/null" | grep -q "parol6_arm_controller.*active"; then
      log "✅ Gazebo up — parol6_arm_controller ACTIVE"
      return 0
    fi
    echo -n "."
  done
  warn "Gazebo may not be fully ready. Check: docker exec parol6_dev cat /tmp/gazebo.log"
}

# ─────────────────────────────────────────────
# MOVEIT — launch move_group + RViz
# ─────────────────────────────────────────────
do_moveit() {
  log "Starting MoveIt/RViz (background) → log: /tmp/moveit.log"
  $DOCKER bash -lc "$DCMD && ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false > /tmp/moveit.log 2>&1" &

  log "Waiting for move_group to be ready..."
  for i in $(seq 1 30); do
    sleep 2
    if $DOCKER bash -c "grep -q 'start planning now' /tmp/moveit.log 2>/dev/null"; then
      log "✅ MoveIt up — 'You can start planning now!'"
      return 0
    fi
    echo -n "."
  done
  warn "MoveIt may not be ready. Check: docker exec parol6_dev cat /tmp/moveit.log"
}

# ─────────────────────────────────────────────
# SIM TIME — set use_sim_time on move_group + rviz2
# ─────────────────────────────────────────────
do_simtime() {
  log "Setting use_sim_time=true on /move_group and /rviz2..."
  $ROS "$DCMD && ros2 param set /move_group use_sim_time true" && \
  $ROS "$DCMD && ros2 param set /rviz2 use_sim_time true" && \
  log "✅ Sim time set" || warn "Failed to set sim time (nodes may not be up yet)"
}

# ─────────────────────────────────────────────
# VISION — launch vision pipeline
# ─────────────────────────────────────────────
do_vision() {
  log "Starting vision pipeline (background) → log: /tmp/vision.log"
  $DOCKER bash -lc "$DCMD && ros2 launch parol6_vision vision_moveit.launch.py > /tmp/vision.log 2>&1" &

  log "Waiting for path generation..."
  for i in $(seq 1 30); do
    sleep 2
    if $DOCKER bash -c "grep -q 'Generated path' /tmp/vision.log 2>/dev/null"; then
      log "✅ Vision pipeline — path generated"
      return 0
    fi
    echo -n "."
  done
  warn "Vision pipeline may not have generated path yet. Check: docker exec parol6_dev cat /tmp/vision.log"
}

# ─────────────────────────────────────────────
# INJECT — publish test path + trigger execution
# ─────────────────────────────────────────────
do_inject() {
  log "Publishing 6-waypoint test path to /vision/welding_path..."
  $ROS "$DCMD && python3 -c \"
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time

rclpy.init()
node = Node('inject_path_script')
pub = node.create_publisher(Path, '/vision/welding_path', 10)

path = Path()
path.header.frame_id = 'base_link'
path.header.stamp = node.get_clock().now().to_msg()

# FK-confirmed home position: x=0.200, y=0.000, z=0.334
# FK-confirmed home orientation: x=0.7071068, y=0, z=-0.7071068, w=0
for x in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    ps = PoseStamped()
    ps.header = path.header
    ps.pose.position.x = x
    ps.pose.position.y = 0.0
    ps.pose.position.z = 0.334
    ps.pose.orientation.x = 0.7071068
    ps.pose.orientation.y = 0.0
    ps.pose.orientation.z = -0.7071068
    ps.pose.orientation.w = 0.0
    path.poses.append(ps)

time.sleep(1.5)
pub.publish(path)
print('Published 6-waypoint test path')
time.sleep(1.0)
node.destroy_node()
rclpy.shutdown()
\""

  log "Triggering execution via service call..."
  RESPONSE=$($ROS "$DCMD && ros2 service call /moveit_controller/execute_welding_path std_srvs/srv/Trigger '{}' 2>&1")
  if echo "$RESPONSE" | grep -q "success=True"; then
    log "✅ Execution triggered — watch Terminal 3 (vision) for 'Sequence Complete'"
  else
    warn "Service response: $RESPONSE"
  fi
}

# ─────────────────────────────────────────────
# STATUS — check current state of all nodes
# ─────────────────────────────────────────────
do_status() {
  log "=== Node Status ==="
  $ROS "$DCMD && ros2 node list 2>/dev/null" | grep -E 'move_group|rviz|controller|moveit' || echo "  (none)"

  log "=== Controllers ==="
  $ROS "$DCMD && ros2 control list_controllers 2>/dev/null" || echo "  (unavailable)"
}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
MODE="${1:-all}"

check_container

case "$MODE" in
  kill)
    do_kill
    ;;
  gazebo)
    do_gazebo
    ;;
  moveit)
    do_moveit
    ;;
  vision)
    do_simtime
    do_vision
    ;;
  inject)
    do_inject
    ;;
  status)
    do_status
    ;;
  all)
    log "=== PAROL6 Full Stack Launch ==="
    do_kill
    log ""

    log "--- Step 1: Gazebo ---"
    do_gazebo
    log ""

    log "--- Step 2: MoveIt + RViz ---"
    do_moveit
    log ""

    log "--- Step 3: Sim Time ---"
    do_simtime
    log ""

    log "--- Step 4: Vision Pipeline ---"
    do_vision
    log ""

    log "--- Step 5: Inject Test Path ---"
    sleep 5  # let vision controller settle
    do_inject
    log ""

    log "=== Launch complete. Monitor phases in Docker logs: ==="
    log "  Gazebo:  docker exec parol6_dev cat /tmp/gazebo.log"
    log "  MoveIt:  docker exec parol6_dev cat /tmp/moveit.log"
    log "  Vision:  docker exec parol6_dev cat /tmp/vision.log"
    ;;
  *)
    echo "Usage: $0 [gazebo|moveit|vision|inject|kill|status|all]"
    exit 1
    ;;
esac
