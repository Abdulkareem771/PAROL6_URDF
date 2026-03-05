#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

xhost +local:root >/dev/null 2>&1 || true
xhost +local:docker >/dev/null 2>&1 || true
./start_container.sh

docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false"
