#!/usr/bin/env bash
set -eo pipefail

SHAPE="${1:-Straight}"

if [ -f /.dockerenv ]; then
    # We are inside the container
    cd /workspace
    source install/setup.bash
    python3 /workspace/scripts/launchers/auto_test_runner.py "$SHAPE"
else
    # We are on the host
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"
    
    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh
    
    docker exec -i parol6_dev bash -lc "cd /workspace && source install/setup.bash && python3 /workspace/scripts/launchers/auto_test_runner.py \"$SHAPE\""
fi
