#!/usr/bin/env bash
set -eo pipefail

# ── RViz xcb fix: OpenCV injects its own Qt plugin path which shadows the
# system xcb plugin, crashing RViz. Unset it before launching.
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset QT_PLUGIN_PATH

if [ -f /.dockerenv ]; then
    # We are inside the container
    cd /workspace
    source install/setup.bash

    # Ensure X11 display is set (needed for RViz in Docker)
    export DISPLAY="${DISPLAY:-:1}"
    export XAUTHORITY="${XAUTHORITY:-/tmp/.docker.xauth}"
    export QT_X11_NO_MITSHM=1
    ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=true
else
    # We are on the host
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"

    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh

    docker exec \
        -e DISPLAY="${DISPLAY:-:0}" \
        -e XAUTHORITY="/tmp/.docker.xauth" \
        -e QT_X11_NO_MITSHM="1" \
        -i parol6_dev bash -lc \
        "export DISPLAY='${DISPLAY:-:0}'; export XAUTHORITY=/tmp/.docker.xauth; export QT_X11_NO_MITSHM=1; unset QT_QPA_PLATFORM_PLUGIN_PATH; unset QT_PLUGIN_PATH; cd /workspace && source install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=true"
fi
