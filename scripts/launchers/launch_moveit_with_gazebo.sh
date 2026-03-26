#!/usr/bin/env bash
set -eo pipefail

if [ -f /.dockerenv ]; then
    # We are inside the container
    cd /workspace
    source install/setup.bash
    
    echo "Starting Gazebo Ign in background..."
    ros2 launch parol6 ignition.launch.py &
    GAZEBO_PID=$!
    
    # Trap SIGINT to ensure we kill Gazebo when the user presses Ctrl+C
    trap "echo 'Killing Gazebo...'; kill $GAZEBO_PID; exit" SIGINT SIGTERM
    
    echo "Waiting 5 seconds for simulation to stabilize..."
    sleep 5
    
    echo "Starting MoveIt and RViz..."
    #ros2 launch parol6_vision vision_moveit.launch.py use_bag:=false
    ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false use_sim_time:=true
    
    # If MoveIt naturally dies, kill Gazebo too
    kill $GAZEBO_PID || true
else
    # We are on the host
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$ROOT_DIR"
    
    xhost +local:root >/dev/null 2>&1 || true
    xhost +local:docker >/dev/null 2>&1 || true
    ./start_container.sh
    
    docker exec -it parol6_dev bash -lc "cd /workspace && ./scripts/launchers/launch_moveit_with_gazebo.sh"
fi
