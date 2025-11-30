#!/bin/bash
echo "Starting Direct Xbox Control..."

# Start Joy Node
ros2 run joy joy_node --ros-args -p dev:=/dev/input/js0 -p deadzone:=0.1 &
JOY_PID=$!

# Start Direct Controller
python3 /workspace/xbox_direct_control.py &
CTRL_PID=$!

echo "Systems running. Press Ctrl+C to stop."
wait
