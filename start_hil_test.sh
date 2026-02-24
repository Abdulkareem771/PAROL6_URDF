#!/bin/bash
echo "Starting Hardware in the Loop test with Teensy!"
echo "NOTE: ESP32 MUST be plugged in generating PWM, Teensy MUST be on /dev/ttyACM0"
echo "Launching ROS 2 Hardware Interface & RViZ..."
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch parol6_hardware real_robot.launch.py serial_port:=/dev/ttyACM0"
