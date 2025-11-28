#!/bin/bash
# Simple Mobile Control Launcher

echo "🤖 Starting PAROL6 Mobile Control..."

# Build the workspace first
cd ~/Desktop/PAROL6_URDF
docker run -it --rm \
  -v /home/kareem/Desktop/PAROL6_URDF:/workspace \
  parol6-ultimate:latest \
  bash -c "cd /workspace && colcon build --symlink-install"

echo "✅ Build complete. Starting mobile control..."

# Start the mobile control system
docker run -it --rm \
  --name parol6_mobile \
  --network host \
  -p 9090:9090 \
  -p 8080:8080 \
  -v /home/kareem/Desktop/PAROL6_URDF:/workspace \
  parol6-ultimate:latest \
  bash -c "
    source /opt/ros/humble/setup.bash
    source /workspace/install/setup.bash
    
    # Start Ignition Gazebo
    echo 'Starting Ignition Gazebo...'
    ros2 launch parol6 ignition.launch.py &
    sleep 15
    
    # Start mobile bridge
    echo 'Starting Mobile ROS Bridge...'
    ros2 launch mobile_control mobile_control.launch.py
    
    echo 'Mobile control system running!'
    echo 'Web interface: http://localhost:8080/mobile_web/'
  "
