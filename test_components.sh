#!/bin/bash
# Test individual components to diagnose failures

echo "1. Testing Gazebo Server (Headless)..."
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch gazebo_ros gazebo.launch.py gui:=false" &
PID=$!
sleep 10
kill $PID
echo "Done."

echo "2. Testing Robot Spawning..."
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 run gazebo_ros spawn_entity.py -entity parol6 -file /workspace/PAROL6/urdf/PAROL6.urdf"

echo "3. Testing Controllers..."
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 control list_controllers"
