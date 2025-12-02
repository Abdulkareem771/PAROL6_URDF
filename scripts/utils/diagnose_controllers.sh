#!/bin/bash
echo "=== Controller Diagnostic ==="
docker exec -it parol6_dev bash -c "
  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash
  
  echo '1. Checking nodes:'
  ros2 node list
  
  echo ''
  echo '2. Checking joint states:'
  timeout 3 ros2 topic echo /joint_states --once || echo '❌ No joint states published'
  
  echo ''
  echo '3. Checking controller services:'
  ros2 service list | grep -E '(controller|gz_ros2_control)'
  
  echo ''
  echo '4. Checking if robot is spawned:'
  ign model --list 2>/dev/null || echo 'Gazebo not responding'
  
  echo ''
  echo '5. Checking URDF plugin:'
  ros2 param get /robot_state_publisher robot_description | grep -A5 'gz_ros2_control'
"
