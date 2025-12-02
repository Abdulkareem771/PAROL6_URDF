#!/bin/bash
# Quick reference card for PAROL6 commands

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                    PAROL6 QUICK REFERENCE CARD                       ║
╚══════════════════════════════════════════════════════════════════════╝

📦 WORKSPACE SETUP
──────────────────────────────────────────────────────────────────────
  cd /workspace
  source /opt/ros/humble/setup.bash
  colcon build --symlink-install
  source install/setup.bash

🚀 LAUNCH COMMANDS
──────────────────────────────────────────────────────────────────────
  # Interactive launcher (recommended)
  ./launch.sh

  # MoveIt demo (no Gazebo)
  ros2 launch parol6_moveit_config demo.launch.py

  # Gazebo simulation
  ros2 launch parol6 gazebo.launch.py

  # Gazebo + MoveIt
  ros2 launch parol6 gazebo.launch.py          # Terminal 1
  ros2 launch parol6 Movit_RViz_launch.py      # Terminal 2

🎮 CONTROLLER COMMANDS
──────────────────────────────────────────────────────────────────────
  # List controllers
  ros2 control list_controllers

  # Load controller
  ros2 control load_controller parol6_arm_controller

  # Activate controller
  ros2 control set_controller_state parol6_arm_controller active

  # List hardware interfaces
  ros2 control list_hardware_interfaces

📊 MONITORING COMMANDS
──────────────────────────────────────────────────────────────────────
  # Joint states
  ros2 topic echo /joint_states

  # Planning scene
  ros2 topic echo /monitored_planning_scene

  # TF tree
  ros2 run tf2_tools view_frames

  # List all topics
  ros2 topic list

  # Node info
  ros2 node info /move_group

🤖 ROBOT INFORMATION
──────────────────────────────────────────────────────────────────────
  Planning Group: parol6_arm
  Joints: L1, L2, L3, L4, L5, L6 (6-DOF)
  End Effector: L6
  
  Named States:
    - home:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    - ready: [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]

🔧 TESTING COMMANDS
──────────────────────────────────────────────────────────────────────
  # Run all tests
  ./test_setup.sh

  # Check URDF
  check_urdf /workspace/PAROL6/urdf/PAROL6.urdf

  # Verify package
  ros2 pkg list | grep parol6

🐍 PYTHON API EXAMPLE
──────────────────────────────────────────────────────────────────────
  # Run example controller
  cd /workspace/parol6_moveit_config/scripts
  python3 example_controller.py

  # Or use moveit_commander
  from moveit_commander import MoveGroupCommander
  arm = MoveGroupCommander("parol6_arm")
  arm.set_named_target("home")
  arm.go()

📝 MANUAL TRAJECTORY COMMAND
──────────────────────────────────────────────────────────────────────
  ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
    trajectory_msgs/msg/JointTrajectory \
    "{
      joint_names: [L1, L2, L3, L4, L5, L6],
      points: [
        {positions: [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],
         time_from_start: {sec: 2}}
      ]
    }"

🐳 DOCKER COMMANDS
──────────────────────────────────────────────────────────────────────
  # Start container
  docker run -it --rm --name parol6_dev \
    --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$(pwd):/workspace" \
    parol6-robot:latest

  # Enter running container
  docker exec -it parol6_dev bash

  # Stop container
  docker stop parol6_dev

  # List running containers
  docker ps

📚 DOCUMENTATION FILES
──────────────────────────────────────────────────────────────────────
  README.md           - Comprehensive guide
  SETUP_COMPLETE.md   - Setup completion summary
  ARCHITECTURE.md     - System architecture
  QUICKREF.sh         - This reference card

🆘 TROUBLESHOOTING
──────────────────────────────────────────────────────────────────────
  Problem: Package not found
  Solution: source /workspace/install/setup.bash

  Problem: Controllers not loading
  Solution: ros2 control load_controller parol6_arm_controller

  Problem: Planning fails
  Solution: Increase timeout in ompl_planning.yaml

  Problem: Gazebo crashes
  Solution: export QT_X11_NO_MITSHM=1

  Problem: No IK solution
  Solution: Try different start/goal poses, check joint limits

📞 RESOURCES
──────────────────────────────────────────────────────────────────────
  MoveIt 2:    https://moveit.picknik.ai/humble/
  ros2_control: https://control.ros.org/humble/
  ROS 2 Docs:  https://docs.ros.org/en/humble/

╔══════════════════════════════════════════════════════════════════════╗
║  TIP: Run './launch.sh' for an interactive menu!                    ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
