# Gazebo Quick Start

## Clean Setup: Gazebo + MoveIt RViz

Use this exact order to avoid TF/time and duplicate-node issues.

1. **Prepare host GUI permissions**
```bash
cd ~/Desktop/PAROL6_URDF
xhost +local:root
xhost +local:docker
./start_container.sh
```

2. **Start Gazebo first (Terminal 1)**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

3. **Start MoveIt RViz second (Terminal 2)**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

4. **Set simulation time**
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 param set /move_group use_sim_time true"
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 param set /rviz2 use_sim_time true"
```

5. **Verify core services**
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 topic list | grep /clock"
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 control list_controllers"
```

Expected controllers:
```text
joint_state_broadcaster  ...  active
parol6_arm_controller    ...  active
```

6. **Run motion**
- In RViz: `Plan` then `Execute`.

If execute still does not move, verify trajectory publishing:
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 topic echo /parol6_arm_controller/joint_trajectory --once"
```

---

## Clean Shutdown

Stop in reverse order to avoid zombie processes and frozen relaunches.

1. In MoveIt RViz terminal: press `Ctrl+C` and wait for clean exit logs.
2. In Gazebo terminal: press `Ctrl+C` and wait for clean exit logs.
3. Exit any open container shells with `exit`.
4. Optional when done for the day:
```bash
docker stop parol6_dev
```

---

## If GUI Fails To Open

Error pattern: `could not connect to display :0` or X11 authorization error.

Run on host:
```bash
cd ~/Desktop/PAROL6_URDF
xhost +local:root
xhost +local:docker
docker restart parol6_dev
./start_container.sh
```

Then relaunch Gazebo.

---

## Full Guide

See `docs/gazebo/GAZEBO_GUIDE.md` for detailed workflows.
