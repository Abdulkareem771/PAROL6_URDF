# Gazebo Quick Start

## Config Split (Important)

- Simulation uses:
  - `PAROL6/config/ros2_controllers_sim.yaml`
  - `parol6_moveit_config/config/ros2_controllers_sim.yaml`
- Hardware-oriented settings remain in:
  - `PAROL6/config/ros2_controllers.yaml`
  - `parol6_moveit_config/config/ros2_controllers.yaml`

## How The Split Works Internally

### Simulation Path (Gazebo + MoveIt)

1. `ros2 launch parol6 ignition.launch.py` loads URDF from `PAROL6/urdf/PAROL6.urdf`.
2. In that URDF, `gz_ros2_control` plugin reads:
   - `PAROL6/config/ros2_controllers_sim.yaml`
3. `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false` starts MoveIt/RViz only.
4. MoveIt sends trajectory goals to Gazebo's controller manager action server:
   - `/parol6_arm_controller/follow_joint_trajectory`
5. In this mode, Gazebo owns the controllers, so `demo.launch.py` must not spawn fake controllers.

### Fake/Standalone MoveIt Path (No Gazebo)

1. `ros2 launch parol6_moveit_config demo.launch.py` (default `use_fake_hardware:=true`).
2. `demo.launch.py` launches `ros2_control_node` internally.
3. It uses:
   - `parol6_moveit_config/config/ros2_controllers_sim.yaml`
4. This gives a local fake controller stack for planning/testing without Gazebo.

### Real Hardware Path

1. Hardware stacks keep using hardware-oriented files:
   - `PAROL6/config/ros2_controllers.yaml`
   - `parol6_moveit_config/config/ros2_controllers.yaml`
2. These are allowed to keep `position + velocity` interfaces for firmware/hardware pipelines.
3. Real mode should run with an external controller manager (hardware), not fake MoveIt controllers.

## Why This Prevents Future Conflicts

- Gazebo URDF and MoveIt sim launch now point to dedicated simulation configs.
- Hardware tuning can continue in hardware files without breaking simulation activation.
- Embedded branch changes to velocity interfaces no longer force Gazebo to fail.

## Operation Difference: Sim vs Real

### Simulation
- Time source: `/clock` from Gazebo (`use_sim_time=true`).
- Controller owner: Gazebo `controller_manager`.
- Launch for MoveIt: `use_fake_hardware:=false`.
- Interface expectation: `position` (as defined in simulation URDF ros2_control block).

### Real Hardware
- Time source: wall-clock (typically `use_sim_time=false`).
- Controller owner: real hardware ros2_control/hardware interface.
- MoveIt should connect to external hardware controllers.
- Interface expectation: can include `position + velocity` based on hardware plugin/firmware support.

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
ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false
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
See `docs/LAUNCH_METHODS.md` for separate launcher files for each operation mode.
