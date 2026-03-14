# Launch Methods (Separate Operation Modes)

Defines one launcher per operation mode. Each launcher is **Docker-aware** — it detects whether it is being invoked from inside or outside the container and adapts accordingly.

## Launcher Files

| Script | Description |
|--------|-------------|
| `scripts/launchers/launch_gazebo_only.sh` | Physics simulation only (no MoveIt) |
| `scripts/launchers/launch_moveit_with_gazebo.sh` | MoveIt planning into a running Gazebo world |
| `scripts/launchers/launch_moveit_fake.sh` | MoveIt + RViz with fake controllers (no hardware) |
| `scripts/launchers/launch_moveit_real_hw.sh` | MoveIt + RViz connected to the real Teensy hardware |
| `scripts/launchers/launch_moveit_real_hw_tested_single_motor.sh` | Legacy tested single-motor real hardware bringup copied from `Tested_Working_SingleMotor_Integration(Day4)` |

## How Launchers Work

All launcher scripts share this execution logic:

```
if running inside Docker container (/.dockerenv exists):
    source /workspace/install/setup.bash
    ros2 launch <package> <launchfile> <args>
else (running on host):
    xhost +local:root  (grant X11 access)
    ./start_container.sh  (boot / attach parol6_dev)
    docker exec -i parol6_dev bash -c "source ... && ros2 launch ..."
```

> [!IMPORTANT]
> The GUI's **ROS2 Launch** tab calls these scripts directly from **inside** the container.
> The scripts auto-detect this and skip the `docker exec` path.
> If you run them from the host terminal, they will start/attach the container automatically.

## Method 1: Gazebo Only (Simulation World)

Use when you want physics simulation and robot visualization without MoveIt.

```bash
./scripts/launchers/launch_gazebo_only.sh
```

What it starts:
- `ros2 launch parol6 ignition.launch.py`

What it does **not** start:
- MoveIt / RViz

## Method 2: Gazebo AND MoveIt (Simulated)

Use when you want to run the full simulated software stack (Physics + Motion Planning) locally without hardware.

```bash
./scripts/launchers/launch_moveit_with_gazebo.sh
```

What it starts:
1. `ros2 launch parol6 ignition.launch.py` (in background)
2. `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false` (in foreground, after 5s delay)

> Why `use_fake_hardware:=false`: MoveIt connects directly to the Gazebo controller manager.

## Method 3: MoveIt Fake (RViz Only, No Hardware)

Use for motion planning validation, URDF checking, and UI testing without any hardware or physics.

```bash
./scripts/launchers/launch_moveit_fake.sh
```

What it starts:
- `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=true`

Behavior:
- Spawns `mock_components/GenericSystem` (the Ignition plugin is automatically swapped out at launch time)
- Robot state echoes commands instantly (perfect tracking)
- Good for checking joint limits, collision geometry, and path planning

## Method 4: MoveIt Real Hardware

Use when the Teensy 4.1 is physically connected via USB.

```bash
./scripts/launchers/launch_moveit_real_hw.sh
```

What it starts (two sequential launch files):
1. `ros2 launch parol6_hardware real_robot.launch.py` — hardware driver + controller manager + joint_state_broadcaster + parol6_arm_controller
2. `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false` — MoveIt + RViz connected to the running controller manager

> [!IMPORTANT]
> **Do not** run `real_robot.launch.py` separately AND then trigger the GUI launcher — the controller spawner runs from `real_robot.launch.py`. Running it twice causes a `Controller already loaded from active state` crash. The launcher handles both launches in the correct order.

The script now actually does this sequence. It starts `real_robot.launch.py` in the background, **polls `/controller_manager/list_controllers`** until it appears (instead of a fixed sleep), waits one extra second for controllers to fully activate, and only then starts MoveIt/RViz.

Prerequisites:
- Teensy 4.1 connected to host USB
- Device node allocated as `/dev/ttyACM0` (or set via env override)
- Docker container has `/dev/ttyACM0` passed through (`--device` flag in `start_container.sh`)

Optional environment overrides:

```bash
PAROL6_SERIAL_PORT=/dev/ttyACM1 PAROL6_BAUD_RATE=115200 ./scripts/launchers/launch_moveit_real_hw.sh
```

## Method 5: Real Hardware (Tested Single-Motor Legacy)

Use this if you need the exact launch behavior that was validated on real hardware in branch `Tested_Working_SingleMotor_Integration(Day4)`.

```bash
./scripts/launchers/launch_moveit_real_hw_tested_single_motor.sh
```

What it starts:
- `ros2 launch parol6_hardware real_robot_tested_single_motor.launch.py`

Behavior:
- This launch includes MoveIt/RViz internally, matching the tested branch layout.
- It uses isolated copies:
  - `parol6_hardware/launch/real_robot_tested_single_motor.launch.py`
  - `parol6_moveit_config/launch/demo_tested_single_motor.launch.py`
  - `parol6_moveit_config/config/ros2_controllers_tested_single_motor.yaml`
- Existing methods (1-4) are unchanged.

If GUI shows:
```text
file 'real_robot_tested_single_motor.launch.py' was not found in the share directory
```
rebuild in `parol6_dev` so installed launch files are refreshed:
```bash
docker exec -i parol6_dev bash -lc "cd /workspace && source /opt/ros/humble/setup.bash && colcon build --packages-select parol6 parol6_hardware parol6_moveit_config"
```

### Hardware Telemetry Protocol

The hardware interface (`parol6_hardware/src/parol6_system.cpp`) communicates with the Teensy over serial at 115200 baud.

**Command (ROS → Teensy)** (sent by `parol6_system.cpp` at `ROS_COMMAND_RATE_HZ`):
```
<SEQ,J1_pos,J2_pos,J3_pos,J4_pos,J5_pos,J6_pos,J1_vel,J2_vel,J3_vel,J4_vel,J5_vel,J6_vel>
```
Frames with a sequence number not strictly greater than the last accepted frame are rejected with:
```
STALE_CMD
```

**Feedback (Teensy → ROS)** (sent at `FEEDBACK_RATE_HZ`):
```
<ACK,SEQ,J1_pos,J2_pos,J3_pos,J4_pos,J5_pos,J6_pos,J1_vel,J2_vel,J3_vel,J4_vel,J5_vel,J6_vel,lim_state>
```
- All position values in radians, velocity in rad/s
- `lim_state` — bitmask: bit 0=J1 … bit 5=J6 limit switch triggered
- The hardware interface applies kinematic sign correction for J1, J3, J6 before sending and after receiving

**Special commands:**
```
<HOME>      → Start homing sequence (replies HOMING_DONE or HOMING_FAULT)
<ENABLE>    → Clear SOFT_ESTOP
```

## Recommended Startup Order

### Gazebo + MoveIt Simulation

```bash
# This script now automatically starts BOTH Gazebo and MoveIt
./scripts/launchers/launch_moveit_with_gazebo.sh
```

### Real Hardware + MoveIt

```bash
./scripts/launchers/launch_moveit_real_hw.sh
```

### Real Hardware + MoveIt (Tested Single-Motor Legacy)

```bash
./scripts/launchers/launch_moveit_real_hw_tested_single_motor.sh
```

### Fake MoveIt Only

```bash
./scripts/launchers/launch_moveit_fake.sh
```

## Shutdown & Cleanup

1. Close the ROS 2 Configurator GUI.
2. If running scripts manually, press `Ctrl+C` in the launcher terminal.
3. All child processes (MoveIt, RViz, controller manager) are terminated by the launch system automatically.
4. Optional hard stop:
```bash
docker stop parol6_dev
```

### ☠️ GUI "Kill All" Feature

If you experience issues like **RViz freezing**, **"Requesting initial scene failed"**, or terminal lockups, it means there are dangling / zombie ROS 2 nodes running silently in the container's background fighting over the network.

Click the **☠️ Kill All** button at the bottom of the ROS 2 Launch Tab in the Firmware Configurator GUI. This safely executes a mass `pkill` targetting Gazebo, MoveIt, and RViz inside the container to give you a clean slate without having to restart the entire Docker environment.
