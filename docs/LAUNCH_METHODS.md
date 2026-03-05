# Launch Methods (Separate Operation Modes)

This page defines one launcher per operation mode so each method can run alone.

## Launcher Files

- `scripts/launchers/launch_gazebo_only.sh`
- `scripts/launchers/launch_moveit_with_gazebo.sh`
- `scripts/launchers/launch_moveit_fake.sh`
- `scripts/launchers/launch_moveit_real_hw.sh`
- `scripts/launchers/launch_vision_bag_pipeline.sh`
- `scripts/launchers/launch_all_vision_gazebo.sh`
- `scripts/launchers/stop_all_vision_gazebo.sh`
- `scripts/launchers/inject_reachable_weld_path.sh`

All launchers:
- run from host side
- ensure GUI permissions (`xhost`)
- start/attach `parol6_dev`
- source `/workspace/install/setup.bash`

## Method 1: Gazebo Only (Simulation World)

Use when you want only physics simulation and robot visualization.

```bash
./scripts/launchers/launch_gazebo_only.sh
```

What it starts:
- `ros2 launch parol6 ignition.launch.py`

What it does not start:
- MoveIt RViz

## Method 2: MoveIt With Gazebo (External Controllers)

Use when Gazebo is already running and you want Plan/Execute into Gazebo.

```bash
./scripts/launchers/launch_moveit_with_gazebo.sh
```

What it starts:
- `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false`

Why `use_fake_hardware:=false`:
- MoveIt must connect to Gazebo controller manager.
- It must not spawn a second internal fake controller manager.

## Method 3: MoveIt Fake (Standalone)

Use when you want MoveIt planning without Gazebo or hardware.

```bash
./scripts/launchers/launch_moveit_fake.sh
```

What it starts:
- `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=true`

Behavior:
- Starts internal fake ros2_control stack.
- Good for pure planning and UI checks.

## Method 4: MoveIt Real Hardware Bridge Mode

Use when real hardware controller manager/driver is running (no Gazebo).

```bash
./scripts/launchers/launch_moveit_real_hw.sh
```

What it starts:
- `ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false`

Prerequisite:
- real hardware path/controller manager must be launched first (for example `start_real_robot.sh` or your real hardware launch pipeline).

## Method 5: Vision Bag Pipeline Only

Use when you want vision nodes + rosbag + moveit_controller, while Gazebo/MoveIt are started separately.

```bash
./scripts/launchers/launch_vision_bag_pipeline.sh
```

## Method 6: All Three Together (Gazebo + MoveIt + Vision)

Use when you want one command to start all parts with logs.

```bash
./scripts/launchers/launch_all_vision_gazebo.sh
```

Stop all:
```bash
./scripts/launchers/stop_all_vision_gazebo.sh
```

Inject a known reachable welding path for wiring validation:
```bash
./scripts/launchers/inject_reachable_weld_path.sh
```

## Recommended Startup Order Per Mode

### Gazebo + MoveIt Simulation

1. `./scripts/launchers/launch_gazebo_only.sh`
2. wait 10-15 seconds
3. `./scripts/launchers/launch_moveit_with_gazebo.sh`

### Real Hardware + MoveIt

1. start real hardware pipeline first
2. `./scripts/launchers/launch_moveit_real_hw.sh`

### Fake MoveIt Only

1. `./scripts/launchers/launch_moveit_fake.sh`

## Shutdown (Clean)

1. Press `Ctrl+C` in MoveIt terminal.
2. Press `Ctrl+C` in Gazebo terminal (if running).
3. Exit container shells with `exit`.
4. Optional:
```bash
docker stop parol6_dev
```
