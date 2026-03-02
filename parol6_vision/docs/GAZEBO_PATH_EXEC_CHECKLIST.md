# Gazebo Path Execution Checklist (Vision -> MoveIt -> Gazebo)

Goal: verify that `path_generator` publishes correct path poses and robot executes that path in Gazebo.

This checklist includes a **reachability test mode** in `vision_moveit.launch.py`:
- `enforce_reachable_test_path: true`
- It clamps path points into a conservative workspace before planning.
- Purpose: validate `path_generator -> moveit_controller -> Gazebo` connection even if raw path is physically out of reach.

## Required Terminals

## Terminal 1: Gazebo (controller owner)
```bash
cd ~/Desktop/PAROL6_URDF
./scripts/launchers/launch_gazebo_only.sh
```

## Terminal 2: MoveIt RViz (external controllers)
```bash
cd ~/Desktop/PAROL6_URDF
./scripts/launchers/launch_moveit_with_gazebo.sh
```

## Terminal 3: Vision + Bag + MoveIt controller
```bash
cd ~/Desktop/PAROL6_URDF
./scripts/launchers/launch_vision_bag_pipeline.sh
```

## One-Command Wrapper (all 3 together)

```bash
cd ~/Desktop/PAROL6_URDF
./scripts/launchers/launch_all_vision_gazebo.sh
```

Stop all:
```bash
./scripts/launchers/stop_all_vision_gazebo.sh
```

## Inject Reachable Path (Bypass Detector)

If detector/depth output is noisy or out-of-reach, inject a known good path:

```bash
cd ~/Desktop/PAROL6_URDF
./scripts/launchers/inject_reachable_weld_path.sh
```

This publishes `/vision/welding_path` in `base_link` frame with conservative points.
With current launch config (`auto_execute: true`), robot should start moving automatically.

## Verification Steps

### 1) Confirm path is generated
```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 topic echo /vision/welding_path --once"
```
Expected:
- message type: `nav_msgs/msg/Path`
- `poses:` list not empty
- frame should match planning base (`base_link` in current setup)

Quick pose count check:
```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 topic echo /vision/welding_path --once | grep -c 'position:'"
```
Expected: count >= 3

### 2) Trigger execution from moveit_controller
If `auto_execute: true` (default in current `vision_moveit.launch.py`), skip this step.

```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 service call /moveit_controller/execute_welding_path std_srvs/srv/Trigger '{}'"
```
Expected:
- `success: true` from service response

### 3) Confirm trajectory is sent to ros2_control
Run this, then call execute service again:
```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 topic echo /parol6_arm_controller/joint_trajectory --once"
```
Expected:
- trajectory message appears with non-empty `points`

### 4) Confirm Gazebo robot actually moves
```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 topic echo /joint_states --once"
```
Expected:
- joint positions change over time during execution
- motion visible in Gazebo window

## Failure Isolation (Where It Broke)

- If `/vision/welding_path` is empty: issue is in `red_line_detector -> depth_matcher -> path_generator`.
- If path exists but service fails: issue is `moveit_controller` planning/execution.
- If service succeeds but `/parol6_arm_controller/joint_trajectory` is empty: MoveIt not connected to active controller.
- If trajectory exists but no motion: Gazebo controller inactive or wrong controller manager owner.

## Reachability Test vs Raw Path

- **Connection Test Mode (default in `vision_moveit.launch.py`)**:
  - reachable clamp enabled
  - verifies plumbing and command flow
- **Raw Path Mode**:
  - set `enforce_reachable_test_path` to `false` in `vision_moveit.launch.py`
  - use when evaluating true geometric validity of detector output

## Critical Mode Rule

For Gazebo validation, MoveIt must run with external controllers:
```bash
ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false
```
Using `use_fake_hardware:=true` creates a separate controller manager and breaks Gazebo execution ownership.
