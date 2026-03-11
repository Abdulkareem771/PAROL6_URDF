# MoveIt Controller Debug Context (Pipeline Execution)

## Current Failure Signature (resolved in code)

Observed in `logs/launchers/<timestamp>/vision.log`:
- `Execution worker exception: generator already executing`
- Python traceback from `rclpy.executors` ending with:
  - `ValueError: generator already executing`

This happened because `moveit_controller` was calling `rclpy.spin_until_future_complete(...)`
from worker thread while main thread was already spinning the node.

## Fix Applied

File:
- `parol6_vision/parol6_vision/moveit_controller.py`

Changes:
- Replaced nested `spin_until_future_complete` waits with thread-safe event waits on async futures.
- Added robust `None` and timeout handling for goal handles/results.
- Kept async execution thread model so callbacks do not block executor.

## Validation Commands

Rebuild:
```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select parol6_vision"
```

Relaunch full stack:
```bash
./scripts/launchers/stop_all_vision_gazebo.sh
./scripts/launchers/launch_all_vision_gazebo.sh
```

Inject known reachable path:
```bash
./scripts/launchers/inject_reachable_weld_path.sh
```

Watch only moveit_controller lines:
```bash
LATEST=$(ls -1dt logs/launchers/* | head -n1)
rg -n "moveit_controller|STARTING WELDING SEQUENCE|Approach|Execution finished|exception|MoveIt Error" "$LATEST/vision.log" -S
```

Confirm command reached controller:
```bash
docker exec -it parol6_dev bash -lc "cd /workspace && source install/setup.bash && ros2 topic echo /parol6_arm_controller/joint_trajectory --once"
```

## Expected Success Indicators

- `Execution finished: success=True` in vision log.
- `/parol6_arm_controller/joint_trajectory` publishes points.
- Robot moves in Gazebo.

## Next-Session Context Template

Use this when continuing later:

- Goal: Validate `/vision/welding_path -> moveit_controller -> Gazebo motion`.
- Vision pipeline status: red detector + depth matcher + path generator healthy (26-27 waypoints).
- Prior blocker: `generator already executing` crash in `moveit_controller`.
- Fix: replaced nested rclpy spinning with event-based future waiting.
- Launch method: `./scripts/launchers/launch_all_vision_gazebo.sh`.
- Injection method: `./scripts/launchers/inject_reachable_weld_path.sh`.
- Log focus: `rg "moveit_controller|Execution finished|exception" logs/launchers/<latest>/vision.log`.
