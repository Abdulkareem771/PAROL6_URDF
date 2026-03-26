# PAROL6 Path Injection Troubleshooting Log
**Session Date:** 2026-03-03  
**Goal:** Kill stale processes, restart full stack (Gazebo + MoveIt + Vision), inject vision-generated path, make robot move.

---

## 1. Initial State — Terminal Inspection

**What was checked:** Read terminal output from PID 9380.

**Findings:**
- Gazebo had received `SIGINT (signal 2)` and shut down cleanly.
- Both `joint_state_broadcaster` and `parol6_arm_controller` had been deactivated before crash.
- The terminal was back at `$` prompt — **no simulation running**.
- Two stale background processes were still running:
  - FK service call (running 17+ minutes)
  - Vision `moveit_controller` Python script (running 13+ minutes)

---

## 2. Kill All Stale Processes

**Commands run:**
```bash
docker exec parol6_dev bash -c "pkill -9 -f 'ros2|ign|gazebo|rviz|move_group|robot_state_publisher|controller_manager|red_line|depth_match|path_gen|moveit_con|ros2_bag'"
```

**Result:** Exit code 137 (expected — pkill kills processes including the shell).  
**Container status:** `docker ps` confirmed container `parol6_dev` still `Up`.

---

## 3. Start Gazebo (Step 2 per QUICK_START.md)

**Command:**
```bash
docker exec -d parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6 ignition.launch.py > /tmp/gazebo.log 2>&1"
```

**Waited:** 15 seconds for Gazebo to initialize.

**Log result (`/tmp/gazebo.log`):**
```
[controller_manager]: Loading controller 'joint_state_broadcaster'
[spawner_joint_state_broadcaster]: Configured and activated joint_state_broadcaster
[controller_manager]: Loading controller 'parol6_arm_controller'
[spawner_parol6_arm_controller]: Configured and activated parol6_arm_controller
```
✅ **Both controllers ACTIVE.**

---

## 4. Start MoveIt/RViz (Step 3 per QUICK_START.md)

**Command:**
```bash
docker exec -d parol6_dev bash -lc "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false > /tmp/moveit.log 2>&1"
```

**Waited:** 15 seconds.

**Log result (`/tmp/moveit.log`):**
```
[move_group]: You can start planning now!
[rviz2]: Ready to take commands for planning group parol6_arm.
```
✅ **MoveIt fully initialized.**

---

## 5. Set Simulation Time (Step 4 per QUICK_START.md)

**Commands:**
```bash
ros2 param set /move_group use_sim_time true   # ✅ Set parameter successful
ros2 param set /rviz2 use_sim_time true        # ✅ Set parameter successful
```

---

## 6. Start Vision Pipeline (Step 5 — vision_moveit.launch.py)

**Command:**
```bash
docker exec -d parol6_dev bash -lc "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && ros2 launch parol6_vision vision_moveit.launch.py > /tmp/vision.log 2>&1"
```

**What vision_moveit.launch.py does:**
- Plays rosbag: `/workspace/rosbag2_2026_01_26-23_26_59` (loops, remapping TFs to discard bag TFs)
- Starts `red_line_detector` → publishes `/vision/weld_lines_2d`
- Starts `depth_matcher` → publishes `/vision/weld_lines_3d`
- Starts `path_generator` → publishes `/vision/welding_path` (nav_msgs/Path)
- Starts `moveit_controller` → subscribes `/vision/welding_path`, calls MoveIt to execute

**Vision pipeline result:**
```
[red_line_detector]: Detected 1 line, confidence=1.45, 130 pixels
[depth_matcher]: depth samples ~1096mm, 130 valid points, quality=1.00
[depth_matcher]: Published 1 3D weld lines
[path_generator]: Generated path with 27 waypoints
[moveit_controller]: Received path with 27 points
[moveit_controller]: STARTING WELDING SEQUENCE
```

---

## 7. First Failure — Cartesian Planning: fraction=0.02

**Error:**
```
[moveit_controller]: Reachability normalization: modified 0/27 points
[moveit_controller]: Attempt 1: Step=2.0mm  → Fraction too low: 0.02 < 0.95
[moveit_controller]: Attempt 2: Step=5.0mm  → Fraction too low: 0.02 < 0.95
[moveit_controller]: Attempt 3: Step=10.0mm → Fraction too low: 0.02 < 0.90
[moveit_controller]: All planning attempts failed
```

**Investigation:**
- `enforce_reachable_test_path=True` in the moveit_controller parameters.
- Clamp box: `x=[0.20,0.65], y=[-0.35,0.35], z=[0.10,0.55]`
- **0/27 points were clamped** → positions passed the box check.
- Checked `path_generator.py` → `compute_orientation()` was hardcoded:
  ```python
  return Quaternion(x=0.0, y=0.7071068, z=0.0, w=0.7071068)
  ```
  This means EE Z-axis pointing **+X forward**.

---

## 8. FK Test — Find True EE Orientation

**Command:**
```bash
ros2 service call /compute_fk moveit_msgs/srv/GetPositionFK \
  '{...joints: [0,0,0,0,0,0]}'
```

**Result:**
```
position: (x=0.200, y=0.000, z=0.334)
orientation: x=0.7071, y≈0, z=-0.7071, w≈0
```

**Conclusion:** PAROL6 natural EE orientation at home (all joints=0) is **`(x=0.7071, z=-0.7071)`**, NOT `(y=0.7071, w=0.7071)`.

---

## 9. Fix #1 — Correct Orientation in path_generator.py

**File:** `parol6_vision/parol6_vision/path_generator.py`  
**Function:** `compute_orientation()`

**Change:**
```python
# BEFORE (wrong — EE pointing forward +X, kinematically infeasible):
return Quaternion(x=0.0, y=0.7071068, z=0.0, w=0.7071068)

# AFTER (correct — EE pointing down, confirmed by FK at home):
return Quaternion(x=0.7071068, y=0.0, z=-0.7071068, w=0.0)
```

**Rebuilt package:**
```bash
colcon build --packages-select parol6_vision --symlink-install
# Result: Finished [0.91s] ✅
```

---

## 10. Fix #2 — Missing `link_name` in GetCartesianPath Request

**File:** `parol6_vision/parol6_vision/moveit_controller.py`  
**Function:** `plan_cartesian_with_fallback()`

**Problem:** `req.link_name` was never set → MoveIt didn't know which link to do IK for.

**Change:**
```python
req.link_name = self.ee_link  # 'L6'
req.header.stamp = self.get_clock().now().to_msg()  # Use live clock
```

---

## 11. move_group Accidentally Killed

**Issue:** Using `pkill -f ros2` to restart vision nodes also killed `move_group` and `rviz2`.

**Fix:** Restarted MoveIt separately:
```bash
docker exec -d parol6_dev bash -lc "... ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false > /tmp/moveit2.log 2>&1"
```
**Result:** `move_group` came back → "You can start planning now!" ✅

**Lesson Learned:** When restarting vision nodes, kill only vision-specific processes by name:
```bash
pkill -9 -f 'red_line_detector|depth_matcher|path_generator|moveit_controller|ros2_bag'
```

---

## 12. Waypoint Debug Logging Added

Added temporary debug prints to `path_generator.py` to see first 5 waypoint xyz:

**Output:**
```
WP[0]: x=0.345  y=-0.163  z=0.345  q=(0.707, 0.000, -0.707, 0.000)
WP[1]: x=0.344  y=-0.167  z=0.342
WP[2]: x=0.343  y=-0.172  z=0.338
WP[3]: x=0.343  y=-0.176  z=0.335
WP[4]: x=0.342  y=-0.180  z=0.332
```

**Positions look reasonable** — inside the workspace bounding box.

---

## 13. IK Test — Confirm Positions are NOT Reachable

**Command:**
```bash
ros2 service call /compute_ik moveit_msgs/srv/GetPositionIK '{
  ik_request: {
    group_name: "parol6_arm", ik_link_name: "L6",
    pose_stamped: {
      header: {frame_id: "base_link"},
      pose: {position: {x: 0.345, y: -0.163, z: 0.345},
             orientation: {x: 0.7071068, y: 0.0, z: -0.7071068, w: 0.0}}
    }
  }
}'
```

**Result:** `error_code: val=-31` = `NO_IK_SOLUTION`

**Conclusion:** Even with the corrected orientation, position `(0.345, -0.163, 0.345)` has no valid IK. The vision-generated 3D points from the bag are **out of the robot reachable workspace** despite passing the bounding box clamp check.

---

## 14. Manual Test Path Injection

Published a known-good 5-waypoint path near the FK-confirmed home position:
```python
waypoints = [
  (0.20, 0.0, 0.334),
  (0.23, 0.0, 0.334),
  (0.26, 0.0, 0.334),
  (0.29, 0.0, 0.334),
  (0.32, 0.0, 0.334),
]
orientation = (x=0.7071068, y=0, z=-0.7071068, w=0)
```
Published to `/vision/welding_path` via custom Python node.
Then called:
```bash
ros2 service call /moveit_controller/execute_welding_path std_srvs/srv/Trigger '{}'
# Response: success=True, message='Execution started'
```

**Status:** Result pending — monitoring `/tmp/vision_final.log`.

---

## 15. Root Cause Summary

| # | Problem | Status |
|---|---------|--------|
| 1 | Wrong orientation quaternion in `path_generator.py` (y=0.7071 instead of x=0.7071,z=-0.7071) | ✅ Fixed |
| 2 | Missing `link_name` field in `GetCartesianPath` request | ✅ Fixed |
| 3 | Vision-generated 3D waypoints have no IK solution (camera TF may be miscalibrated) | 🔍 Investigating |
| 4 | `pkill -f ros2` kills move_group when restarting vision | ✅ Documented |

---

## 16. Files Modified

| File | Change |
|------|--------|
| `parol6_vision/parol6_vision/path_generator.py` | Fixed `compute_orientation()` quaternion; added debug logging |
| `parol6_vision/parol6_vision/moveit_controller.py` | Added `req.link_name` and `req.header.stamp` to Cartesian path request |

---

## 17. Next Steps (If Robot Still Doesn't Move)

1. **Fix camera TF** — The Kinect at `x=1.44m, z=0.10m` with depth=1.096m back-projects to coordinates outside actual robot reach. Reduce camera X position in `vision_moveit.launch.py`:
   ```python
   arguments=['--x', '0.60', '--y', '0.0', '--z', '0.65', ...]
   ```

2. **Verify IK at all 27 waypoints** — Run `/compute_ik` service call for each waypoint to identify which are reachable.

3. **Tighten workspace clamp** in `moveit_controller` launch parameters to match true robot reachable sphere (max ~0.55m radius from base).

4. **Visualize path in RViz** — Enable Path and Marker displays to see where the 27 waypoints land relative to the robot.
