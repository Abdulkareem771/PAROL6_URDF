# Manual Path Injection — Full Reference

> **Key finding from logs:** The robot DOES move to home successfully.
> Cartesian planning fails because MoveIt sees an **empty JointState** (sim_time mismatch).
> The fix is to set `use_sim_time` before launching vision, and ensure the
> joint states are flowing from Gazebo before calling Cartesian path.

---

## Quick Start (use the launcher)

```bash
cd ~/Desktop/PAROL6_URDF
./scripts/inject_path.sh
```

See [inject_path.sh](../../scripts/inject_path.sh) for full source.

---

## Full Manual Sequence

### Terminal 0 — Kill everything first
```bash
docker exec parol6_dev bash -c \
  "pkill -9 -f 'ign|gazebo|rviz|move_group|robot_state|controller|red_line|depth_match|path_gen|moveit_con|ros2_bag' 2>/dev/null; sleep 2; echo KILLED"
```

---

### Terminal 1 — Start Gazebo
```bash
docker exec -it parol6_dev bash -lc \
  "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && ros2 launch parol6 ignition.launch.py"
```

**Wait for:**
```
[spawner_parol6_arm_controller]: Configured and activated parol6_arm_controller
```
**Common warnings (safe to ignore):**
- `[QT] file::qml/IgnCard.qml: TypeError: Cannot read property ...` — Qt rendering warnings, harmless.
- `[RTPS_TRANSPORT_SHM Error] Failed init_port` — DDS shared memory warning, does not affect ROS 2 communication.

---

### Terminal 2 — Start MoveIt + RViz
```bash
docker exec -it parol6_dev bash -lc \
  "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py use_fake_hardware:=false"
```

**Wait for:**
```
[move_group]: You can start planning now!
```

**Important — `use_fake_hardware:=false`** tells MoveIt NOT to spawn its own controller stack.
It will connect to Gazebo's `parol6_arm_controller` instead.

---

### Terminal 3 — Set sim time (CRITICAL — do before vision)
```bash
docker exec parol6_dev bash -lc \
  "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && \
   ros2 param set /move_group use_sim_time true && \
   ros2 param set /rviz2 use_sim_time true && \
   echo SIM_TIME_OK"
```

> ⚠️ **Why this matters:** Gazebo publishes `/clock`. If `move_group` uses wall-clock
> but joint states use sim-time, the robot state lookup returns **empty** — causing
> `Found empty JointState message` and Cartesian IK failures (fraction ≈ 3.7%).

**Then start vision in the same terminal:**
```bash
docker exec -it parol6_dev bash -lc \
  "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && ros2 launch parol6_vision vision_moveit.launch.py"
```

**Wait for:**
```
[path_generator]:   Generated path with 27 waypoints
[moveit_controller]: STARTING WELDING SEQUENCE
```

---

### Terminal 4 — Inject test path
```bash
docker exec parol6_dev bash -lc "
source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && python3 -c \"
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time

rclpy.init()
node = Node('test_path_pub')
pub = node.create_publisher(Path, '/vision/welding_path', 10)

path = Path()
path.header.frame_id = 'base_link'
path.header.stamp = node.get_clock().now().to_msg()

# Straight line along X, at height confirmed by FK at home (joints=[0,0,0,0,0,0])
# FK result: position=(0.200, 0.000, 0.334), orientation=(x=0.7071, z=-0.7071)
for x in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    ps = PoseStamped()
    ps.header = path.header
    ps.pose.position.x = x
    ps.pose.position.y = 0.0
    ps.pose.position.z = 0.334
    ps.pose.orientation.x = 0.7071068
    ps.pose.orientation.y = 0.0
    ps.pose.orientation.z = -0.7071068
    ps.pose.orientation.w = 0.0
    path.poses.append(ps)

time.sleep(1.5)
pub.publish(path)
print('Test path published — 6 waypoints from x=0.20 to x=0.30')
time.sleep(1.0)
node.destroy_node()
rclpy.shutdown()
\"
"
```

---

### Terminal 4 — Trigger execution
```bash
docker exec parol6_dev bash -lc \
  "source /opt/ros/humble/setup.bash && cd /workspace && source install/setup.bash && \
   ros2 service call /moveit_controller/execute_welding_path std_srvs/srv/Trigger '{}'"
```

**Expected response:**
```
std_srvs.srv.Trigger_Response(success=True, message='Execution started')
```

---

## What the Logs Mean

### move_group terminal (Terminal 2)

| Log line | Meaning |
|----------|---------|
| `Goal request accepted!` | Gazebo controller accepted the trajectory ✅ |
| `Controller 'parol6_arm_controller' successfully finished` | Robot completed the move ✅ |
| `Completed trajectory execution with status SUCCEEDED` | Full success ✅ |
| `Found empty JointState message` | sim_time not set — robot state unknown ❌ |
| `followed 3.703704% of requested trajectory` | Cartesian IK failed for most waypoints ❌ |

### vision terminal (Terminal 3)

| Log line | Meaning |
|----------|---------|
| `Move to home succeeded` | Phase 1 done, robot is at [0,0,0,0,0,0] ✅ |
| `Success! Planned fraction: 1.00` | All waypoints reachable, trajectory ready ✅ |
| `Phase 3: Executing Weld` | Trajectory sent to Gazebo controller ✅ |
| `Sequence Complete` | Robot finished the weld path ✅ |
| `Fraction too low: 0.02 < 0.95` | Waypoints not reachable — wrong orientation or out-of-range ❌ |
| `All planning attempts failed` | All 3 step-size attempts failed ❌ |

---

## Waypoint Design Rules

The safe reachable workspace for `orientation=(x=0.7071, z=-0.7071)` (EE pointing down):

```
x: 0.15 → 0.45 m   (forward from base)
y: -0.20 → 0.20 m  (lateral)
z: 0.25 → 0.50 m   (height)
```

**Home position (all joints = 0) confirmed by FK:**
```
x=0.200, y=0.000, z=0.334
orientation: x=0.7071068, y=0, z=-0.7071068, w=0
```

Path examples:
```python
# Along X (default test):
for x in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    pos = (x, 0.0, 0.334)

# Lateral sweep:
for y in [-0.15, -0.08, 0.0, 0.08, 0.15]:
    pos = (0.28, y, 0.334)

# Diagonal:
import numpy as np
for t in np.linspace(0, 1, 10):
    pos = (0.20 + t*0.15, -0.10 + t*0.20, 0.334)
```

---

## Known Issues and Fixes

### `Found empty JointState message`
**Cause:** `use_sim_time` not set on `move_group` before Cartesian planning.  
**Fix:** Always run `ros2 param set /move_group use_sim_time true` **before** vision launch.

### `Fraction too low: 0.02`
**Cause:** One of:
- Wrong orientation quaternion (was `y=0.7071, w=0.7071` — fixed to `x=0.7071, z=-0.7071`)
- Missing `link_name` in GetCartesianPath request (fixed in `moveit_controller.py`)
- Waypoints outside reachable workspace

### `Execution already in progress`
**Cause:** Previous sequence still running.  
**Fix:** Wait ~2 minutes, or restart vision launch.

### `No path received yet`
**Cause:** Publish path (Step 1) before calling trigger (Step 2).

### rclpy shutdown errors on Ctrl+C
```
rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called
```
**Safe to ignore** — Ctrl+C sends SIGINT to all nodes simultaneously; all nodes shut down.

---

## Files Changed During Debug

| File | What Changed |
|------|-------------|
| `parol6_vision/parol6_vision/path_generator.py` | `compute_orientation()`: fixed quaternion from `(y=0.7071, w=0.7071)` → `(x=0.7071, z=-0.7071)` |
| `parol6_vision/parol6_vision/moveit_controller.py` | Added `req.link_name = self.ee_link` and `req.header.stamp` to Cartesian path request |
