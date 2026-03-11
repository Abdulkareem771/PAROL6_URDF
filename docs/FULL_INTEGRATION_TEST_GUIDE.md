# Full Integration Test Guide

**Test the complete pipeline: RViz → MoveIt → ros2_control → Teensy 4.1**

This guide runs after `COMMUNICATION_TESTING_GUIDE.md` passes. Motors must be powered.

---

## Prerequisites

- [ ] Serial communication test passed (0% packet loss, < 15 ms latency)
- [ ] Direction verification complete (all joints jog in correct direction)
- [ ] Teensy flashed with production config.h (`FEEDBACK_RATE_HZ=25`, safety limits set)
- [ ] Real hardware launch stack tested → controllers active
- [ ] Robot in mid-range pose, workspace clear

---

## 🚀 Step 1 — Launch the real hardware stack

```bash
PAROL6_SERIAL_PORT=/dev/ttyACM0 ./scripts/launchers/launch_moveit_real_hw.sh
```

Wait for the terminal to print:
```
Controller manager ready — starting MoveIt + RViz...
```

Verify in a second terminal:
```bash
# Inside Docker
ros2 control list_controllers
```
Expected:
```
parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster]      active
```

---

## 🔍 Step 2 — Confirm live joint state

```bash
ros2 topic hz /joint_states
# Target: ~25 Hz

ros2 topic echo /joint_states --once
# Joint positions should match the robot's physical pose
```

Check RViz: the ghost robot should be in the **same pose as the physical robot**.  
If it shows all zeros but the robot is not at zero → `ROS_DIR_INVERT` or `HOME_OFFSETS_RAD` issue.

---

## Step 3 — Test 1: Single joint jog via ROS

Use the **🕹 Jog** tab in the GUI (which sends `<SEQ,pos...>` commands) OR use:

```bash
# Jog J1 by 0.1 rad using ros2 control
ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  '{
    joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6],
    points: [{
      positions: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
      velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      time_from_start: {sec: 2, nanosec: 0}
    }]
  }'
```

**Expected:**
- J1 moves 0.1 rad in the correct direction
- `/joint_states` position updates to ~0.1 rad for J1
- No `FAULT` or `SOFT_ESTOP` in serial tab

---

## Step 4 — Test 2: MoveIt plan and execute

In RViz → **MotionPlanning** panel:

1. **Start State**: Set to `<current>`
2. **Goal State**: Drag end-effector marker ~5 cm from current position
3. Click **Plan** — inspect the preview path, confirm it stays within workspace
4. Click **Execute**

### What to watch

**Serial tab** (Firmware Configurator):
- Packets should arrive continuously at 25 Hz
- `STALE_CMD` should NOT appear
- `lim_state` should stay `0`

**ROS terminal**:
```bash
ros2 topic echo /parol6_arm_controller/state --once
# error field should approach zero as robot reaches target
```

**Success**: Robot reaches target, no faults, no stalls.

---

## Step 5 — Test 3: Roundtrip latency measurement

While a trajectory is executing, measure the command loop:

```bash
# In Docker
ros2 topic hz /joint_states
ros2 topic hz /parol6_arm_controller/joint_trajectory
```

| Metric | Target | Acceptable |
|--------|--------|-----------|
| `/joint_states` rate | 25 Hz | ≥ 20 Hz |
| Goal tracking error | < 0.05 rad | < 0.15 rad |
| Trajectory completion | ≤ goal_time (5 s) | within 2× plan time |
| SOFT_ESTOP events | 0 | 0 |
| FAULT events | 0 | 0 |

---

## Step 6 — Test 4: Watchdog / ESTOP test

Simulate a connection drop and verify recovery:

1. With robot stationary, stop sending commands by **disconnecting the GUI serial tab**
2. Wait 300 ms (> `COMMAND_TIMEOUT_MS=200`)
3. Check serial tab → `SOFT_ESTOP` should appear in firmware state
4. Reconnect serial → send `<ENABLE>` → robot should accept commands again

> ✅ This confirms the watchdog protects against silent host crashes.

---

## Step 7 — Test 5: Velocity limit check

> ⚠️ Only if you want to manually confirm the supervisor works. Use with caution.

Send a trajectory with a very high velocity at a single joint that exceeds `MAX_VEL_RAD_S`:

```bash
# J1 max = 3.0 rad/s — request 5 rad/s briefly
ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  '{
    joint_names: [joint_L1, ...],
    points: [{
      positions: [0.5, 0.0, ...],
      velocities: [5.0, 0.0, ...],
      time_from_start: {sec: 0, nanosec: 100000000}
    }]
  }'
```

Expected: `FAULT: Runaway Velocity` in serial tab. Robot stops.  
Recovery: power cycle or reflash (this is a latched FAULT, not SOFT_ESTOP).

---

## ✅ Full integration pass criteria

| Test | Pass condition |
|------|---------------|
| 1. Single joint jog | Correct direction, correct magnitude |
| 2. MoveIt plan+execute | Reaches goal, no faults |
| 3. Latency | joint_states ≥ 20 Hz, tracking error < 0.15 rad |
| 4. Watchdog | SOFT_ESTOP on dropout, recovery on `<ENABLE>` |
| 5. Velocity limit | FAULT triggers, prevents runaway |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| RViz shows wrong pose | Check `ROS_DIR_INVERT` matches `DIR_INVERT` |
| Trajectory aborts immediately | `trajectory` constraint = 0.0 is too tight — set 0.05 in `ros2_controllers.yaml` |
| Goal time exceeded | Increase `goal_time` in `ros2_controllers.yaml` or reduce move distance |
| Robot overshoots | KP too high for that joint — reduce incrementally |
| SOFT_ESTOP during fast motion | `COMMAND_TIMEOUT_MS` too short — raise to 400 ms |
| Controller manager not starting | Check `real_robot.launch.py` for parameter errors; run with `--log-level debug` |

---

**Last Updated**: 2026-03-11  
**Maintained by**: PAROL6 Team
