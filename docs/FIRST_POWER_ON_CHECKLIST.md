# PAROL6 First Power-On Checklist

Follow this checklist **in order** every time you connect the real robot for a new session.  
Do not skip phases — each phase verifies the next one is safe to proceed.

---

## Phase 0 — Pre-power hardware checks

- [ ] Teensy 4.1 connected to host USB (`/dev/ttyACM0` or `ttyACM1`)
- [ ] All stepper drivers powered (24 V PSU on, LED indicators lit)
- [ ] Enable (EN) signal from drivers confirmed routed to Teensy control
- [ ] All encoder cables seated (MT6816 boards, 5 V)
- [ ] Limit switch cables connected to Teensy digital inputs (if using homing today)
- [ ] E-stop wired and functional — test it manually before any motion
- [ ] Robot in a pose where **all joints have room to move in both directions** (mid-range)
- [ ] Workspace clear — nobody within the robot's reach

---

## Phase 1 — Serial link verification (no motion)

### 1a. Flash the firmware

Open the Firmware Configurator GUI:
```bash
./scripts/launchers/launch_configurator.sh
```
Go to **⚡ Flash** tab → **Generate config.h** → review validation warnings → **Flash**.

> **Check before flashing:**
> - `FEEDBACK_RATE_HZ` = 25 (match `ROS_COMMAND_RATE_HZ`)
> - `LIMIT_ENABLED` = all false for first session (enable after verifying wiring polarity)
> - `HOME_OFFSETS_RAD` = real physical values (or leave 0.0 and note post-homing pose)

### 1b. Open serial monitor

In the GUI, go to **🔌 Serial** tab → select port `/dev/ttyACM0` → baud `115200` → **Connect**.

Expected output immediately after boot:
```
(no output — firmware is silent until it receives <ENABLE>)
```

### 1c. Send ENABLE

In the serial tab command box, type and send:
```
<ENABLE>
```

Expected: no reply (that's correct — `<ENABLE>` just clears SOFT_ESTOP, no acknowledgement).

Then watch for feedback packets appearing at 25 Hz:
```
<ACK,0,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0>
```

✅ **Pass**: Feedback packets appear at ~25 Hz with stable `lim_state=0`  
❌ **Fail**: No output → check USB cable, port, and that firmware was flashed successfully

---

## Phase 2 — Direction verification (motor power ON, motors free)

> **Keep your hand on the E-stop.** Each joint will move a small angle.

### 2a. Jog J1 positive

In the GUI **🕹 Jog** tab:
- Set step = `0.1 rad`
- Press **J1 +▶**

Expected: J1 motor rotates in the direction that **increases** joint angle per your URDF convention.

If it moves the **wrong direction**: toggle `DIR_INVERT[0]` in the GUI → re-flash → retest.

### 2b. Repeat for J2–J6

Test each joint individually. Record the actual positive direction observed:

| Joint | Expected positive | Observed | DIR_INVERT correct? |
|-------|------------------|----------|---------------------|
| J1 | CCW from above | | |
| J2 | Shoulder forward | | |
| J3 | Elbow forward | | |
| J4 | Wrist CW | | |
| J5 | Wrist up | | |
| J6 | Tool CW | | |

> After any DIR_INVERT change → reflash → resend `<ENABLE>` → retest.

### 2c. Verify encoder sign matches motor direction

While jogging, check the position readout in the **📈 Oscilloscope** tab.  
**Joint position should increase when jogging positive.** If it decreases, `GEAR_RATIOS` sign or encoder wiring needs correction.

---

## Phase 3 — ROS interface verification (no trajectory execution yet)

### 3a. Launch real hardware stack

```bash
PAROL6_SERIAL_PORT=/dev/ttyACM0 ./scripts/launchers/launch_moveit_real_hw.sh
```

Wait for:
```
Controller manager ready — starting MoveIt + RViz...
```

### 3b. Confirm joint state echoes in RViz

RViz should show the robot in its **current physical pose** (not all zeros unless it's physically at zero).

```bash
# Verify joint states are live
ros2 topic hz /joint_states
# Expected: ~25 Hz
```

### 3c. Confirm controllers are active

```bash
ros2 control list_controllers
# Expected:
#   parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
#   joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster]      active
```

---

## Phase 4 — First trajectory (slow, small motion)

> Do not proceed until Phase 3 passes completely.

### 4a. Plan a small motion in RViz

In RViz → **MotionPlanning** panel:
1. Set **Planning Group**: `parol6_arm`
2. In **Query** tab, set **Start State** = current
3. Move the end-effector marker **≤ 5 cm** from its current position
4. Click **Plan**
5. Visually verify the planned path in RViz stays within the workspace

### 4b. Execute

Click **Execute**.

Watch in the serial tab:
- `STALE_CMD` should not appear (indicates the ROS sequence counter is working)
- Encoder positions should track the planned trajectory

**Success criteria:**
- Robot reaches the target within the goal tolerance (~0.15 rad)
- No `FAULT` or `SOFT_ESTOP` messages
- No joint hits a physical limit

---

## Phase 5 — Homing (only after Phases 1–4 pass)

> Only do this if you physically have limit switches wired and verified.

### 5a. Enable limit switches in config

Go to GUI → **⚙️ Features** tab → enable limit switches for the joints that have them.  
Set correct `Switch Type`, `Pull`, and `Polarity` per the wiring table in `parol6_firmware/README.md`.

**Reflash after any limit switch config change.**

### 5b. Verify switch signal before homing

In the GUI serial tab, manually trigger each limit switch by hand.  
The `lim_state` bitmask in the feedback packet should change from `0` to `1<<axis`.

Example for J1: `<ACK,12,0.00,...,1>` (lim_state=1 = bit 0 set)

### 5c. Set HOME_OFFSETS_RAD

Measure the actual joint angle at the switch-trigger position.  
Update `HOME_OFFSETS_RAD` in the GUI → reflash.

### 5d. Run homing

In the GUI **🏠 Jog** tab → **Home All** button (or send `<HOME>` via serial).

Expected sequence:
```
(robot moves slowly toward each switch in HOMING_ORDER)
HOMING_DONE
```

After `HOMING_DONE`: verify the robot is at the correct physical pose and joint states show `HOME_OFFSETS_RAD`.

---

## Troubleshooting Quick Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| No feedback packets after ENABLE | USB not connected / wrong port / firmware not flashed | Check port, reflash |
| `SOFT_ESTOP` immediately | Command timeout before ROS connects | Send `<ENABLE>` again after ROS launches |
| `FAULT: Runaway Velocity` | `MAX_VEL_RAD_S` too low or encoder signal bad | Check encoder cables, raise limit if physically safe |
| `STALE_CMD` in serial | Command sequence number regressed | Normal after reconnect — send `<ENABLE>` to reset |
| Joint moves wrong direction | `DIR_INVERT` or `ROS_DIR_INVERT` mismatch | Verify in jog test, toggle and reflash |
| Position tracks encoder but motor stalls | KP too low | Increase `KP_GAINS` for that joint incrementally |
| Oscillation / hunting | KP too high or alpha-beta filter lag | Lower KP, raise AB_ALPHA slightly |
| RViz shows wrong pose | `ROS_DIR_INVERT` disagrees with `DIR_INVERT` | Set both consistently, reflash |
| `HOMING_FAULT` | Limit switch not triggered within `max_travel` ticks | Check switch wiring, lower `HOMING_SPEED`, increase `max_travel` |

---

**Last Updated**: 2026-03-11  
**Maintained by**: PAROL6 Team
