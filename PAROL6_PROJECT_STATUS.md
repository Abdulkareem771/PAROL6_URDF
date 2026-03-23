# PAROL6 Robot Arm — Project Status & Configuration Reference

> **Last Updated:** 2026-03-11  
> **Purpose:** Reference document for AI agents and developers working on this project.

---

## 1. System Overview

| Component | Details |
|---|---|
| **Robot** | PAROL6 6-DOF robot arm |
| **MCU** | Teensy 4.1 (i.MXRT1062, 600 MHz Cortex-M7) |
| **Motor Drivers** | MKS SERVO42C (step/direction mode, closed-loop stepper) |
| **Encoders** | MT6816 (PWM output, 12-bit / 4096 positions per revolution) |
| **Software** | ROS 2 Humble, MoveIt 2, `ros2_control`, JointTrajectoryController |
| **Build** | Docker container (`start_real_robot.sh`), colcon workspace |
| **Communication** | Teensy USB Serial → ROS `parol6_hardware` plugin (LibSerial) |

### Architecture Diagram

```
MoveIt (plan) → JointTrajectoryController → parol6_hardware (write)
                                                    ↓ USB Serial
                                              Teensy 4.1 Firmware
                                           (control.cpp @ 500 Hz)
                                                    ↓
                                        MKS SERVO42C (step/dir)
                                                    ↓
                                        MT6816 encoder (PWM) → ISR
```

---

## 2. Key File Locations

| File | Purpose |
|---|---|
| `PAROL6/firmware/realtime_servo_teensy_old/config.h` | **Active firmware config** — pins, gear ratios, gains, speed limits |
| `PAROL6/firmware/realtime_servo_teensy_old/control.cpp` | Control loop (500 Hz), encoder reading, direction logic |
| `PAROL6/firmware/realtime_servo_teensy_old/serial_comm.cpp` | Serial protocol, arming, grace period |
| `PAROL6/firmware/realtime_servo_teensy_old/motor.cpp` | FlexPWM step generation, direction pins |
| `parol6_hardware/src/parol6_system.cpp` | ROS 2 hardware interface (read/write serial) |
| `parol6_hardware/config/parol6_controllers.yaml` | Controller manager config (update rate, tolerances) |
| `parol6_hardware/urdf/parol6.ros2_control.xacro` | Hardware interface URDF definition |
| `parol6_hardware/launch/real_robot.launch.py` | Launch file for real robot |
| `parol6_moveit_config/config/ros2_controllers.yaml` | MoveIt controller config |

> [!IMPORTANT]
> The **active firmware** is `realtime_servo_teensy_old`. There is also a `realtime_servo_teensy` (newer) and `realtime_servo_control` (ESP32), but the old Teensy firmware is what runs on the robot.

---

## 3. Current Hardware Configuration (config.h)

### Pin Assignments

| Joint | Step Pin | Dir Pin | Encoder Pin | Encoder Enabled |
|---|---|---|---|---|
| J1 | 2 | 24 | 14 | ❌ false |
| J2 | 4 | 35 | 12 | ❌ false |
| J3 | 5 | 34 | 11 | ❌ false |
| J4 | 8 | 27 | 17 | ❌ false |
| J5 | 7 | 40 | 19 | ✅ true |
| J6 | 6 | 28 | 18 | ❌ false |

### Motor Parameters

| Joint | Microsteps | Gear Ratio | Dir Sign | Steps/Rev (total) |
|---|---|---|---|---|
| J1 | 4 | 1.0 | +1 | 800 |
| J2 | 16 | 20.0 | -1 | 3200 |
| J3 | 16 | 16.5 | -1 | 3200 |
| J4 | 16 | 4.0 | +1 | 3200 |
| J5 | 16 | 1.0 | -1 | 3200 |
| J6 | 16 | 10.0 | +1 | 3200 |

### Control Parameters

| Joint | Kp | Kd | Max Velocity (rad/s) |
|---|---|---|---|
| J1 | 1.0 | 0.0 | 0.5 |
| J2 | 1.0 | 0.0 | 0.5 |
| J3 | 1.0 | 0.0 | 0.5 |
| J4 | 2.0 | 0.0 | 1.0 |
| J5 | 2.0 | 0.0 | 1.2 |
| J6 | 2.0 | 0.0 | 1.0 |

| Parameter | Value |
|---|---|
| Control loop | 500 Hz (2000 µs) |
| Feedback rate | 50 Hz (20000 µs) |
| Velocity deadband | 0.02 rad/s |
| Max step frequency | 20,000 Hz |
| Position error limit | 0.5 rad |

> [!WARNING]
> The **gear ratio comments contradict their values** (e.g., J2 says "Direct drive" but ratio is 20.0). The VALUES are believed correct; the comments are stale. Verify physically if uncertain.

---

## 4. Control Loop Math

The Teensy runs this at 500 Hz per joint:

```
velocity_command = desired_velocity + Kp × (desired_position − actual_position)
velocity_command = clamp(velocity_command, ±MAX_JOINT_VELOCITIES)

motor_vel   = |velocity_command| × GEAR_RATIO
step_freq   = (motor_vel × STEPS_PER_REV × MICROSTEPS) / (2π)
step_freq   = min(step_freq, 20000)  // hardware cap

forward     = (velocity_command × MOTOR_DIR_SIGN) >= 0
```

### Critical Speed Calculation (J2 example)

```
At MAX_JOINT_VELOCITIES = 0.5 rad/s:
  motor_vel  = 0.5 × 20.0 = 10.0 rad/s
  step_freq  = (10.0 × 3200) / (2π) = 5,093 Hz ← safe
  motor RPM  = (10.0 / 2π) × 60 = 95 RPM   ← well within NEMA17 torque curve

At old MAX = 3.0 rad/s:
  motor_vel  = 3.0 × 20.0 = 60.0 rad/s
  step_freq  = (60.0 × 3200) / (2π) = 30,558 Hz → capped to 20,000 Hz
  motor RPM  = 375 RPM  ← STALLS under load!
```

---

## 5. Startup Sequence & Safety

### Teensy Side
1. **Boot:** Motors disarmed (`armed = false`), all outputs zero
2. **First ROS command received:** `controlArm()` called — snaps `desired_position = actual_position`
3. **Grace period:** Next 100 commands (~1 second) are ignored to allow ROS to sync
4. **Normal operation:** Commands accepted, control loop tracks trajectory

### ROS Side (`parol6_system.cpp`)
1. **`on_configure()`:** Opens serial port, configures LibSerial (VTIME=1, VMIN=0)
2. **`on_activate()` (BLOCKING):**
   - Waits 2 seconds for USB serial stabilization
   - Drains stale serial data
   - Polls up to 10 seconds for valid `<ACK,...>` feedback
   - Sets `hw_state_positions_` and `hw_command_positions_` to real encoder values
   - **If this fails:** returns ERROR → ros2_control_node crashes
3. **`read()` (100 Hz):** Parses `<ACK,seq,p0,v0,...>` feedback, updates `hw_state_positions_`
4. **`write()` (100 Hz):** Sends `<SEQ,p0,v0,...>` commands; gated until `first_feedback_received_`

> [!CAUTION]
> The blocking `on_activate()` will fail if the wrong firmware is flashed (e.g., `encoder_debug` sends table-formatted output, not ACK packets). **Always verify the correct firmware is loaded.**

---

## 6. Serial Protocol

### Command (ROS → Teensy)
```
<SEQ,J0_pos,J0_vel,J1_pos,J1_vel,J2_pos,J2_vel,J3_pos,J3_vel,J4_pos,J4_vel,J5_pos,J5_vel>\n
```
- 13 values: 1 sequence number + 6 joints × (position + velocity)
- Positions in radians, velocities in rad/s

### Feedback (Teensy → ROS)
```
<ACK,seq,J0_pos,J0_vel,J1_pos,J1_vel,...,J5_pos,J5_vel>\n
```
- 14 tokens total (ACK + seq + 6 × 2 joint values)
- `vel` field is actually `velocity_command` (not measured velocity)

---

## 7. Current Issues & Status

### ✅ RESOLVED: Startup Violence
- **Problem:** Motors violently moving to position 0 on startup
- **Root Cause:** JointTrajectoryController latch `hw_state_positions_ = {0,...}` before encoder feedback
- **Fix:** Blocking `on_activate()` waits for real feedback + Teensy grace period

### ✅ RESOLVED: Anti-Glitch Filter Lockup
- **Problem:** Encoder glitch filter caused position jumps and lockups
- **Fix:** Filter completely removed; ISR priority fix ensures clean encoder reads

### 🔴 ACTIVE: Trajectory Overshoot (J2)
**The primary unresolved issue.** When MoveIt sends a trajectory for J2:

1. Motor starts moving in the correct direction
2. Motor overshoots the target significantly (3-6x past target)
3. Control loop tries to recover at max velocity
4. **Motor stalls** at high step frequency (exceeds NEMA17 torque curve)
5. Position error grows → runaway in wrong direction

**Evidence from debug log:**
```
Stable:    CMD pos=0.302 → Feedback pos=0.302, vel=0.000  ✅
Traj start: CMD pos=0.299, vel=-0.161                      ✅ (gentle ramp)
1s later:   Feedback pos=-0.775, vel=3.000                  ❌ OVERSHOT!
Recovery:   CMD pos=0.013, Feedback pos=-1.885, vel=3.000   ❌ WRONG DIRECTION
```

**Possible root causes (not yet confirmed):**
1. **MOTOR_DIR_SIGN[1] = -1 may be wrong** — recovery direction appears inverted
2. **Velocity integration divergence** — non-encoder joints (J2 has encoder=false) use open-loop position estimation, which doesn't reflect actual motor position
3. **Step frequency capping** — at 20kHz cap, motor stalls under load
4. **Kp too high** — was 4.0, now reduced to 1.0 (not yet tested)
5. **MAX_JOINT_VELOCITIES too high** — was 3.0, now reduced to 0.5 (not yet tested)

### Debug Logging Available
The ROS hardware interface has debug logging in `write()`:
```
📤 CMD: pos=[...] vel=[...]  ← what ROS sends (throttled ~1 Hz)
📥 Raw feedback: <ACK,...>   ← what Teensy reports
```

---

## 8. Encoder vs Non-Encoder Joints

> [!IMPORTANT]
> Currently **only J5** has its encoder enabled. All other joints use **velocity integration** (open-loop position estimation).

For non-encoder joints, position is estimated as:
```cpp
actual_position += velocity_command × dt
```
This means `actual_position` on the Teensy is just the **integral of commanded velocity** — it does NOT reflect the real physical position. If the motor stalls, loses steps, or the direction is wrong, the firmware won't know.

For encoder joints, position comes from the MT6816 PWM decode with multi-turn tracking.

---

## 9. Build & Deploy Quick Reference

### Flash Teensy Firmware
1. Open `PAROL6/firmware/realtime_servo_teensy_old/` in Arduino IDE / PlatformIO
2. Select Teensy 4.1 board
3. Compile and upload

### Build ROS Package
```bash
# Inside Docker container (or source ROS environment)
cd /workspace  # or wherever the colcon workspace root is
colcon build --packages-select parol6_hardware
source install/setup.bash
```

### Launch Real Robot
```bash
./start_real_robot.sh
# Inside the container:
ros2 launch parol6_hardware real_robot.launch.py
```

### Controller Config
The active controller config is `parol6_hardware/config/parol6_controllers.yaml`:
- Update rate: 100 Hz
- Goal tolerance: 0.2 rad per joint
- Trajectory tolerance: 5.0 rad (very permissive for debugging)
- Goal time: 60 seconds

---

## 10. Debugging Checklist

When debugging motor control issues:

1. **Check firmware** — Is `realtime_servo_teensy_old` flashed? (Not `encoder_debug` or the newer firmware)
2. **Check encoder flags** — Is `ENCODER_ENABLED[i]` correct for the joint being tested?
3. **Check gear ratios** — Does `GEAR_RATIOS[i]` match the physical gearbox?
4. **Check direction** — Does `MOTOR_DIR_SIGN[i]` match the encoder polarity?
5. **Check speeds** — At `MAX_JOINT_VELOCITIES`, what step frequency results? Avoid >10 kHz under load.
6. **Check the log** — Look for `📤 CMD:` and `📥 Raw feedback:` to trace exactly what's being sent vs received.
7. **Check startup** — Verify `✅ Got real feedback` appears in the log with correct positions.

---

## 11. Known Gotchas

| Issue | Detail |
|---|---|
| **Wrong firmware = crash** | `on_activate()` blocks waiting for `<ACK,...>` format. Debug firmwares send different formats. |
| **Grace period = 1s delay** | After arming, Teensy ignores commands for 100 ticks. Trajectory execution during this window is dropped. |
| **Non-encoder position is fake** | `actual_position` for non-encoder joints is just integrated velocity — not real position. Stalls are invisible. |
| **Docker USB timing** | On first boot, USB serial needs ~2 seconds to enumerate inside Docker. |
| **Config comments lie** | Many comments in `config.h` contradict their values (legacy from refactoring). Trust the numbers, not the comments. |
| **URDF uses different file** | `real_robot.launch.py` loads `PAROL6.urdf` for MoveIt but `parol6.urdf.xacro` for ros2_control. Both must be consistent. |
