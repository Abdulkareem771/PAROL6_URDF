# ROS Hardware Interface Testing Guide

**Objective**: Verify the `parol6_hardware` ros2_control plugin correctly reads encoder state and writes motor commands through the Teensy firmware.

---

## Overview

The hardware interface (`parol6_system.cpp`) does three things:
1. **Read** — parses `<ACK,...>` feedback packets from Teensy into `joint_states`
2. **Write** — encodes `<SEQ,pos,vel>` command packets to Teensy at 25 Hz
3. **State** — reports hardware state to ros2_control (active / error)

This guide tests each layer independently.

---

## Prerequisites

- Firmware flashed and serial communication test passed
- Docker container running: `docker exec -it parol6_dev bash`
- Built workspace: `cd /workspace && source install/setup.bash`

---

## Test 1 — Read path: encoder → joint_states

### 1a. Launch hardware interface only (no MoveIt)

```bash
# Inside Docker
ros2 launch parol6_hardware real_robot.launch.py \
  serial_port:=/dev/ttyACM0 \
  baud_rate:=115200 \
  allow_spoofing:=false
```

In a second terminal:
```bash
# Watch joint_states with timestamps
ros2 topic echo /joint_states
```

**Expected:**
```yaml
header:
  stamp:
    sec: 1741653600
name: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6]
position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # ← encoder readings
velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### 1b. Verify encoder reads change with physical motion

Manually rotate J1's output shaft slightly by hand (robot unpowered is fine).  
The `position[0]` in `/joint_states` should change in real time.

**Sign convention**: turning in the positive direction (as defined by URDF) should give **positive** position change. If it's negative, `DIR_INVERT[0]` needs toggling.

### 1c. Confirm publish rate

```bash
ros2 topic hz /joint_states
# Expected: 25 Hz ± 2 Hz
```

---

## Test 2 — Write path: controller → Teensy command

### 2a. Check command is reaching Teensy

Enable the arm controller and send a zero-velocity hold command:
```bash
# After launching real_robot.launch.py:
ros2 control switch_controllers \
  --activate parol6_arm_controller \
  --deactivate ""
```

Open the Firmware Configurator serial tab and observe — you should see `<ACK,...>` packets arriving continuously, meaning the hardware interface is writing commands and Teensy is responding.

### 2b. Verify sign correction applied correctly

The hardware interface applies sign correction to J1, J3, J6 (configurable via `DIR_INVERT`).

```bash
# Check what command the controller is sending vs what ROS reports
ros2 topic echo /parol6_arm_controller/state --field reference.positions
# Compare with:
ros2 topic echo /joint_states --field position
# They should match (encoder tracks command) for a stationary robot
```

---

## Test 3 — allow_spoofing=false enforcement

With `allow_spoofing:=false` (the default for real hardware), the hardware interface will return an `ERROR` lifecycle state if:

- The serial port is not present
- The received packet cannot be parsed
- Position values are out of bounds

### 3a. Test missing serial port

```bash
ros2 launch parol6_hardware real_robot.launch.py \
  serial_port:=/dev/nonexistent \
  baud_rate:=115200 \
  allow_spoofing:=false
```

Expected: hardware interface logs an error and `list_controllers` shows the controller as **unconfigured** or **inactive** (not active).

### 3b. Verify normal operation restores

Reconnect Teensy → relaunch → controllers go **active**.

---

## Test 4 — Controller response bandwidth

Measure how quickly the arm controller error converges after a step command:

```bash
# With robot stationary at 0, command 0.2 rad on J1
ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  '{
    joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6],
    points: [{
      positions: [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
      velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      time_from_start: {sec: 3, nanosec: 0}
    }]
  }'

# In parallel, log the joint position
ros2 topic echo /joint_states --field position[0]
```

**Expected**: Position rises smoothly from 0 to ~0.2 rad within 3 seconds, no oscillation.

| Behaviour | Cause | Fix |
|-----------|-------|-----|
| Overshoot + ringing | KP too high | Lower `KP_GAINS[0]` |
| Slow convergence, doesn't reach goal | KP too low | Raise `KP_GAINS[0]` |
| Steady-state error stays > 0.05 rad | KI needed | Add small KI (0.01) and MAX_INTEGRAL |
| Motor stalls, position stuck | Current limit on driver too low | Increase driver current |

---

## Success criteria

| Test | Pass condition |
|------|---------------|
| 1. Encoder → joint_states | Publishes at 25 Hz, sign correct, values change with shaft rotation |
| 2. Write path | `<ACK,...>` visible in serial tab during active control |
| 3. allow_spoofing enforcement | Missing port → ERROR state, not crash |
| 4. Step response | 0.2 rad reached within 3 s, no oscillation > 0.02 rad |

---

## Common issues

| Symptom | Fix |
|---------|-----|
| Hardware interface in `inactive` immediately | `allow_spoofing:=false` + no Teensy → expected. Connect Teensy. |
| Position stuck at 0.0 always | Encoder cable not connected or wrong pin in config |
| Position jumps by large random amounts | Encoder signal interference — use shielded cable, add 100 nF cap at pin |
| `joint_states` at 10 Hz not 25 | Reflash with `FEEDBACK_RATE_HZ=25` |
| Arm controller `inactive` after spoofing error | Send `<ENABLE>` in serial tab, then restart controller via `ros2 control switch_controllers` |

---

**Last Updated**: 2026-03-11  
**Maintained by**: PAROL6 Team
