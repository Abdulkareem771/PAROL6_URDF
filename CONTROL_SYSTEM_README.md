# PAROL6 Robot Arm - Control System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Hardware Configuration](#hardware-configuration)
3. [Software Architecture](#software-architecture)
4. [Current Implementation](#current-implementation)
5. [Known Issues & Limitations](#known-issues--limitations)
6. [Attempted Solutions](#attempted-solutions)
7. [Recommended Next Steps](#recommended-next-steps)

---

## System Overview

This document describes the control system for the PAROL6 6-DOF robot arm using MKS SERVO42C closed-loop stepper motors controlled via ESP32 microcontroller and ROS 2 Control.

### Control Philosophy

**Hybrid Step/Dir + Velocity Control**:
- ROS 2 sends position AND velocity commands at 100Hz
- ESP32 generates step/dir pulses matching commanded velocity
- MKS SERVO42C handles closed-loop position control internally
- This approach gives ROS full control over velocity profiles while leveraging MKS's closed-loop capabilities

---

## Hardware Configuration

### Motors: MKS SERVO42C

**Specifications**:
- Closed-loop stepper motors with integrated encoder
- 200 steps/revolution (1.8° per full step)
- Step/Dir interface for position control
- UART interface for configuration and diagnostics (NOT used for real-time control)
- Internal 48MHz controller running FOC (Field-Oriented Control)

**UART Limitations**:
- Encoder readings via UART have high packet loss
- 48MHz controller is busy with FOC, cannot provide reliable real-time feedback
- UART only used for verification, NOT closed-loop control

### Microcontroller: ESP32

**Current Setup**:
- Dual-core Xtensa LX6 @ 240MHz
- 3 hardware UART channels
- UART to PC: 115200 baud
- Generates step/dir pulses for 6 motors
- Runs at 100Hz control loop

**Limitations**:
- Software-based step generation (no hardware timers used)
- Max reliable step rate: ~20kHz
- UART bandwidth limited at 115200 baud
- WiFi/Bluetooth can cause timing jitter (disabled in firmware)

### Joint Configuration

| Joint | Microsteps | Gearbox | Steps/Joint Rev | Step Size (°) | Step Size (rad) |
|-------|------------|---------|-----------------|---------------|-----------------|
| J1 | 16 | 1:1 | 3,200 | 0.1125 | 0.00196 |
| J2 | 4 | 20:1 | 16,000 | 0.0225 | 0.000393 |
| J3 | 16 | 1:1 | 3,200 | 0.1125 | 0.00196 |
| J4 | 16 | 1:1 | 3,200 | 0.1125 | 0.00196 |
| J5 | 16 | 1:1 | 3,200 | 0.1125 | 0.00196 |
| J6 | 16 | 1:1 | 3,200 | 0.1125 | 0.00196 |

**Key Insight**: J2 has 5× finer step resolution due to 20:1 gearbox, but requires 20× more steps to move the same joint angle.

---

## Software Architecture

### ROS 2 Control Stack

```
MoveIt 2 (Motion Planning)
    ↓
Joint Trajectory Controller (100Hz)
    ↓
Hardware Interface (parol6_system.cpp)
    ↓
UART (115200 baud)
    ↓
ESP32 Firmware (stepdir_velocity_control.ino)
    ↓
Step/Dir Signals → MKS SERVO42C Motors
```

### Communication Protocol

**TX (ROS → ESP32)**: 13 values per packet
```
<SEQ,J1_pos,J1_vel,J2_pos,J2_vel,J3_pos,J3_vel,J4_pos,J4_vel,J5_pos,J5_vel,J6_pos,J6_vel>
```

**RX (ESP32 → ROS)**: 14 values per packet
```
<ACK,SEQ,J1_pos,J1_vel,J2_pos,J2_vel,J3_pos,J3_vel,J4_pos,J4_vel,J5_pos,J5_vel,J6_pos,J6_vel>
```

**Packet Size**: ~78 bytes TX, ~84 bytes RX  
**Frequency**: 100Hz  
**Bandwidth**: ~16,200 bytes/second (~130,000 baud required)

---

## Current Implementation

### Firmware: `stepdir_velocity_control.ino`

**Key Features**:
1. **Per-Motor Microstepping**: Different microstep settings per joint
2. **Per-Motor Dead Zones**: Different position tolerances based on step resolution
3. **Velocity Smoothing**: Exponential smoothing (alpha=0.2) to prevent hard snaps
4. **Non-Blocking Step Generation**: Up to 200 steps per cycle to prevent loop blocking

**Configuration**:
```cpp
// Microstepping (steps/rev = 200 × MICROSTEPS)
const int MICROSTEPS[6] = {16, 4, 16, 16, 16, 16};

// Gearbox ratios
const float GEAR_RATIOS[6] = {1.0, 20.0, 1.0, 1.0, 1.0, 1.0};

// Dead zones (when to stop trying to reach target)
const float DEAD_ZONES[6] = {0.002, 0.001, 0.002, 0.002, 0.002, 0.002};

// Velocity smoothing (0.2 = 20% new, 80% old per cycle)
float alpha = 0.2;
```

### ROS Configuration

**Controller Update Rate**: 100Hz  
**Command Interfaces**: Position + Velocity  
**State Interfaces**: Position + Velocity

**Goal Tolerances** (`parol6_controllers.yaml`):
```yaml
joint_L1: {trajectory: 2.0, goal: 0.002}   # 0.11° - limited by step resolution
joint_L2: {trajectory: 2.0, goal: 0.001}   # 0.057° - finer due to gearbox
joint_L3: {trajectory: 2.0, goal: 0.002}   # 0.11°
joint_L4: {trajectory: 2.0, goal: 0.002}   # 0.11°
joint_L5: {trajectory: 2.0, goal: 0.002}   # 0.11°
joint_L6: {trajectory: 2.0, goal: 0.002}   # 0.11°
```

**Velocity Limits** (`joint_limits.yaml`):
```yaml
joint_L1: {max_velocity: 1.5, max_acceleration: 2.0}
joint_L2: {max_velocity: 1.5, max_acceleration: 2.0}  # Limited for 20:1 gearbox
joint_L3-L6: {max_velocity: 6.0, max_acceleration: 4.0}
```

---

## Known Issues & Limitations

### 1. Position Tracking Snap (Primary Issue)

**Problem**: Motor "snaps" hard at the start of each new trajectory, especially J2 (~80° snap at motor shaft).

**Root Cause**:
- Firmware tracks position by counting steps sent (open-loop)
- Motor may not reach exact target due to:
  - Goal tolerance (motor stops when "close enough")
  - Step resolution limits (can't reach finer than 1 step)
  - MKS internal positioning
- Next trajectory starts from "assumed" position
- Motor snaps to catch up to where firmware thinks it is

**Example**:
```
Target: -1.000 rad
Motor reaches: -0.997 rad (within 0.003 rad tolerance, ROS says "done")
Firmware believes: -1.000 rad (based on steps sent)

Next target: -1.500 rad
ROS trajectory: -1.000 → -1.500 rad
Motor reality: -0.997 → snaps to -1.000 → moves to -1.500
```

### 2. J2 Timeout Issues

**Problem**: J2 (geared joint) frequently times out before reaching target.

**Root Cause**:
- J2 needs 16,000 steps/joint revolution (vs 3,200 for others)
- At 1.5 rad/s: requires 3,820 steps/second
- At 100Hz update: 38 steps per cycle
- Tight tolerance (0.001 rad) is achievable but takes longer
- Trajectory times out before motor completes movement

**Observed Behavior**:
- Motor reaches close to target (e.g., -0.971 rad)
- Velocity drops to 0.000 (motor stops moving)
- Trajectory times out
- Next execution: position jumps (snap)

### 3. Step Resolution vs Tolerance Mismatch

**Problem**: Goal tolerance tighter than achievable step resolution.

**J1 Example** (16 microsteps, direct drive):
- Step size: 0.1125° = 0.00196 rad
- Goal tolerance: 0.002 rad
- Motor can get within 1 step, but that's ~0.002 rad
- Sometimes reaches target, sometimes times out

**J2 Example** (4 microsteps, 20:1 gearbox):
- Step size at joint: 0.0225° = 0.000393 rad
- Goal tolerance: 0.001 rad
- Can achieve tolerance, but takes many steps
- Often times out before completion

### 4. UART Bandwidth Limitation

**Current**: 115200 baud, 100Hz
- Theoretical capacity: 11,520 bytes/second
- Actual usage: ~16,200 bytes/second
- **Result**: Occasional packet loss (seen in logs)

**At 500Hz** (desired):
- Required: ~81,000 bytes/second
- Minimum baud: 921,600
- ESP32 Arduino has reliability issues at high baud rates

### 5. No Real-Time Encoder Feedback

**Limitation**: Cannot use MKS encoder for closed-loop control
- MKS UART has high packet loss
- 48MHz controller busy with FOC
- Encoder readings unreliable for real-time use
- Only suitable for post-movement verification

**Impact**: Position drift accumulates over time with no correction

---

## Attempted Solutions

### Attempt 1: Tighten Goal Tolerance to Match Firmware Dead Zone

**Approach**: Set goal tolerance to 0.0001 rad to match firmware dead zone  
**Result**: ❌ **Failed** - J1 timed out (tolerance unreachable with 0.1125° steps)

### Attempt 2: Uniform Realistic Tolerance (0.002 rad)

**Approach**: Set all joints to 0.002 rad (achievable with 16 microsteps)  
**Result**: ⚠️ **Partial** - J1 works, but J2 still snaps (can achieve tighter tolerance)

### Attempt 3: Per-Joint Tolerances Based on Step Resolution

**Approach**: 
- J1, J3-J6: 0.002 rad (step resolution limit)
- J2: 0.0005 rad (tighter due to gearbox)

**Result**: ❌ **Failed** - J2 timed out (too tight, takes too long)

### Attempt 4: Balanced Per-Joint Tolerances

**Approach**:
- J1, J3-J6: 0.002 rad
- J2: 0.001 rad (balanced)

**Result**: ⚠️ **Current State** - J2 still times out frequently, ~80° snap persists

### Attempt 5: Velocity Smoothing Adjustment

**Approach**: Reduce alpha from 0.5 → 0.2 → 0.05 to prevent hard snaps  
**Result**: ✅ **Helped** - Reduced initial snap, but doesn't fix position tracking snap

### Attempt 6: Per-Motor Microstepping

**Approach**: J2 uses 4 microsteps (faster), others use 16 (precision)  
**Result**: ✅ **Helped** - J2 moves faster, but still times out due to 100Hz limit

---

## Recommended Next Steps

### Option A: Upgrade to Teensy 4.1 + 500Hz (Recommended)

**Hardware**: Teensy 4.1 (~$30)  
**Effort**: 1-2 hours firmware port  
**Benefits**:
- ✅ Can handle 500Hz at 921,600+ baud reliably
- ✅ Hardware timers for precise step generation
- ✅ No WiFi/Bluetooth interference
- ✅ 5× more position updates → less position error
- ✅ Should solve timeout issues

**Limitations**:
- ⚠️ Still open-loop (no encoder feedback)
- ⚠️ Small snap may persist (< 5° instead of 80°)

**Implementation**:
1. Port firmware to Teensy 4.1
2. Implement hardware timer step generation
3. Increase baud to 921,600 or 2 Mbps
4. Increase ROS update rate to 500Hz
5. Test and tune tolerances

### Option B: Hybrid Control (Quick Fix)

**Approach**: 
- J2: Position-only control (let MKS handle acceleration)
- J1, J3-J6: Velocity control

**Effort**: 1-2 hours  
**Benefits**:
- ✅ No firmware changes needed
- ✅ J2 snap eliminated (MKS handles smoothing)
- ✅ Other joints keep velocity control

**Limitations**:
- ⚠️ Less control over J2 velocity profile
- ⚠️ Mixed control strategies

### Option C: Accept Looser Tolerance (Temporary)

**Approach**: Set J2 tolerance to 0.003 rad (original)  
**Effort**: 2 minutes  
**Benefits**:
- ✅ Trajectories complete reliably
- ✅ No timeouts

**Limitations**:
- ❌ Snap persists (~1.2° at joint, ~24° at motor)
- ❌ Not a real solution

### Option D: External Encoders (Ultimate Solution)

**Approach**: Add external absolute encoders to each joint  
**Effort**: 1-2 weeks (hardware + firmware + ROS integration)  
**Benefits**:
- ✅ True closed-loop control
- ✅ No position drift
- ✅ No snap
- ✅ Perfect positioning

**Limitations**:
- ❌ Expensive (~$50-100 per encoder × 6)
- ❌ Significant mechanical work
- ❌ Complex firmware rewrite

---

## Technical Specifications Summary

### Current System Capabilities

| Parameter | Value | Limit |
|-----------|-------|-------|
| **ROS Update Rate** | 100Hz | ESP32 UART bandwidth |
| **UART Baud Rate** | 115200 | Reliable at this rate |
| **Max Step Rate** | ~20kHz | Software timing limit |
| **Position Accuracy** | ±0.1° (J1) | Step resolution |
| **Position Accuracy** | ±0.03° (J2) | Gearbox × step resolution |
| **Packet Loss** | Occasional | UART bandwidth limit |

### Theoretical Limits with Teensy 4.1

| Parameter | Value | Improvement |
|-----------|-------|-------------|
| **ROS Update Rate** | 500Hz+ | 5× faster |
| **UART Baud Rate** | 2 Mbps | 17× faster |
| **Max Step Rate** | 100kHz+ | 5× faster |
| **Position Accuracy** | Same | Limited by step resolution |
| **Packet Loss** | Minimal | Better UART handling |

---

## File Locations

### Firmware
- **Main**: `/PAROL6/firmware/stepdir_velocity_control/stepdir_velocity_control.ino`

### ROS Configuration
- **Controllers**: `/parol6_hardware/config/parol6_controllers.yaml`
- **Joint Limits**: `/parol6_moveit_config/config/joint_limits.yaml`
- **URDF**: `/parol6_hardware/urdf/parol6.ros2_control.xacro`

### Hardware Interface
- **Header**: `/parol6_hardware/include/parol6_hardware/parol6_system.hpp`
- **Implementation**: `/parol6_hardware/src/parol6_system.cpp`

---

## Conclusion

The current system works but has inherent limitations due to:
1. **100Hz update rate** - too slow for geared joints
2. **Open-loop position tracking** - no encoder feedback
3. **ESP32 UART bandwidth** - limits update rate increase

**Best path forward**: Upgrade to Teensy 4.1 with 500Hz update rate and hardware timer step generation. This addresses the immediate timeout and snap issues while staying within the constraint of no real-time encoder feedback.

For ultimate performance, external absolute encoders would be required, but this is a significant undertaking.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-07  
**Author**: System Documentation
