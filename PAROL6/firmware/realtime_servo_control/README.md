# Real-Time Servo Control Firmware - Testing Guide

## Overview
Complete rewrite of ESP32 firmware with industrial-grade architecture:
- **Position servo with velocity feedforward control**
- **Hardware timer-based step generation** (non-blocking)
- **FreeRTOS tasks** for deterministic timing
- **No `delayMicroseconds()` or blocking calls**

## Quick Start

### 1. Upload Firmware
```bash
# Open in Arduino IDE
cd /home/far2deluxe/Desktop/Servo42C_ESP/PAROL6_URDF/PAROL6/firmware/realtime_servo_control
# Open realtime_servo_control.ino
# Select: ESP32 Dev Module
# Upload
```

### 2. Restart ROS2
```bash
cd /home/far2deluxe/Desktop/Servo42C_ESP/PAROL6_URDF
./start_real_robot.sh
```

### 3. Test Trajectory
Open RViz, plan and execute a simple motion.

## What Changed

### Old Firmware Issues
❌ Velocity-only control → motor stops at waypoints  
❌ Blocking step generation → timing issues  
❌ No position correction → drifts from trajectory  

### New Firmware Architecture
✅ **Position servo** - tracks commanded position  
✅ **Velocity feedforward** - respects MoveIt velocity  
✅ **Hardware timers** - precise non-blocking steps  
✅ **FreeRTOS** - deterministic 500Hz control loop  
✅ **Servo law**: `vel_cmd = desired_vel + Kp × error`  

## Control Parameters

**Servo Gains** (in `config.h`):
- `Kp = 2.0` (initial, may need tuning)
- `Kd = 0` (no velocity derivative term)

**Timing**:
- Control loop: 500 Hz (2ms period)
- Feedback rate: 50 Hz (20ms period)

**Safety**:
- Position error limit: 0.5 rad
- Velocity clamping per joint (from `joint_limits.yaml`)

## Expected Behavior

✅ **Smooth continuous motion** - no stopping at waypoints  
✅ **Follows trajectory timing** - completes within MoveIt timeframe  
✅ **Position accuracy** - tracks commanded path  
✅ **No timeouts** - trajectory execution succeeds  

## If Issues Occur

### Motor Oscillates
**Cause**: Kp too high  
**Fix**: Reduce `Kp` in `config.h` (try 1.0, then 0.5)

### Motor Lags Behind
**Cause**: Kp too low  
**Fix**: Increase `Kp` (try 3.0, then 5.0)

### Motor Doesn't Move
**Check**:
1. Serial connection OK? (check logs)
2. Commands being received? (sequence numbers incrementing)
3. Timer frequency set? (add debug if needed)

### Timeout Still Occurs
**Check**:
1. Does position track commanded position? (compare feedback)
2. Is velocity command non-zero during motion?
3. Control loop frequency stable? (may need profiling)

## Monitoring

Watch ROS logs for:
- `📥 Raw feedback` - should show changing positions
- Sequence numbers - should increment smoothly
- Position values - should track trajectory

## Next Steps

1. **Initial test** - verify basic motion
2. **Tune Kp** - adjust for smooth tracking
3. **Stress test** - fast trajectories
4. **Long duration** - verify stability

## Files Modified

All firmware in: `/PAROL6/firmware/realtime_servo_control/`
- `config.h` - Pins, parameters, Kp gains
- `motor.cpp` - Hardware timer step generation
- `control.cpp` - Servo control loop
- `serial_comm.cpp` - UART protocol
- `realtime_servo_control.ino` - Main entry point

No ROS-side changes needed!
