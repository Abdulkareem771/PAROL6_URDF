# PAROL6 Communication Debugging Guide

## Problem
- Uploaded `stepdir_velocity_control.ino` to ESP32
- Launched MoveIt
- White robot model doesn't move
- Physical motors don't move

## Diagnostic Steps

### Step 1: Check Serial Port
```bash
# Inside Docker container
ls -la /dev/ttyUSB*
# or
ls -la /dev/ttyACM*
```

**Expected**: You should see `/dev/ttyUSB0` or similar

**If not found**:
- ESP32 not connected
- USB cable issue
- Driver issue

### Step 2: Test Serial Communication Manually
```bash
# Inside Docker container
screen /dev/ttyUSB0 115200
# or
minicom -D /dev/ttyUSB0 -b 115200
```

**Expected**: You should see feedback messages from ESP32:
```
<ACK,0,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000>
<ACK,1,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000>
```

**If you see nothing**: ESP32 firmware not running or wrong baud rate

**To exit screen**: Press `Ctrl+A` then `K` then `Y`

### Step 3: Check ROS Logs
Look for hardware interface errors:
```bash
# Check if hardware interface is connecting
ros2 topic echo /diagnostics
```

Look for messages like:
- ✅ "Serial opened successfully"
- ❌ "Failed to open serial port"
- ❌ "Invalid feedback: expected 14 tokens"

### Step 4: Monitor Serial Output
Open Arduino IDE Serial Monitor at 115200 baud while ROS is running.

**Expected**:
- You should see commands coming from ROS
- You should see feedback going back

**Example**:
```
Incoming: <0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00>
Sending: <ACK,0,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000>
```

---

## Common Issues

### Issue 1: Wrong Serial Port
**Symptom**: No communication at all

**Fix**: Check actual port name
```bash
dmesg | grep tty  # Shows which port ESP32 is on
```

Then launch with correct port:
```bash
ros2 launch parol6_hardware real_robot.launch.py serial_port:=/dev/ttyACM0
```

### Issue 2: Permission Denied
**Symptom**: "Permission denied" error

**Fix**: Add user to dialout group (inside Docker)
```bash
chmod 666 /dev/ttyUSB0
```

### Issue 3: Protocol Mismatch
**Symptom**: "Invalid feedback: expected 14 tokens, got 8"

**Cause**: Old firmware still running (position-only)

**Fix**: Re-upload `stepdir_velocity_control.ino`

### Issue 4: Buffer Overflow
**Symptom**: "Command buffer overflow"

**Cause**: Serial buffer too small

**Fix**: Already fixed (buffer = 512 bytes)

---

## Quick Test Commands

### Test 1: Check if ESP32 is sending data
```bash
cat /dev/ttyUSB0
```
You should see gibberish or feedback messages

### Test 2: Send test command to ESP32
```bash
echo "<0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0>" > /dev/ttyUSB0
```

Check Serial Monitor - ESP32 should respond with ACK

### Test 3: Check ROS topics
```bash
ros2 topic list | grep joint
```

Expected:
- `/joint_states`
- `/parol6_arm_controller/joint_trajectory`

### Test 4: Echo joint states
```bash
ros2 topic echo /joint_states
```

Expected: Should show position and velocity for all 6 joints

---

## What to Check Next

Please run these commands and report back:

1. **Inside Docker container**:
```bash
ls -la /dev/ttyUSB*
```

2. **Open Serial Monitor** (Arduino IDE) at 115200 baud
   - Do you see any output?
   - Does it show the 14-value feedback format?

3. **Check ROS logs** for errors:
```bash
# Look for hardware interface messages
ros2 topic echo /diagnostics | grep parol6
```

Let me know what you find!
