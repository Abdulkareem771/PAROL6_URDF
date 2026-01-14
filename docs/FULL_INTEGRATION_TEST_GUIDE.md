# Full ROS-ESP32 Integration Test Guide

**Test the Complete Pipeline: RViz → MoveIt → ROS Driver → ESP32**

This guide shows you how to manually control the robot in RViz and watch the commands arrive at the ESP32 in real-time with formatted timestamps.

---

## 🎯 What You'll See

When you move the robot in RViz, you'll see on the ESP32 monitor:
- ✅ Exact timestamp when command arrived (microseconds + formatted time)
- ✅ All 6 joint positions
- ✅ Sequence numbers (to detect lost packets)
- ✅ Packet statistics

---

## 📋 Prerequisites

1. ✅ ESP32 connected via USB
2. ✅ `parol6_dev` container running
3. ✅ Updated firmware flashed (with enhanced logging)

---

## 🚀 Step-by-Step Test Procedure

### Step 1: Reflash Enhanced Firmware (with Better Logging)

```bash
# Exit any existing ESP32 monitors first (Ctrl + ])

# Inside Docker container
docker exec -it parol6_dev bash
cd /workspace/esp32_benchmark_idf
. /opt/esp-idf/export.sh
idf.py build
idf.py -p /dev/ttyUSB0 flash
```

**Wait for**: `Hash of data verified.`

---

### Step 2: Start ESP32 Monitor (Terminal 1)

```bash
# Still inside container
idf.py -p /dev/ttyUSB0 monitor
```

**Expected output:**
```
========================================
  ESP32 Benchmark Firmware (ESP-IDF)
========================================
Ready to receive commands...
READY: ESP32_BENCHMARK_V2
```

**Leave this terminal open** - you'll watch commands here!

---

### Step 3: Launch ROS with Real Robot Mode (Terminal 2)

```bash
# New terminal - enter container
docker exec -it parol6_dev bash
cd /workspace

# Build (will skip esp32_benchmark due to COLCON_IGNORE)
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Launch real robot mode
ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=real
```

**Wait for**: RViz window opens

---

### Step 4: Move Robot in RViz

In RViz:
1. **Drag the interactive marker** (end-effector sphere)
2. **Click "Plan"** button
3. **Click "Execute"** button

---

### Step 5: Watch ESP32 Monitor (Terminal 1)

You'll see output like this for **each waypoint**:

```
<ACK,0,1234567>
════════════════════════════════════════════════════════════════
  COMMAND RECEIVED [SEQ: 0]
════════════════════════════════════════════════════════════════
  Timestamp:  1.234 seconds (1.234567 s)
  Raw µs:     1234567
────────────────────────────────────────────────────────────────
  Joint Positions (radians):
    J1: +0.5234  |  J2: -0.8921  |  J3: +1.2345
    J4: -0.4567  |  J5: +0.7890  |  J6: -0.3210
────────────────────────────────────────────────────────────────
  Statistics:
    Total Received: 1  |  Packets Lost: 0
════════════════════════════════════════════════════════════════

<ACK,1,1334567>
════════════════════════════════════════════════════════════════
  COMMAND RECEIVED [SEQ: 1]
════════════════════════════════════════════════════════════════
  Timestamp:  1.334 seconds (1.334567 s)
  Raw µs:     1334567
...
```

**Each command shows:**
- ✅ Sequence number increments
- ✅ Timestamp increases (~50-100ms between commands)
- ✅ Joint values change smoothly
- ✅ No packet loss

---

## 📊 What to Analyze

### 1. **Command Rate**
Look at timestamp differences:
```
Command 0: 1.234 seconds
Command 1: 1.334 seconds  → 100ms gap
Command 2: 1.434 seconds  → 100ms gap
```

**Good**: Regular intervals (~50-100ms)  
**Bad**: Irregular gaps or > 200ms

### 2. **Packet Loss**
Watch the statistics line:
```
Total Received: 100  |  Packets Lost: 0
```

**Good**: 0 packets lost  
**Bad**: Any lost packets

### 3. **Sequence Numbers**
```
SEQ: 0
SEQ: 1
SEQ: 2
...
```

**Good**: Continuous counting  
**Bad**: Jumps (e.g., 0, 1, 5 ← lost 2, 3, 4)

---

## 🔬 Advanced: Measure End-to-End Latency

### Method 1: Visual Timing

1. **In RViz**: Note the time when you click "Execute"
2. **In ESP32 Monitor**: Note timestamp of first command
3. **Calculate**: Latency = (ESP32 time) - (RViz click time)

**Expected**: < 100ms

### Method 2: Use PC Logs

```bash
# Check driver logs (created automatically when mode:=real)
ls -lh /workspace/logs/driver_commands_*.csv

# View recent commands
tail -20 /workspace/logs/driver_commands_*.csv
```

Compare PC log timestamps with ESP32 monitor timestamps.

---

## 📸 Screenshot & Evidence Collection

### Take Screenshots of:

1. **RViz Window** - Showing planned trajectory
2. **ESP32 Monitor** - Showing formatted command output
3. **Together** - Both windows visible (for thesis documentation)

### Save Logs:

```bash
# PC side logs
cp /workspace/logs/driver_commands_*.csv ~/thesis_evidence/

# ESP32 logs (if using SD card)
# or copy from monitor output
```

---

## ✅ Success Criteria

**Your test is successful if:**
- ✅ Every RViz "Execute" creates ESP32 output
- ✅ Sequence numbers are continuous (no gaps)
- ✅ Packet loss = 0
- ✅ Timestamps show regular intervals
- ✅ Joint values match RViz trajectory

---

## ⚠️ Troubleshooting

### Issue: No ESP32 Output When Executing

**Check:**
```bash
# Is driver finding the ESP32?
# Look for this in ROS terminal:
# [real_robot_driver]: Connected to Microcontroller at /dev/ttyUSB0
```

**Fix**: Restart ROS launch

---

### Issue: Timestamps Don't Increment

**Cause**: ESP32 crashed or reset

**Fix**: Press RESET on ESP32, reflash if needed

---

### Issue: "esp32_benchmark" Build Error

**Cause**: COLCON_IGNORE not recognized  

**Fix**:
```bash
# Verify file exists
ls /workspace/esp32_benchmark_idf/COLCON_IGNORE

#If missing:
touch /workspace/esp32_benchmark_idf/COLCON_IGNORE

# Clean and rebuild
cd /workspace
rm -rf build install log
colcon build --symlink-install
```

---

## 📊 Expected Performance

| Metric | Target | Your Result |
|--------|--------|-------------|
| Packet Loss | 0% | ___ |
| Command Rate | 10-20 Hz | ___ Hz |
| Latency | < 100ms | ___ ms |
| Jitter | < 10ms | ___ ms |

---

## 🎓 For Thesis Documentation

**What to Include:**

1. **Screenshot**: RViz + ESP32 monitor side-by-side
2. **Table**: Performance metrics (from table above)
3. **Log Sample**: Copy 5-10 formatted ESP32 messages
4. **Graph**: Plot timestamps from `driver_commands_*.csv`

**Caption Example:**
> "Real-time verification of MoveIt trajectory execution. Commands generated by the motion planner (left, RViz) arrive at the ESP32 microcontroller (right) with timestamps showing < 5ms jitter and zero packet loss over 100+ waypoints."

---

**Last Updated**: January 2026  
**Maintained by**: PAROL6 Team
