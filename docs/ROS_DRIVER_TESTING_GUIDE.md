# ROS Driver Communication Testing Guide

**Objective**: Test the complete pipeline: **RViz → MoveIt → ROS Driver → ESP32**

This verifies:
- ✅ MoveIt plans execute correctly
- ✅ ROS driver formats commands properly
- ✅ Serial communication is reliable
- ✅ ESP32 receives accurate data
- ✅ End-to-end latency is acceptable

---

## 🔧 Setup

### 1. Upload ESP32 Firmware

Upload `PAROL6/firmware/benchmark_firmware.ino` to your ESP32 (same as before).

### 2. Update ROS Driver (Already Done!)

The driver now logs all commands to:
```
/workspace/logs/driver_commands_YYYYMMDD_HHMMSS.csv
```

### 3. Connect ESP32

```bash
# Find port
ls /dev/ttyUSB* /dev/ttyACM*

# Give permission if needed
sudo chmod 666 /dev/ttyUSB0
```

---

## 🚀 Running the Test

### Step 1: Start ROS System

```bash
# Inside Docker
docker exec -it parol6_dev bash
cd /workspace

# Launch robot (enables logging by default)
ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=real
```

**What happens:**
- RViz opens
- Driver connects to ESP32 at `/dev/ttyUSB0` (or `/dev/ttyACM0`)
- Logging starts automatically → `/workspace/logs/driver_commands_*.csv`

### Step 2: Plan and Execute in RViz

1. **Drag end-effector** to desired position
2. Click **"Plan"** → MoveIt generates trajectory
3. Click **"Execute"** → Sends to real_robot_driver → ESP32

**Watch terminals:**
- **ROS terminal**: Shows "Executing Trajectory..."
- **ESP32 Serial Monitor** (if open): Shows `LOG,timestamp,SEQ:X,J:[...]`

### Step 3: Stop and Collect Logs

```bash
# Stop ROS (Ctrl+C)
^C

# Logs are in:
ls /workspace/logs/

# You should see:
# driver_commands_20260112_194500.csv  (PC side)
```

### Step 4: Get ESP32 Log

**Option A: Serial Monitor** (if you had it open)
- Copy all `LOG,` lines to a file
- Save as `esp32_log.csv`

**Option B: SD Card** (if USE_SD_LOGGING=true)
- Remove SD card from ESP32
- Copy `comm_test.log` to PC

**Option C: Request Stats**
If firmware supports it, you can query stats via serial command.

---

## 📊 Analyze Results

### Run Analysis Script

```bash
# Inside Docker or on host (needs matplotlib)
pip3 install pandas matplotlib numpy

# Run analysis
python3 scripts/analyze_communication_logs.py \
  --pc-log /workspace/logs/driver_commands_20260112_194500.csv \
  --esp-log /path/to/esp32_log.csv \
  --output ros_esp_analysis.png
```

**Output:**
```
==================================================
PACKET LOSS ANALYSIS
==================================================
Commands Sent (PC):     245
Commands Received (ESP): 245
Packets Lost:            0
Loss Rate:               0.00%

==================================================
LATENCY ANALYSIS
==================================================
Valid Samples:    245
Mean Latency:     4.56 ms
Median Latency:   4.23 ms
Min Latency:      3.12 ms
Max Latency:      8.91 ms
Std Deviation:    1.23 ms

✓ Report saved: ros_esp_analysis.png
```

---

## 📈 What the Report Shows

### Graph 1: Packet Loss Pie Chart
- **Target**: 100% received, 0% lost
- **Warning**: If any loss, check USB cable/connection

### Graph 2: Latency Over Time
- Shows latency for each command
- **Good**: Flat line ~5ms
- **Bad**: Spikes > 50ms (indicates system lag)

### Graph 3: Latency Distribution
- Histogram showing jitter
- **Good**: Tight peak (consistent timing)
- **Bad**: Wide spread (unpredictable delays)

### Graph 4: Command Rate
- How fast MoveIt sends commands
- **Typical**: 10-20 Hz (every 50-100ms)
- Shows if trajectory timing is smooth

### Graph 5: Statistics Summary
- Text summary of all metrics

---

## ✅ Success Criteria

| Metric | Good | Acceptable | Problem |
|--------|------|------------|---------|
| Packet Loss | 0% | < 0.1% | > 1% |
| Mean Latency | < 5ms | < 15ms | > 50ms |
| Max Latency | < 10ms | < 30ms | > 100ms |
| Jitter (Std Dev) | < 3ms | < 10ms | > 20ms |

**Interpretation:**
- **< 5ms latency**: Excellent! Motors will track smoothly
- **< 15ms latency**: Good enough for most applications
- **> 50ms latency**: Problem! Commands lag behind MoveIt plan

---

## 🔍 Troubleshooting

### Issue: High Latency (> 50ms)

**Possible Causes:**
1. **Other ROS nodes consuming CPU** → Close unnecessary nodes
2. **Serial Monitor open** → Close Arduino IDE
3. **Docker resource limits** → Increase Docker CPU allocation
4. **USB hub lag** → Connect ESP32 directly to laptop

**Fix:**
```bash
# Check ROS node performance
ros2 topic hz /joint_states
# Should be near 20 Hz

# Check Docker resources
docker stats parol6_dev
```

---

### Issue: Packet Loss (> 1%)

**Causes:**
1. Bad USB cable
2. Serial buffer overflow (commands too fast)
3. ESP32 crashed/rebooted

**Fix:**
- Replace USB cable
- Check ESP32 Serial Monitor for error messages
- Add delay in driver (increase `time.sleep(0.05)` to `0.1`)

---

### Issue: Data Mismatch

If joint values don't match between PC and ESP32:

**Causes:**
1. Parsing error in ESP32 firmware
2. Serial corruption
3. Floating point precision loss

**Fix:**
- Check ESP32 `parseData()` function
- Verify format: `<seq,j1,j2,j3,j4,j5,j6>`
- Use fixed precision (4 decimal places)

---

## 📝 Example Test Session

```bash
# 1. Start system
ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=real

# 2. In RViz: Plan and Execute 5 different trajectories
#    - Home → Up
#    - Up → Left
#    - Left → Right
#    - Right → Forward
#    - Forward → Home

# 3. Stop ROS
^C

# 4. Check logs
ls -lh /workspace/logs/
# driver_commands_20260112_194500.csv  (1.2 MB, 245 commands)

# 5. Copy ESP32 log (from Serial Monitor or SD)
# Save as: /workspace/logs/esp32_log.csv

# 6. Analyze
python3 scripts/analyze_communication_logs.py \
  --pc-log /workspace/logs/driver_commands_20260112_194500.csv \
  --esp-log /workspace/logs/esp32_log.csv

# 7. Review ros_esp_analysis.png
# Expected: 0% loss, ~5ms latency
```

---

## 🎯 Next Steps

Once verification passes:
1. ✅ **Update firmware** to real motor control code
2. ✅ **Connect motors** (one at a time!)
3. ✅ **Test with low speed** first
4. ✅ **Gradually increase** trajectory speed
5. ✅ **Full welding application** integration

---

## 📞 Expected Results (Benchmark)

**System**: ESP32 + USB 2.0 + ROS 2 Humble

| Component | Latency Contribution |
|-----------|---------------------|
| MoveIt Planning | ~10-50ms (one-time) |
| ROS message passing | ~1-2ms |
| Serial transmission | ~2-3ms |
| ESP32 processing | ~1ms |
| **Total (per waypoint)** | **~5-10ms** |

**Command Rate**: 
- MoveIt typically sends waypoints at 10-20 Hz
- Driver can handle up to 100 Hz if needed

---

**Last Updated**: January 2026  
**Maintained by**: PAROL6 Team
