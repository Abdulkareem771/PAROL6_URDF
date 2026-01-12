# Full ROS Pipeline Test - Hardware Preparation

**Test: RViz → MoveIt → Driver → Real ESP32**

This workflow lets you:
- ✅ See **exactly** what your motors will receive (positions, velocities, accelerations)
- ✅ Measure **end-to-end latency** from MoveIt to ESP32
- ✅ Verify **message format** for motor control firmware
- ✅ Test with **REAL ESP32** (not virtual/simulation)

---

## 🎯 Setup (One-Time)

### 1. Build Updated Driver

The driver was fixed to send correct format with sequence numbers.

```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select parol6_driver
exit
```

---

## 🚀 Testing Procedure

### Terminal 1: ESP32 Monitor

```bash
docker exec -it parol6_dev bash
cd /workspace/esp32_benchmark_idf
. /opt/esp-idf/export.sh
idf.py -p /dev/ttyUSB0 monitor
```

**Expected:**
```
READY: ESP32_BENCHMARK_V2
```

**Leave this running** - you'll watch MoveIt commands arrive here!

---

### Terminal 2: Launch Real Robot Mode

```bash
./start_real_robot.sh
```

**What happens:**
1. Builds workspace
2. Launches MoveIt + RViz
3. Starts `real_robot_driver` (connects to ESP32)
4. Ready to plan and execute!

**Expected in Terminal 2:**
```
[real_robot_driver]: Connected to Microcontroller at /dev/ttyUSB0
[move_group]: You can start planning now!
```

RViz window should open with robot visible.

---

### Terminal 3: Move Robot in RViz

1. **Drag** the interactive marker (sphere at end-effector)
2. Click **"Plan"** → MoveIt calculates trajectory
3. Click **"Execute"** → Sends to driver → ESP32

---

### Terminal 1: Watch ESP32 Output

You'll see commands like this:

```
<ACK,0,1234567>
I (1234) BENCHMARK: SEQ:0 J:[0.5234,-0.8921,1.2345,-0.4567,0.7890,-0.3210]
<ACK,1,1334567>
I (1334) BENCHMARK: SEQ:1 J:[0.5244,-0.8925,1.2350,-0.4570,0.7895,-0.3215]
<ACK,2,1434567>
I (1434) BENCHMARK: SEQ:2 J:[0.5254,-0.8929,1.2355,-0.4573,0.7900,-0.3220]
...
```

**Message Format Breakdown:**

Each line from ESP32 has two parts:

**1. ACK Response (sent to PC):**
```
<ACK,130,9916752>
```
- `ACK` - Acknowledgment that command was received
- `130` - Sequence number (matches the command)
- `9916752` - ESP32 timestamp in microseconds (time since power-on)

**2. Debug Info (for monitoring):**
```
I (10218) BENCHMARK: SEQ:130 J:[0.028,0.001,-0.236,0.929,1.634,1.886]
```
- `I` - Info level log
- `(10218)` - Time in milliseconds since boot
- `BENCHMARK` - Tag (firmware component name)
- `SEQ:130` - Sequence number (confirms order)
- `J:[...]` - Joint positions for all 6 joints in **radians**
  - J1: Base rotation (0.028 rad ≈ 1.6°)
  - J2: Shoulder pitch (0.001 rad ≈ 0.06°)
  - J3: Elbow pitch (-0.236 rad ≈ -13.5°)
  - J4: Wrist pitch (0.929 rad ≈ 53.2°)
  - J5: Wrist roll (1.634 rad ≈ 93.6°)
  - J6: End-effector twist (1.886 rad ≈ 108°)

**Timing Analysis:**
```
SEQ:130 → timestamp 9916752 µs (9.916 seconds)
SEQ:131 → timestamp 9966756 µs (9.966 seconds)  
Difference: 50ms between commands
```

Typical rate: **10-20 Hz** (one command every 50-100ms)

**What This Means for Motors:**
- Commands arrive **every 50-100ms**
- You have 50-100ms to move motors to new position
- Positions are **absolute** (not incremental)
- Sequence numbers let you detect lost packets

---

## 📊 Check Logs (PC Side)

After executing, check what the driver sent:

```bash
# Stop ROS (Ctrl+C in Terminal 2)

# View logs
ls /workspace/logs/

# You'll see:
# driver_commands_20260113_003000.csv
```

**Log contains:**
- `seq` - Sequence number
- `timestamp_pc_us` - PC timestamp (microseconds)
- `j1_pos, j2_pos, ...` - Joint positions (radians)
- `j1_vel, j2_vel, ...` - Joint velocities (rad/s)
- `j1_acc, j2_acc, ...` - Joint accelerations (rad/s²)
- `command_sent` - Raw command string

**View the log:**
```bash
cat /workspace/logs/driver_commands_*.csv | head -20
```

---

## 🔬 What You Learn

### Message Format

**What ESP32 receives:**
```
<SEQ,J1,J2,J3,J4,J5,J6>
```

Example:
```
<0,0.5234,-0.8921,1.2345,-0.4567,0.7890,-0.3210>
<1,0.5244,-0.8925,1.2350,-0.4570,0.7895,-0.3215>
```

**What's in the PC log (full data):**
```csv
seq,timestamp_pc_us,j1_pos,j2_pos,...,j1_vel,j2_vel,...,j1_acc,j2_acc,...
0,1736781234567890,0.5234,-0.8921,...,0.05,0.03,...,0.01,0.02,...
1,1736781234667890,0.5244,-0.8925,...,0.05,0.03,...,0.01,0.02,...
```

### Latency Measurement

**PC log:** `timestamp_pc_us = 1736781234567890` (when command sent)  
**ESP32:** `Timestamp: 1234567 µs` (when received by ESP32)

**Note:** ESP32 timestamp resets on boot, so you can't directly compare. But you can measure:
- **Command interval:** Time between consecutive commands (PC log)
- **ESP32 response time:** How fast ESP32 ACKs (monitor output)

---

### Velocities and Accelerations

MoveIt calculates these! They're in the PC log:

```csv
j1_vel,j2_vel,j3_vel,j4_vel,j5_vel,j6_vel,j1_acc,j2_acc,j3_acc,j4_acc,j5_acc,j6_acc
0.05,0.03,-0.02,0.01,0.04,-0.01,0.01,0.02,0.00,0.01,0.01,0.00
```

**For motors:** You can use these to:
- Set motor speeds (from velocities)
- Plan smooth motion (from accelerations)
- Predict next position

---

## 🛠️ For Motor Control Firmware

Based on this test, your motor firmware should:

1. **Parse incoming format:**
   ```cpp
   // ESP32 receives: <SEQ,J1,J2,J3,J4,J5,J6>
   int seq;
   float joints[6];
   sscanf(buffer, "<%d,%f,%f,%f,%f,%f,%f>", 
          &seq, &joints[0], &joints[1], &joints[2], 
          &joints[3], &joints[4], &joints[5]);
   ```

2. **Handle timing:**
   - New command every ~50-100ms
   - Execute position before next arrives
   - Interpolate if needed

3. **Send ACK:**
   ```cpp
   printf("<ACK,%d,%lld>\n", seq, esp_timer_get_time());
   ```

4. **Add motor control:**
   ```cpp
   // Replace benchmark logging with:
   for(int i = 0; i < 6; i++) {
       moveMotor(i, joints[i]);  // Your motor function
   }
   ```

---

## ✅ Success Checklist

After running this test, you should know:

- [ ] ESP32 receives commands (**verified in monitor**)
- [ ] Message format is correct (`<SEQ,J1,...>`)
- [ ] Command rate (~10-20 Hz)
- [ ] No packet loss (sequence numbers continuous)
- [ ] PC log shows positions + velocities + accelerations
- [ ] Latency is acceptable (< 100ms end-to-end)

**If all checked → Ready for motor integration!**

---

## 🐛 Troubleshooting

### ESP32 shows "Invalid message format"

**Cause:** Driver not rebuilt after fix

**Fix:**
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select parol6_driver
exit
```

### No commands arriving at ESP32

**Check:**
1. ESP32 monitor shows "READY"
2. ROS terminal shows "Connected to Microcontroller"
3. You clicked "Execute" (not just "Plan")

### RViz doesn't open

**Fix:**
```bash
xhost +local:docker
./start_real_robot.sh
```

---

**This is exactly what you need to prepare for hardware integration!**
