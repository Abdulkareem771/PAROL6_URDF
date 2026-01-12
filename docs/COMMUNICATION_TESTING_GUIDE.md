# ROS-ESP32 Communication Testing Guide

**Purpose**: Verify serial communication integrity BEFORE connecting motors to ensure:
- Zero packet loss
- Acceptable latency (< 10ms typical)
- Proper data format (positions, velocities, accelerations)
- Timestamp synchronization

---

## 🎯 What We're Testing

| Test | Purpose |
|------|---------|
| **Latency** | Round-trip time (laptop → ESP32 → laptop) |
| **Bandwidth** | Max commands/second without loss |
| **Data Integrity** | Verify all 19 values received correctly |
| **Packet Loss** | Detect missing commands via sequence numbers |

---

## 🔧 Hardware Setup

### Requirements
- ESP32 DevKit (any variant)
- USB cable (data, not charge-only!)
- Laptop with Python 3 + matplotlib

### Wiring
No wiring needed! Just connect ESP32 via USB.

**Find Port:**
```bash
# Linux
ls /dev/ttyUSB* /dev/ttyACM*

# Common: /dev/ttyUSB0 or /dev/ttyACM0
```

---

## 📝 Step 1: Upload Benchmark Firmware

1. **Open Arduino IDE**
2. **Load**: `PAROL6/firmware/benchmark_firmware.ino`
3. **Configure**:
   ```cpp
   #define USE_SD_LOGGING false  // Set true if SD card attached
   #define LOG_TO_SERIAL true    // Print received data
   ```
4. **Select Board**: ESP32 Dev Module
5. **Select Port**: Your detected port
6. **Upload** ✅

### Verify Upload
Open **Serial Monitor** (115200 baud):
```
READY: ESP32_BENCHMARK_V1
Waiting for commands...
```

---

## 🚀 Step 2: Run Communication Test

### Install Dependencies
```bash
pip3 install pyserial matplotlib numpy
```

### Run Test (Basic)
```bash
cd /path/to/PAROL6_URDF/scripts
python3 test_driver_communication.py --port /dev/ttyUSB0
```

**Default Settings:**
- 100 packets
- 50ms delay between packets
- Saves report to `comm_test_report.png`

### Run Test (Custom)
```bash
# Stress test: 1000 packets, 10ms delay
python3 test_driver_communication.py \
  --port /dev/ttyUSB0 \
  --packets 1000 \
  --delay 10 \
  --output stress_test.png
```

---

## 📊 Understanding Results

### Good Results ✅
```
PERFORMANCE REPORT
==================================================
Packets Sent:     100
ACKs Received:    100
Packets Lost:     0
Loss Rate:        0.00%
Avg Latency:      4.23 ms
Min Latency:      3.81 ms
Max Latency:      6.45 ms
```

### Bad Results ⚠️
```
Packets Sent:     100
ACKs Received:    87
Packets Lost:     13
Loss Rate:        13.00%    ← PROBLEM!
Avg Latency:      45.2 ms   ← TOO HIGH!
```

**If you see bad results:**
- ❌ Bad USB cable (use data cable, not charge-only)
- ❌ Port conflict (close Arduino Serial Monitor)
- ❌ Wrong baud rate (must be 115200)
- ❌ Slow laptop (close other apps)

---

## 📈 Generated Report

The test creates `comm_test_report.png` with 4 graphs:

### Graph 1: Latency Over Time
- Shows if latency is stable or spiky
- **Good**: Flat line around 3-5ms
- **Bad**: Random spikes > 20ms

### Graph 2: Latency Distribution
- Histogram of latency values
- **Good**: Tight peak (low jitter)
- **Bad**: Wide spread (high jitter)

### Graph 3: Packet Success/Loss Pie Chart
- Visual success rate
- **Target**: >99% success

### Graph 4: Statistics Summary
- Text summary of all metrics

---

## 🔬 Advanced: Log Analysis

### ESP32 Logs (Serial Monitor)
You'll see output like:
```
LOG,1234567,SEQ:42,J:[0.523,-1.234,0.876,-0.432,1.098,-0.765]
```

**Fields:**
- `1234567`: Microsecond timestamp (ESP32 clock)
- `SEQ:42`: Sequence number
- `J:[...]`: Joint positions (first 6 values)

### Laptop Logs (CSV)
Check `comm_test_report.csv`:
```csv
seq,latency_ms
0,4.23
1,3.98
2,4.45
```

Use this for custom analysis in Excel/Python.

---

## 🎓 Interpreting for Real Robot

### Acceptable Thresholds

| Metric | Target | Acceptable | Problematic |
|--------|--------|------------|-------------|
| Packet Loss | 0% | < 0.1% | > 1% |
| Avg Latency | < 5ms | < 10ms | > 20ms |
| Max Latency | < 10ms | < 20ms | > 50ms |
| Jitter (Std Dev) | < 2ms | < 5ms | > 10ms |

### Why This Matters

**For a 6-DOF robot following a trajectory:**
- MoveIt sends waypoints every 50ms (20Hz)
- Each waypoint has 6 positions + velocities + accelerations = 18 values
- If latency > 50ms, commands pile up → jerky motion
- If packet loss > 0%, robot misses waypoints → path deviation

**Our Goal**: < 5ms latency, 0% loss

---

## 🛠️ Troubleshooting

### Issue: "Failed to connect"
**Cause**: Wrong port or permissions

**Fix**:
```bash
# Add user to dialout group
sudo usermod -aG dialout $USER
# Logout and login

# Or manual permission
sudo chmod 666 /dev/ttyUSB0
```

---

### Issue: High latency (> 20ms)
**Causes**:
1. Serial Monitor open (close it!)
2. Other USB devices on same hub
3. Slow baud rate (should be 115200)

**Fix**:
- Use direct USB port (not hub)
- Close Arduino IDE Serial Monitor
- Verify baud rate matches in both code

---

### Issue: Random packet loss (e.g., 2-5%)
**Causes**:
1. USB cable too long (> 2m)
2. Electromagnetic interference
3. Buffer overflow (sending too fast)

**Fix**:
- Use shorter, shielded cable
- Increase `--delay` parameter (test with 100ms)
- Move away from power supplies/motors

---

## ✅ Next Steps After Passing Tests

Once you achieve:
- ✅ 0% packet loss
- ✅ < 5ms average latency
- ✅ < 2ms jitter

You're ready to:
1. **Update firmware** to real motor control code
2. **Integrate with ROS driver** (`real_robot_driver.py`)
3. **Connect motors** (one at a time for safety)
4. **Run MoveIt** with real hardware

---

## 📞 Support

**If tests consistently fail:**
1. Share your `comm_test_report.png`
2. Share terminal output
3. Check:
   - ESP32 model
   - USB cable quality
   - Laptop OS/version

**Expected Results** (ESP32, 115200 baud, USB 2.0):
- Latency: 3-5ms average
- Loss: 0% (for 100-1000 packets)
- Max latency: < 10ms

---

**Last Updated**: January 2026  
**Maintained by**: PAROL6 Team
