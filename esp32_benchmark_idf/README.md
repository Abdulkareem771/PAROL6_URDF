# ESP32 Benchmark Firmware - Complete Guide

**Test ROS → ESP32 Communication Pipeline**

This guide walks you through everything from scratch: building firmware, flashing ESP32, testing communication, and analyzing results.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [One-Time Setup](#one-time-setup)
3. [Build & Flash Firmware](#build--flash-firmware)
4. [Testing](#testing)
5. [ROS Integration](#ros-integration)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Hardware
- ✅ ESP32 development board
- ✅ USB cable
- ✅ Computer running Linux (or WSL/Docker)

### Software
- ✅ Docker with `parol6-ultimate:latest` image
- ✅ ESP-IDF installed in Docker (`/opt/esp-idf/`)
- ✅ ROS 2 Humble

**Check Docker image:**
```bash
docker images | grep parol6-ultimate
```

---

## 🚀 One-Time Setup

### Step 1: Start Container

```bash
cd /path/to/PAROL6_URDF
./start_container.sh
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════╗
║     PAROL6 Unified Container Manager                       ║
╚══════════════════════════════════════════════════════════════╝

[INFO] Container 'parol6_dev' exists
[✓] Container started
```

---

### Step 2: Fix Python Environment (IMPORTANT!)

This prevents ESP-IDF Python from conflicting with ROS builds.

```bash
# Enter container
docker exec -it parol6_dev bash

# Run fix script
cd /workspace
./fix_python_env.sh

# Exit
exit
```

**Expected output:**
```
✅ Added PYTHON_EXECUTABLE to ~/.bashrc
✅ Set PYTHON_EXECUTABLE for current session
```

**This is permanent** - you only need to do it once!

---

### Step 3: Check ESP32 Connection

```bash
# On host machine
ls /dev/ttyUSB* /dev/ttyACM*

# Should show something like:
# /dev/ttyUSB0
```

**If not found:**
- Plug in ESP32 via USB
- Check cable (try different one if needed)
- Try different USB port

**Fix permissions:**
```bash
sudo chmod 666 /dev/ttyUSB0
```

---

## 📦 Build & Flash Firmware

### Method 1: Automatic (Recommended for First Time)

```bash
cd /path/to/PAROL6_URDF/esp32_benchmark_idf
./flash.sh /dev/ttyUSB0
```

**What happens:**
1. ✅ Starts temporary Docker container
2. ✅ Loads ESP-IDF environment
3. ✅ Builds firmware (~30 seconds first time)
4. ✅ Flashes to ESP32
5. ✅ Opens serial monitor

**Expected output:**
```
========================================
  ESP32 Benchmark Firmware (ESP-IDF)
========================================
Ready to receive commands...
READY: ESP32_BENCHMARK_V2
```

**To exit monitor:** Press `Ctrl + ]`

---

### Method 2: Manual (Step-by-Step)

For more control, build manually:

#### 2a. Enter Container

```bash
docker exec -it parol6_dev bash
```

#### 2b. Load ESP-IDF Environment

```bash
. /opt/esp-idf/export.sh
```

**Expected:**
```
Done! You can now compile ESP-IDF projects.
```

#### 2c. Navigate to Project

```bash
cd /workspace/esp32_benchmark_idf
```

#### 2d. Configure Target (First Time Only)

```bash
idf.py set-target esp32
```

**Expected:**
```
-- Configuring done
-- Generating done
-- Build files have been written to...
```

#### 2e. Build Firmware

```bash
idf.py build
```

**Build time:**
- First build: ~30-60 seconds
- Incremental: ~5-10 seconds

**Expected final line:**
```
Project build complete. To flash, run:
 idf.py -p /dev/ttyUSB0 flash
```

#### 2f. Flash to ESP32

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

**Flash time:** ~10 seconds

**Expected:**
```
Hash of data verified.

========================================
  ESP32 Benchmark Firmware (ESP-IDF)
========================================
Ready to receive commands...
READY: ESP32_BENCHMARK_V2
```

**Monitor will show incoming commands in real-time!**

---

## 🧪 Testing

### Test 1: Standalone Communication Test

Test PC ↔ ESP32 communication **without ROS**.

#### Step 1: Ensure Monitor is Closed

If ESP32 monitor is running, press `Ctrl + ]` to exit.

#### Step 2: Run Test Script

```bash
# On host machine (outside Docker)
cd /path/to/PAROL6_URDF
python3 scripts/test_driver_communication.py --port /dev/ttyUSB0
```

**Expected output:**
```
============================================================
  ROS-ESP32 Communication Integrity Test
============================================================
✓ Connected to /dev/ttyUSB0 at 115200 baud

[  1/100] ✓ ACK received | Latency: 29.75ms
[  2/100] ✓ ACK received | Latency: 29.72ms
...
[100/100] ✓ ACK received | Latency: 29.79ms

==================================================
PERFORMANCE REPORT
==================================================
Packets Sent:     100
ACKs Received:    100
Packets Lost:     0
Loss Rate:        0.00%
Avg Latency:      29.78 ms
```

**Success criteria:**
- ✅ Loss Rate: 0.00%
- ✅ Avg Latency: < 50ms

**If test fails, see [Troubleshooting](#troubleshooting)**

---

### Test 2: View Results

The test creates visualizations:

```bash
# View graph
xdg-open comm_test_report.png

# View CSV data
cat comm_test_report.csv
```

---

## 🤖 ROS Integration

Test the **full pipeline**: RViz → MoveIt → ROS Driver → ESP32

### Step 1: ESP32 Monitor (Terminal 1)

```bash
docker exec -it parol6_dev bash
cd /workspace/esp32_benchmark_idf
. /opt/esp-idf/export.sh
idf.py -p /dev/ttyUSB0 monitor
```

**Leave this open** - you'll watch commands arrive here!

---

### Step 2: Launch ROS (Terminal 2)

```bash
docker exec -it parol6_dev bash
cd /workspace

# Source ROS
source /opt/ros/humble/setup.bash

# Build (if not already done)
colcon build --symlink-install

# Source workspace
source install/setup.bash

# Launch robot in real mode
ros2 launch parol6_driver unified_bringup.launch.py mode:=real
```

**Expected:**
- RViz window opens
- Terminal shows: `[real_robot_driver]: Connected to Microcontroller at /dev/ttyUSB0`

---

### Step 3: Move Robot in RViz

1. **Drag** the interactive marker (orange/blue sphere at end-effector)
2. Click **"Plan"** button → MoveIt calculates trajectory
3. Click **"Execute"** button → Sends to ESP32

---

### Step 4: Watch Terminal 1 (ESP32 Monitor)

You'll see commands arriving:

```
<ACK,0,1234567>
I (1234) BENCHMARK: SEQ:0 J:[0.500,0.300,-0.200,0.100,0.400,-0.100]
<ACK,1,1334567>
I (1334) BENCHMARK: SEQ:1 J:[0.510,0.305,-0.205,0.102,0.402,-0.102]
<ACK,2,1434567>
...
```

**Each line shows:**
- `<ACK,SEQ,TIMESTAMP>` - Response to PC
- `I (time) BENCHMARK: SEQ:X J:[...]` - Joint positions

---

### Step 5: Collect Logs

ROS automatically logs all commands:

```bash
# After stopping ROS (Ctrl+C in Terminal 2)
ls /workspace/logs/

# You'll see:
# driver_commands_20260112_194500.csv
```

---

### Step 6: Analyze Logs (Optional)

Compare PC and ESP32 logs:

```bash
# Copy ESP32 monitor output to file
# (or use SD card log if firmware configured for it)

# Run analysis
python3 scripts/analyze_communication_logs.py \
  --pc-log /workspace/logs/driver_commands_*.csv \
  --esp-log esp32_log.csv
```

Creates graphs showing latency, packet loss, and data integrity!

---

## 🐛 Troubleshooting

### Issue: "Port not found" or "Permission denied"

**Symptoms:**
```
Error: /dev/ttyUSB0 not found
```

**Solutions:**

1. **Check connection:**
   ```bash
   ls /dev/ttyUSB* /dev/ttyACM*
   ```

2. **Fix permissions:**
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   ```

3. **Add user to dialout group (permanent):**
   ```bash
   sudo usermod -a -G dialout $USER
   # Logout and login again
   ```

4. **Try different port:**
   ```bash
   # ESP32 might be on different device
   ls -l /dev/tty*
   ```

---

### Issue: "idf.py: command not found"

**Cause:** ESP-IDF environment not loaded

**Fix:**
```bash
. /opt/esp-idf/export.sh
```

**Permanent fix:**
```bash
echo '. /opt/esp-idf/export.sh' >> ~/.bashrc
```

---

### Issue: "catkin_pkg not found" during colcon build

**Cause:** ESP-IDF Python is being used instead of system Python

**Fix:**
```bash
# Run the fix script (one-time)
docker exec -it parol6_dev bash
cd /workspace
./fix_python_env.sh
exit

# Or manually:
export PYTHON_EXECUTABLE=/usr/bin/python3
colcon build --symlink-install
```

---

### Issue: "100% Packet Loss" in test

**Cause:** ESP32 monitor or another program is using the port

**Fix:**

1. **Close ESP32 monitor:**
   - Press `Ctrl + ]` in monitor terminal

2. **Check for other programs:**
   ```bash
   sudo lsof /dev/ttyUSB0
   # Kill any processes using the port
   ```

3. **Reset ESP32:**
   - Press RESET button on board

4. **Retry test:**
   ```bash
   python3 scripts/test_driver_communication.py --port /dev/ttyUSB0
   ```

---

### Issue: RViz doesn't open

**Symptoms:**
```
Package 'parol6_moveit_config' not found
```

**Fix:**

1. **Check you're using correct launch file:**
   ```bash
   ros2 launch parol6_driver unified_bringup.launch.py mode:=real
   ```

2. **Rebuild workspace:**
   ```bash
   cd /workspace
   rm -rf build install log
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

3. **Fix X11 permissions (if RViz window doesn't appear):**
   ```bash
   xhost +local:root
   ```

---

### Issue: Build errors

**Full clean rebuild:**
```bash
cd /workspace/esp32_benchmark_idf
idf.py fullclean
idf.py set-target esp32
idf.py build
```

**Check ESP-IDF version:**
```bash
cd /opt/esp-idf
git describe --tags
# Should be v5.1.x or similar
```

---

## 📊 Advanced Configuration

### Change Baud Rate

Edit `main/benchmark_main.c`:
```c
.baud_rate = 115200,  // Change to 921600 for high speed
```

Rebuild:
```bash
idf.py build flash
```

### Enable SD Card Logging

(For firmware versions that support it - check source code)

---

## 📚 Additional Resources

- **Full Integration Test:** [docs/FULL_INTEGRATION_TEST_GUIDE.md](../docs/FULL_INTEGRATION_TEST_GUIDE.md)
- **Driver Testing:** [docs/ROS_DRIVER_TESTING_GUIDE.md](../docs/ROS_DRIVER_TESTING_GUIDE.md)
- **Container Guide:** [docs/UNIFIED_CONTAINER_GUIDE.md](../docs/UNIFIED_CONTAINER_GUIDE.md)

---

## 📁 Project Structure

```
esp32_benchmark_idf/
├── README.md              ← You are here
├── main/
│   ├── benchmark_main.c   ← ESP32 firmware source
│   └── CMakeLists.txt     ← Component build config
├── CMakeLists.txt         ← Project build config
├── flash.sh               ← Quick flash script
└── COLCON_IGNORE          ← Prevents ROS from building this
```

---

## ✅ Success Checklist for Teammates

- [ ] Container started with `./start_container.sh`
- [ ] Python environment fixed with `./fix_python_env.sh`
- [ ] ESP32 connected and detected (`ls /dev/ttyUSB0`)
- [ ] Firmware flashed successfully
- [ ] Standalone test shows 0% packet loss
- [ ] ROS launches and RViz opens
- [ ] ESP32 monitor shows commands when moving robot in RViz
- [ ] Logs saved in `/workspace/logs/`

**If all checked, you're ready for motor connection!**

---

**Last Updated:** January 2026  
**Maintained by:** PAROL6 Team

**Questions?** See detailed guides in `docs/` folder or ask the team!
