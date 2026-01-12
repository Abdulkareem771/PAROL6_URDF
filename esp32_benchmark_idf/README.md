# ESP32 Benchmark Firmware - ESP-IDF Build Guide

## 🎯 Overview

This is the ESP-IDF version of the benchmark firmware - no Arduino IDE needed!
Everything builds and flashes from inside the Docker container.

---

## 🔧 Prerequisites

1. **ESP32 connected via USB**
2. **Docker container running** with ESP-IDF installed
3. **USB port accessible** to Docker

---

## 📦 Build & Flash

### Step 1: Start Unified Container

```bash
cd /path/to/PAROL6_URDF

# Start the persistent container (if not already running)
./start_container.sh
```

**This creates a container named `parol6_dev` that persists across sessions.**

---

### Step 2: Enter Container

```bash
docker exec -it parol6_dev bash
```

You're now inside the container shell.

---

### Step 3: Load ESP-IDF Environment

```bash
# Inside container
. /opt/esp-idf/export.sh
```

**Expected output:**
```
Done! You can now compile ESP-IDF projects.
```

---

### Step 4: Navigate to Project

```bash
cd /workspace/esp32_benchmark_idf
```

---

### Step 5: Configure Project (First Time Only)

```bash
# Set target to esp32
idf.py set-target esp32

# Optional: Configure settings
idf.py menuconfig
# Press 'Q' to quit, 'Y' to save (or just save defaults)
```

---

### Step 6: Build Firmware

```bash
idf.py build
```

**Build time**: ~30-60 seconds (first build), ~5-10 seconds (incremental)

**Expected output:**
```
Project build complete. To flash, run:
 idf.py -p /dev/ttyUSB0 flash
```

---

### Step 7: Flash to ESP32

```bash
# Flash and open monitor
idf.py -p /dev/ttyUSB0 flash monitor

# Or just flash without monitor
idf.py -p /dev/ttyUSB0 flash
```

**Flash time**: ~10 seconds

**Expected output in monitor:**
```
========================================
  ESP32 Benchmark Firmware (ESP-IDF)
========================================
Ready to receive commands...
Send 'STATS' to view statistics

READY: ESP32_BENCHMARK_V2
```

**Exit monitor**: Press `Ctrl + ]`

---

## 🚀 Alternative: One-Command Flash (From Host)

Instead of entering the container, you can flash directly from your host machine:

```bash
cd /path/to/PAROL6_URDF/esp32_benchmark_idf

# This script handles everything
./flash.sh /dev/ttyUSB0
```

**What it does:**
1. Starts a temporary container with USB access
2. Loads ESP-IDF
3. Builds firmware
4. Flashes to ESP32
5. Opens serial monitor

---

## 🧪 Testing

### From Python Test Script (Recommended)

```bash
# Exit the ESP32 monitor first (Ctrl + ])

# In another terminal on HOST machine
cd /path/to/PAROL6_URDF
python3 scripts/test_driver_communication.py --port /dev/ttyUSB0
```

**Or from inside the unified container:**

```bash
# In another terminal, enter container
docker exec -it parol6_dev bash

# Navigate and run
cd /workspace
python3 scripts/test_driver_communication.py --port /dev/ttyUSB0
```

---

### From ROS Driver (Full Integration Test)

```bash
# Inside unified container
./run_robot.sh real

# Or manually:
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch parol6_moveit_config unified_bringup.launch.py mode:=real
```

---

### Manual Testing (Serial Monitor)

```bash
# Inside container where you flashed
# After 'idf.py flash monitor', you're already in monitor

# Type commands manually:
<0,0.5,0.3,-0.2,0.1,0.4,-0.1>

# View stats:
STATS
```

**Expected response:**
```
<ACK,0,1234567>
I (1234) BENCHMARK: SEQ:0 J:[0.500,0.300,-0.200,0.100,0.400,-0.100]
```

---

## 🔍 Monitor Output

The firmware will log:
```
I (1234) BENCHMARK: SEQ:0 J:[0.500,0.300,-0.200,0.100,0.400,-0.100]
<ACK,0,1234567>
I (1245) BENCHMARK: SEQ:1 J:[0.520,0.310,-0.210,0.105,0.405,-0.105]
<ACK,1,1245678>
```

**Exit monitor**: `Ctrl + ]`

---

## ⚙️ Customization

### Change Baud Rate

Edit `main/benchmark_main.c`:
```c
.baud_rate = 115200,  // Change to 921600 for high speed
```

Then rebuild:
```bash
idf.py build flash
```

### Enable Verbose Logging

```bash
idf.py menuconfig
# Component config → Log output → Default log verbosity → Verbose
```

---

## 🐛 Troubleshooting

### Issue: "No serial data"

**Check port permissions:**
```bash
# On host
sudo chmod 666 /dev/ttyUSB0
```

**Check Docker has device access:**
```bash
docker run --device=/dev/ttyUSB0 ... # Must be specified
```

---

### Issue: "idf.py: command not found"

**Solution**: Load ESP-IDF environment
```bash
. /opt/esp/idf/export.sh
```

Add to `~/.bashrc` in container for persistence:
```bash
echo '. /opt/esp/idf/export.sh' >> ~/.bashrc
```

---

### Issue: "esptool.py failed"

**Causes:**
1. Wrong port specified
2. ESP32 in boot mode (hold BOOT button, press RESET)
3. USB cable issue

**Fix:**
```bash
# Try manual flash with slower speed
idf.py -p /dev/ttyUSB0 -b 115200 flash
```

---

### Issue: Build errors

**Clean rebuild:**
```bash
idf.py fullclean
idf.py build
```

---

## 📊 Comparison: Arduino IDE vs ESP-IDF

| Feature | Arduino IDE | ESP-IDF |
|---------|-------------|---------|
| Install | Separate app | In Docker |
| Build Speed | Slower | Faster |
| Code Size | Larger | Optimized |
| Features | Basic | Full SDK |
| USB Access | Direct | Via Docker flag |

**Recommendation**: Use ESP-IDF for production, Arduino for quick prototypes.

---

## 🚀 Next Steps

After successful flash:
1. ✅ Run communication test: `python3 scripts/test_driver_communication.py`
2. ✅ Test with ROS: Launch robot and plan/execute in RViz
3. ✅ Analyze logs: `python3 scripts/analyze_communication_logs.py`

---

**Last Updated**: January 2026  
**Maintained by**: PAROL6 Team
