# ESP32 Communication Testing - Simple Guide

**Use Your Existing Working Scripts + Test ESP32**

---

## ✅ What Works (Your Original Workflow)

```bash
# 1. Start Simulation
./start_ignition.sh

# 2. Add MoveIt (in another terminal)
./add_moveit.sh

# Result: Robot appears, planning works!
```

---

## 🔧 Testing ESP32 Communication

### Option 1: Standalone Test (No ROS)

```bash
# Flash ESP32 firmware
cd esp32_benchmark_idf
./flash.sh /dev/ttyUSB0

# Test communication
python3 scripts/test_driver_communication.py --port /dev/ttyUSB0

# Expected: 0% packet loss, ~30ms latency
```

**This is all you need to verify ESP32 works!**

---

### Option 2: Test with Real Robot (Optional)

**Only do this if you want to test with actual hardware.**

```bash
# 1. Make sure ESP32 is plugged in and ready
ls /dev/ttyUSB0

# 2. Use your working real robot script
./start_real_robot.sh

# 3. Plan and execute in RViz
# Commands will be sent to ESP32
```

**Check logs:**
```bash
ls /workspace/logs/driver_commands_*.csv
```

---

## 📊 What Was Fixed

The driver now sends the correct format:
- **Before:** `<J1,J2,J3,J4,J5,J6>` ❌
- **After:** `<SEQ,J1,J2,J3,J4,J5,J6>` ✅

ESP32 will accept commands and respond with timestamps!

---

## 🎯 Recommendation

**For your thesis work:**

1. **Simulation development:** Use `./start_ignition.sh` + `./add_moveit.sh` (works great!)
2. **ESP32 verification:** Use standalone test script (Option 1 above)
3. **Real hardware:** Use `./start_real_robot.sh` when ready for motors

**Don't use the new unified scripts** - they're incomplete and not needed.

---

**Your original workflow is perfect. We just needed to fix the driver message format, which is done!**
