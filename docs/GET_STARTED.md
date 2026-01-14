# Getting Started - For New Team Members

**Welcome! This guide helps you set up and test the PAROL6 robot system from scratch.**

---

## 📋 Prerequisites

### Hardware
- Computer with Linux (Ubuntu 22.04 recommended)
- ESP32 development board
- USB cable
- (Optional) PAROL6 robot hardware

### Software
- Docker installed
- USB access permissions

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Get the Project

```bash
cd ~/Desktop
git clone <repository-url> PAROL6_URDF
cd PAROL6_URDF
```

### Step 2: Load Docker Image

```bash
# Load the pre-built image (ask team lead for the .tar file)
docker load < parol6-ultimate.tar

# Verify it loaded
docker images | grep parol6-ultimate
```

### Step 3: Test Simulation

```bash
# Start Ignition Gazebo
./start_ignition.sh

# In another terminal, add MoveIt + RViz
./add_moveit.sh
```

**You should see:** Robot in Gazebo + RViz with motion planning interface.

**Try it:** Drag the interactive marker (orange sphere), click "Plan", then "Execute".

---

## 🔧 ESP32 Communication Testing

### Step 1: Flash Firmware

```bash
# Plug in ESP32
ls /dev/ttyUSB0  # Should exist

# Navigate to firmware folder
cd esp32_benchmark_idf

# Read the README for detailed instructions
cat README.md

# Quick flash
./flash.sh /dev/ttyUSB0
```

See **[esp32_benchmark_idf/README.md](../esp32_benchmark_idf/README.md)** for full guide.

### Step 2: Test Communication

```bash
# Simple standalone test
python3 scripts/test_driver_communication.py --port /dev/ttyUSB0
```

**Expected:** 0% packet loss, ~30ms latency

### Step 3: Test with ROS

See **[esp32_benchmark_idf/TESTING_WITH_ROS.md](../esp32_benchmark_idf/TESTING_WITH_ROS.md)** for full ROS pipeline testing.

---

## 📚 Documentation Structure

```
PAROL6_URDF/
├── README.md                          ← Project overview
├── GET_STARTED.md                     ← You are here!
├── start_ignition.sh                  ← Launch simulation
├── add_moveit.sh                      ← Add MoveIt to simulation
├── start_real_robot.sh                ← Launch real hardware mode
│
├── docs/                              ← General documentation
│   ├── RVIZ_SETUP_GUIDE.md           ← Fix RViz display issues
│   ├── FULL_INTEGRATION_TEST_GUIDE.md
│   ├── ROS_DRIVER_TESTING_GUIDE.md
│   └── ...
│
├── esp32_benchmark_idf/               ← ESP32 firmware & testing
│   ├── README.md                      ← Full ESP32 guide
│   ├── QUICK_START.md                 ← Fast testing
│   ├── TESTING_WITH_ROS.md            ← ROS pipeline test
│   ├── main/benchmark_main.c          ← Firmware source
│   └── flash.sh                       ← Build & flash script
│
└── parol6_driver/                     ← ROS driver
    └── parol6_driver/real_robot_driver.py
```

---

## 🎯 Common Tasks

### Test robot in simulation
```bash
./start_ignition.sh
./add_moveit.sh
# Plan and execute in RViz
```

### Test ESP32 communication
```bash
cd esp32_benchmark_idf
./flash.sh /dev/ttyUSB0
python3 ../scripts/test_driver_communication.py --port /dev/ttyUSB0
```

### Test full ROS → ESP32 pipeline
```bash
# See esp32_benchmark_idf/TESTING_WITH_ROS.md
./start_real_robot.sh
# Move robot in RViz, watch ESP32 monitor
```

### RViz issues (robot not visible, no markers)
```bash
# See docs/RVIZ_SETUP_GUIDE.md
```

---

## 🐛 Troubleshooting

### "Container already exists"
```bash
docker stop parol6_dev && docker rm parol6_dev
# Then retry
```

### "Permission denied" on /dev/ttyUSB0
```bash
sudo chmod 666 /dev/ttyUSB0
# Or permanently:
sudo usermod -a -G dialout $USER
# Logout and login again
```

### RViz doesn't open
```bash
xhost +local:docker
# Then retry
```

### ESP32 shows "Invalid message format"
- Check you're using the latest driver code
- Rebuild: `docker exec -it parol6_dev bash`, then `cd /workspace && colcon build --packages-select parol6_driver`

---

## 📖 Next Steps

1. ✅ **Completed simulation test?** → Move to ESP32 testing
2. ✅ **ESP32 working?** → Test full ROS pipeline  
3. ✅ **Everything working?** → Ready for motor integration!

For detailed information, see:
- **ESP32:** [esp32_benchmark_idf/README.md](esp32_benchmark_idf/README.md)
- **ROS Testing:** [esp32_benchmark_idf/TESTING_WITH_ROS.md](esp32_benchmark_idf/TESTING_WITH_ROS.md)
- **RViz Setup:** [docs/RVIZ_SETUP_GUIDE.md](docs/RVIZ_SETUP_GUIDE.md)

---

**Questions?** Ask the team lead or check the detailed guides above!
