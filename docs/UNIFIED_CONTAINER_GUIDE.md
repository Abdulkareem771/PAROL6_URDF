# Unified Container Workflow Guide

**One Container for Everything** - Simulation, Real Hardware, and Vision Testing

---

## 🎯 The Problem (Before)

You currently have:
- `start_ignition.sh` → Creates temporary container for Gazebo
- `./flash.sh` → Creates temporary container for ESP32
- Multiple `docker run` commands → Creates many throwaway containers
- Confusion about which container is running

**Result**: Multiple containers, port conflicts, wasted resources.

---

## ✅ The Solution (Now)

**ONE persistent container** named `parol6_unified` that:
- Runs Gazebo simulation
- Connects to real hardware (ESP32)
- Hosts ROS nodes
- Provides RViz interface
- Handles vision processing

---

## 🚀 Quick Start

### 1. Start the Unified Container

```bash
cd /path/to/PAROL6_URDF
./start_container.sh
```

**Output:**
```
╔══════════════════════════════════════════════════════════════╗
║     PAROL6 Unified Container Manager                       ║
╚══════════════════════════════════════════════════════════════╝

[INFO] Creating new container 'parol6_unified'...
[✓] Detected ESP32 at /dev/ttyUSB0
[✓] Container created and started

╔══════════════════════════════════════════════════════════════╗
║  Container Status: READY                                    ║
╚══════════════════════════════════════════════════════════════╝
```

**This runs ONCE.** The container persists across reboots.

---

### 2. Launch Robot (Any Mode)

```bash
# Gazebo simulation
./run_robot.sh sim

# Real hardware (with ESP32)
./run_robot.sh real

# Fake mode (visualization only)
./run_robot.sh fake
```

**All modes use the SAME container!**

---

## 📚 Common Workflows

### Workflow 1: Simulation Testing

```bash
# Start container (if not already running)
./start_container.sh

# Launch Gazebo
./run_robot.sh sim

# RViz opens → Plan and execute trajectories
# When done, press Ctrl+C
```

---

### Workflow 2: Real Robot Testing

```bash
# Ensure ESP32 is connected
ls /dev/ttyUSB0  # Should exist

# Start container
./start_container.sh

# Launch real robot
./run_robot.sh real

# RViz opens → Robot follows commands via ESP32
```

---

### Workflow 3: Vision Development

```bash
# Start container
./start_container.sh

# Enter container
docker exec -it parol6_unified bash

# Setup vision environment (first time only)
source venv_vision/bin/activate
pip install ultralytics opencv-python

# Run vision nodes
ros2 run parol6_vision yolo_detector
```

---

### Workflow 4: ESP32 Firmware Development

```bash
# Container must be running
./start_container.sh

# Flash firmware (from host OR inside container)
cd esp32_benchmark_idf
./flash.sh /dev/ttyUSB0
```

---

## 🔧 Container Management

### Check Container Status

```bash
docker ps | grep parol6_unified
```

### Enter Container Shell

```bash
docker exec -it parol6_unified bash
```

### Stop Container

```bash
docker stop parol6_unified
```

### Restart Container

```bash
./start_container.sh  # Auto-starts if stopped
```

### Remove Container (Clean Slate)

```bash
docker stop parol6_unified
docker rm parol6_unified

# Next ./start_container.sh creates fresh container
```

---

## 🎯 Benefits of Unified Container

| Benefit | Description |
|---------|-------------|
| **No Port Conflicts** | Only one container accesses `/dev/ttyUSB0` |
| **Persistent State** | Build cache, logs, configs survive |
| **Resource Efficient** | One container vs many throwaway ones |
| **Simplified Workflow** | `./run_robot.sh [mode]` - that's it! |
| **Consistent Environment** | Same Python/ROS versions everywhere |

---

## 📂 Directory Structure Inside Container

```
/workspace/             (Your PAROL6_URDF folder)
├── logs/               (ROS driver logs)
├── venv_vision/        (Vision Python environment)
├── build/              (ROS build artifacts - persistent!)
├── install/            (ROS install - persistent!)
└── ...

/opt/ros/humble/        (ROS 2 installation)
/opt/esp-idf/           (ESP-IDF for firmware)
```

---

## 🐛 Troubleshooting

### Issue: "Container already exists"

**Solution**: This is normal! The script will start it:
```bash
./start_container.sh  # Just run again
```

---

### Issue: "Cannot connect to ESP32"

**Check**:
```bash
# 1. Is ESP32 plugged in?
ls /dev/ttyUSB0

# 2. Restart container to refresh USB access
docker restart parol6_unified

# 3. Check permissions
sudo chmod 666 /dev/ttyUSB0
```

---

### Issue: "RViz doesn't open"

**Fix X11 permissions**:
```bash
xhost +local:root
./start_container.sh  # Recreate container
```

---

### Issue: "Old containers still running"

**Clean up**:
```bash
# Stop all PAROL6 containers
docker stop $(docker ps -q --filter ancestor=parol6-ultimate:latest)

# Remove them
docker rm $(docker ps -aq --filter ancestor=parol6-ultimate:latest)

# Start fresh
./start_container.sh
```

---

## 🔄 Migration from Old Scripts

### Old Way → New Way

| Old Command | New Command |
|-------------|-------------|
| `./start_ignition.sh` | `./run_robot.sh sim` |
| `./bringup.sh real` | `./run_robot.sh real` |
| `docker run ... bash` | `./start_container.sh` then `docker exec -it parol6_unified bash` |

---

## ✅ Recommended Daily Workflow

```bash
# Morning: Start container
./start_container.sh

# Work on simulation
./run_robot.sh sim
# Test, develop, repeat...

# Switch to real hardware
# (Close sim first with Ctrl+C)
./run_robot.sh real

# Evening: Leave container running (it's persistent!)
# OR stop it: docker stop parol6_unified
```

**The container acts like a persistent development server.**

---

**Last Updated**: January 2026  
**Maintained by**: PAROL6 Team
