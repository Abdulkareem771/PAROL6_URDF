# PAROL6 Complete Setup Guide for Teammates

**Status:** Day 3 Complete ✅ | Ready for Motor Integration (Day 4)

This guide provides **everything** you need to build, flash, and test the PAROL6 robot with the validated ros2_control feedback loop.

---

## 📋 Prerequisites

- Ubuntu 22.04 (host system)
- Docker installed
- ESP32 DevKit with USB cable (CP2102 or CH340 driver)
- Arduino IDE or ESP-IDF installed
- USB port available (will appear as `/dev/ttyUSB0`)

---

## 🏗️ System Architecture Overview

### Data Path (High-Level)

```
User (RViz)
    ↓
MoveIt Planner ────► Trajectory Generation
    ↓
parol6_arm_controller ────► 25Hz Control Loop
    ↓
PAROL6System (C++) ────► Serial Protocol Handler
    ↓
USB Serial (/dev/ttyUSB0) ────► 115200 baud
    ↓
ESP32 Firmware ────► Command Parser
    ↓
Motor Drivers ────► MKS Servo42C
    ↓
Encoder Feedback ────► Actual Positions
    ↓
ESP32 ────► Feedback Message
    ↓
USB Serial ────► Back to PAROL6System
    ↓
joint_state_broadcaster ────► /joint_states topic
    ↓
RViz Visualization ────► User sees motion
```

### Key ROS 2 Nodes and Their Roles

| Node | Package | Purpose | Troubleshooting |
|------|---------|---------|----------------|
| **ros2_control_node** | controller_manager | Loads hardware interface, manages controllers | If this crashes, entire system stops. Check serial port access. |
| **PAROL6System** | parol6_hardware | C++ hardware interface, handles serial I/O | Logs show `📥 Raw feedback:`. If silent, ESP32 not responding. |
| **joint_state_broadcaster** | ros2_controllers | Publishes `/joint_states` at 25Hz | If `/joint_states` missing, controller not activated. |
| **parol6_arm_controller** | ros2_controllers | Executes trajectories, enforces tolerances | Aborts if position error too large. Tune tolerances if needed. |
| **move_group** | moveit_ros_move_group | Motion planning (collision-free paths) | Only runs when RViz launch is used. |
| **robot_state_publisher** | robot_state_publisher | Converts joint states → TF transforms | Required for RViz to display robot model. |
| **rviz2** | rviz2 | Visualization and interactive control | If empty, check robot_state_publisher is running. |

### Where does ros2_control fit?

**ros2_control** is the **hardware abstraction layer** that sits between high-level controllers (MoveIt, trajectory planners) and your physical hardware (ESP32, motors).

```
High-Level                ros2_control Framework              Hardware
─────────────────────     ───────────────────────────────     ────────────────
MoveIt / Planners   ──►   Controller Manager                 
                           ├─► Joint Trajectory Controller    
                           │   (parol6_arm_controller)        
                           │                                    
                           └─► Hardware Interface  ──────────► ESP32 + Motors
                               (PAROL6System.cpp)              
                                   │                            
                                   ├─► write(): Send commands  
                                   └─► read(): Get feedback    
```

**Key Concept:** ros2_control **decouples** trajectory planning from hardware communication.

- **Controllers** (like `parol6_arm_controller`) don't know about serial ports or ESP32
- **Hardware Interface** (`PAROL6System`) doesn't know about trajectories or MoveIt
- They communicate through **standardized interfaces**: `CommandInterface` and `StateInterface`

**Why this matters:**
- ✅ Swap ESP32 for CAN bus → Only change `PAROL6System`, controllers stay the same
- ✅ Add a gripper → Add new hardware interface, reuse existing controllers
- ✅ Test in simulation → Use `FakeHardware`, same controllers

**Files involved:**
- `parol6_hardware/src/parol6_system.cpp` - Your custom hardware interface
- `parol6_hardware/config/parol6_controllers.yaml` - Controller configuration
- `parol6_hardware/urdf/parol6.ros2_control.xacro` - Hardware interface definition

**Official docs:** https://control.ros.org/master/index.html

### Expected Performance Metrics

**Validated on 2026-01-16:**

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Control Loop Rate | 25 Hz | 25.0 Hz ±0.28ms | ✅ |
| Serial Baud Rate | 115200 | 115200 | ✅ |
| Max Serial Latency | <100 ms | ~40-50 ms | ✅ |
| Packet Loss | <1% | 0.0% | ✅ |
| Sequence Tracking | Continuous | 0 → 3149+ | ✅ |
| Position Accuracy | ±0.05 rad | ±0.01 rad (test data) | ✅ |

**What this means for you:**
- If you see >1% packet loss → Check USB cable or ESP32 power
- If latency >100ms → System overloaded or serial buffer full
- If control rate <20Hz → Controller timing issue, check CPU usage

---

## 🚀 Quick Start (5 Steps)

### 1. Start the Docker Container

```bash
cd /path/to/PAROL6_URDF
./start_container.sh
```

Expected output:
```
✓ Container '$CONTAINER_NAME' exists
✓ Container is already running
Container Status: READY
```

### 2. Build the Workspace (First Time Only)

```bash
docker exec -it parol6_dev bash
cd /workspace
colcon build --symlink-install
source install/setup.bash
```

**Build time:** ~3-5 minutes

### 3. Flash ESP32 Firmware

**Option A: Arduino IDE (Recommended)**

1. Open `PAROL6/firmware/esp32_feedback_arduino.ino`
2. Select **Board:** ESP32 Dev Module
3. Select **Port:** (your ESP32 port)
4. Click **Upload**

**Option B: ESP-IDF (Advanced)**

```bash
cd /path/to/PAROL6_URDF/esp32_feedback_firmware
docker run --rm -v $PWD:/project -w /project --device=/dev/ttyUSB0 espressif/idf:v5.1 \
  idf.py flash -p /dev/ttyUSB0
```

### 4. Launch the Hardware Interface

**Terminal 1 - Hardware Driver:**
```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch parol6_hardware real_robot.launch.py
```

**What to look for:**
```
✅ Serial opened successfully: /dev/ttyUSB0 @ 115200
✅ First feedback received (seq 0)
📥 Raw feedback: <ACK,150,0.00,0.00,0.00,0.00,0.00,0.00>
[controller_manager]: Activating parol6_arm_controller
[controller_manager]: Successfully activated parol6_arm_controller
```

### 5. Launch RViz for Visualization

**Terminal 2 - Visualization:**
```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py use_sim_time:=false
```

**RViz should open** showing the PAROL6 robot model.

---

## 🎯 Testing the Feedback Loop

### Interactive Motion Test

1. In RViz, drag the **interactive marker** or use the **Planning** tab
2. Click **Plan** → **Execute**
3. **Expected behavior:**
   - Terminal 1 shows: `Received new action goal`
   - Terminal 1 shows updated feedback values
   - RViz updates robot visualization

### Monitor Feedback Data

**View raw serial messages (Terminal 1):**
```
📥 Raw feedback: <ACK,SEQ,J1,J2,J3,J4,J5,J6>
```

**View published joint states (Terminal 3):**
```bash
docker exec -it parol6_dev bash
source /workspace/install/setup.bash
ros2 topic echo /joint_states
```

**Graphical plot:**
```bash
ros2 run rqt_plot rqt_plot /joint_states/position[0]:position[1]:position[2]
```

---

## 🔧 Troubleshooting

### ESP32 Not Detected

**Check USB connection:**
```bash
lsusb | grep -i "CP210\|CH340\|Silicon Labs"
ls -l /dev/ttyUSB*
```

**If broken symlink exists:**
```bash
sudo rm /dev/ttyUSB0  # Remove old symlink
# Unplug and replug ESP32
ls -l /dev/ttyUSB0    # Should show: crw-rw---- 1 root dialout
```

### Container Won't Start

```bash
docker stop parol6_dev
docker rm parol6_dev
./start_container.sh  # Recreate
```

### Build Errors

```bash
# Clean build
cd /workspace
rm -rf build install log
colcon build --symlink-install
```

### RViz Crashes or Empty

**Ensure both launches are running:**
1. Terminal 1: `real_robot.launch.py` (hardware interface)
2. Terminal 2: `demo.launch.py` (visualization)

**Check topics:**
```bash
ros2 topic list | grep joint_states
```

### ROS 2 Commands Fail with `!rclpy.ok()` Error

**Symptom:**
```
xmlrpc.client.Fault: <Fault 1: "<class 'RuntimeError'>:!rclpy.ok()">
```

**Cause:** ROS 2 daemon has crashed or is stuck

**Fix:**
```bash
# Kill stuck processes
pkill -9 ros

# Clean daemon
rm -rf ~/.ros/log/*

# Restart daemon
ros2 daemon stop
ros2 daemon start

# Verify
ros2 topic list
```

**If that doesn't work, restart the container:**
```bash
exit  # Exit container
docker restart parol6_dev
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
```

### rqt_plot or Other RQT Tools Not Found

**Install missing tools:**
```bash
# Inside container
apt-get update
apt-get install ros-humble-rqt-plot ros-humble-rqt-graph
```

---

## 📊 Day 3 Validation Checklist

✅ **Serial Communication**
- [ ] ESP32 detected at `/dev/ttyUSB0`
- [ ] Serial opened with 115200 baud
- [ ] No "Bad file descriptor" errors

✅ **Feedback Loop**
- [ ] `✅ First feedback received` message appears
- [ ] Raw feedback logs print every 2 seconds
- [ ] Sequence number incrementing

✅ **ROS Control**
- [ ] Controllers activate successfully
- [ ] `/joint_states` topic publishes at 25Hz
- [ ] `ros2 topic echo /joint_states` shows values

✅ **Interactive Control**
- [ ] RViz loads with robot model
- [ ] Can send motion commands
- [ ] Controller receives goals

---

## 🎓 Day 4: Motor Integration (Next Steps)

**Current status:** Feedback loop validated with **test data**  
**Next goal:** Connect real motors and send/receive actual positions

### What Changes for Day 4

1. **Hardware:** Connect MKS Servo42C motors to ESP32
2. **Firmware:** Update to send commands to motors and read encoders
3. **Testing:** Verify actual motion matches commands

### For Your Teammate

**Before plugging in motors:**
1. Ensure Day 3 validation passes completely
2. Review the motor control firmware (see below)
3. Test with **one motor first** to verify wiring

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `start_container.sh` | Start/create Docker environment |
| `parol6_hardware/launch/real_robot.launch.py` | Launch hardware interface |
| `parol6_moveit_config/launch/demo.launch.py` | Launch RViz visualization |
| `PAROL6/firmware/esp32_feedback_arduino.ino` | Current feedback test firmware |
| `PAROL6/firmware/esp32_motor_control.ino` | Day 4 motor control firmware (see below) |
| `parol6_hardware/src/parol6_system.cpp` | C++ hardware interface |

---

## 🆘 Getting Help

**Common Issues:**

| Symptom | Solution |
|---------|----------|
| `Permission denied` on serial port | Add user to dialout group: `sudo usermod -a -G dialout $USER` (logout/login) |
| Container not found | Run `./start_container.sh` |
| Build fails | Delete `build/`, `install/`, `log/` and rebuild |
| No `/joint_states` topic | Ensure `real_robot.launch.py` is running |
| RViz empty | Check that both Terminal 1 and 2 are running |

**Day 3 is DONE when:**
- ✅ You see `📥 Raw feedback:` logs every 2 seconds
- ✅ Sequence numbers increment without packet loss
- ✅ RViz shows the robot and accepts motion commands
- ✅ `/joint_states` updates in real-time

---

## 📚 Important Documents Reference

### Quick Reference by Task

| What You Want to Do | Read This Document |
|---------------------|--------------------|
| **Set up from scratch** | This file (`docs/TEAMMATE_SETUP_GUIDE.md`) |
| **Understand complete 5-phase plan** | `docs/ROS2_CONTROL_IMPLEMENTATION_PLAN.md` ⭐ |
| **Understand Day 4/5 plan** | `docs/DAY4_DAY5_IMPLEMENTATION_PLAN.md` |
| **Troubleshoot RViz issues** | `docs/RVIZ_SETUP_GUIDE.md` |
| **Understand the C++ driver** | `parol6_hardware/HARDWARE_INTERFACE_GUIDE.md` |
| **See Day 3 validation proof** | `.gemini/antigravity/brain/.../day3_walkthrough.md` |
| **Test ESP32 standalone** | `esp32_benchmark_idf/QUICK_START.md` |
| **Understand ROS pipeline** | `esp32_benchmark_idf/ROS_SYSTEM_ARCHITECTURE.md` |
| **Develop ESP32 firmware** | `esp32_benchmark_idf/DEVELOPER_GUIDE.md` |
| **Set up Git workflow** | `docs/TEAM_WORKFLOW_GUIDE.md` |
| **Automate project board** | `scripts/PROJECT_AUTOMATION.md` |

### Core Documentation Files

#### Architecture & Planning
- **`docs/ROS2_CONTROL_IMPLEMENTATION_PLAN.md`** ⭐ MASTER PLAN
  - Complete 5-phase migration strategy (SIL → EIL → HIL → PIL)
  - Critical design decisions (update rate, non-blocking I/O, parsing)
  - Validation taxonomy (academic terminology)
  - Dual-gate validation strategy
  - 3000+ lines of comprehensive guidance

#### Setup & Onboarding
- **`docs/TEAMMATE_SETUP_GUIDE.md`** (this file)
  - Complete setup: build, flash, run, test
  - System architecture overview
  - Day 3 validation checklist

- **`docs/DAY4_DAY5_IMPLEMENTATION_PLAN.md`**
  - Day 4: Motor integration step-by-step
  - Day 5: Validation procedures (engineering/thesis gates)
  - Detailed wiring diagrams and troubleshooting

#### Hardware & Firmware
- **`parol6_hardware/HARDWARE_INTERFACE_GUIDE.md`**
  - C++ driver architecture (`PAROL6System`)
  - Serial protocol implementation
  - Packet loss tracking, non-blocking I/O

- **`PAROL6/firmware/esp32_motor_control.ino`**
  - Day 4 motor control firmware (MKS Servo42C)
  - Protocol implementation with inline comments
  - Configuration parameters (steps/rev, microstepping)

- **`esp32_benchmark_idf/DEVELOPER_GUIDE.md`**
  - ESP-IDF concepts (FreeRTOS, UART, timers)
  - Motor integration examples
  - Debugging tips

#### ROS & System Architecture
- **`esp32_benchmark_idf/ROS_SYSTEM_ARCHITECTURE.md`**
  - Complete ROS 2 pipeline (RViz → MoveIt → Driver → ESP32)
  - Topics, actions, parameters explained
  - How to add custom functionality

- **`parol6_moveit_config/`**
  - MoveIt configuration (SRDF, OMPL, kinematics)
  - RViz config: `rviz/moveit.rviz`
  - Controller config: `config/moveit_controllers.yaml`

#### Troubleshooting
- **`docs/RVIZ_SETUP_GUIDE.md`**
  - Robot visibility issues
  - Interactive markers not working
  - Camera positioning, display settings

- **`TROUBLESHOOTING.md`** (if exists)
  - Common issues and solutions
  - Build errors, runtime errors

#### Validation & Evidence
- **`.gemini/antigravity/brain/.../day3_walkthrough.md`**
  - Day 3 validation evidence
  - Raw feedback logs, sequence tracking proof
  - Performance metrics table

- **`logs/` directory**
  - CSV logs: `driver_commands_*.csv`
  - Contains positions, velocities, accelerations
  - Use `scripts/quick_log_analysis.py` to analyze

#### Project Management
- **`docs/TEAM_WORKFLOW_GUIDE.md`**
  - Git branch strategy
  - Pull request process
  - Code review guidelines

- **`scripts/PROJECT_AUTOMATION.md`**
  - GitHub Projects automation
  - Task tracking setup

### External Resources

**ROS 2 Control:**
- Official docs: https://control.ros.org/master/index.html
- Hardware interface tutorial: https://control.ros.org/master/doc/ros2_control/hardware_interface/doc/writing_new_hardware_interface.html

**MoveIt 2:**
- Official docs: https://moveit.picknik.ai/main/index.html
- Tutorials: https://moveit.picknik.ai/main/doc/tutorials/tutorials.html

**ESP32:**
- ESP-IDF Programming Guide: https://docs.espressif.com/projects/esp-idf/en/latest/
- Arduino-ESP32: https://docs.espressif.com/projects/arduino-esp32/en/latest/

**MKS Servo42C:**
- GitHub: https://github.com/makerbase-mks/MKS-SERVO42C
- Communication protocol: See firmware code comments

---

## 🆘 Getting Help

**Questions?** Contact:
- **Kareem** (Project Lead)
- Repository: Check issues/discussions
- Documentation: Start with files above

**Recommended reading order for new teammates:**
1. This file (setup)
2. `HARDWARE_INTERFACE_GUIDE.md` (understand C++ driver)
3. `ROS_SYSTEM_ARCHITECTURE.md` (complete picture)
4. `DAY4_DAY5_IMPLEMENTATION_PLAN.md` (when ready for motors)
