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
rqt_plot /joint_states/position[0]:position[1]:position[2]
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

**Questions?** Contact Kareem or refer to:
- `docs/RVIZ_SETUP_GUIDE.md`
- `parol6_hardware/HARDWARE_INTERFACE_GUIDE.md`
- `esp32_benchmark_idf/TESTING_WITH_ROS.md`
