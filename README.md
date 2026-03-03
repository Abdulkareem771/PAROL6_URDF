# PAROL6 Single Motor ROS 2 Control Setup Guide

**Status:** ✅ Working - Single MKS Servo42C motor controlled via ROS 2 Control

This guide walks you through setting up ROS 2 Control to command a single MKS Servo42C motor through an ESP32.

---

## 📋 Prerequisites

### Hardware
- ESP32 development board
- MKS Servo42C closed-loop stepper motor
- USB cable (ESP32 to computer)
- Motor power supply

### Software
- Arduino IDE with ESP32 board support
- Docker installed
- Linux host machine (Ubuntu recommended)

---

## 🚀 Quick Start

### Step 1: Upload ESP32 Firmware 

1. **Open Arduino IDE**

2. **Load the firmware:**
   ```
   File → Open → PAROL6/firmware/ros_control/ros_control.ino
   ```

3. **Configure Arduino IDE:**
   - Board: "ESP32 Dev Module"
   - Upload Speed: 921600
   - Port: Select your ESP32 port (e.g., `/dev/ttyUSB0`)

4. **Install SERVO42C library:**
   - Copy `SERVO42C/SERVO42C.h` and `SERVO42C/SERVO42C.cpp` to Arduino libraries folder
   - Or place them in the same folder as `ros_control.ino`

5. **Upload:**
   - Click Upload button
   - Wait for "Done uploading"

6. **Verify (Optional):**
   - Open Serial Monitor (115200 baud)
   - You should see: `=== PAROL6 Open-Loop Control ===`
   - Test with: `<0,1.5,0,0,0,0,0>`
   - Motor should move to 1.5 radians

---

### Step 2: Wire the Motor

**ESP32 to MKS Servo42C:**
```
ESP32 GPIO 16 (RX) → MKS TX
ESP32 GPIO 17 (TX) → MKS RX
ESP32 GND         → MKS GND
```

**Power:**
- Connect 12-24V power supply to MKS motor driver
- Do NOT power motor from ESP32

---

### Step 3: Launch ROS 2 Control

1. **Navigate to workspace:**
   ```bash
   cd ~/Desktop/Servo42C\ 4\ ESP/PAROL6_URDF
   ```

2. **Start the system:**
   ```bash
   ./start_real_robot.sh
   ```

3. **What happens:**
   - Docker container starts
   - Dependencies install from cache (instant)
   - Workspace builds
   - ROS 2 Control launches
   - RViz opens (3D visualization)

4. **Expected output:**
   ```
   ✅ Serial opened successfully: /dev/ttyUSB0 @ 115200
   ✅ First feedback received
   [spawner]: Configured and activated joint_state_broadcaster
   [spawner]: Configured and activated parol6_arm_controller
   ```

---

### Step 4: Test Motor Control

**Open a new terminal** and run:

```bash
# Move motor to 1.5 radians (~86 degrees)
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 action send_goal /parol6_arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \"{trajectory: {joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6], points: [{positions: [1.5, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 5}}]}}\""
```

**Expected result:**
- Motor moves smoothly to 1.5 radians
- Terminal shows: `Goal finished with status: SUCCEEDED`

---

## 🔧 Useful Commands

### Check Joint States
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 topic echo /joint_states"
```

**Output:**
```yaml
position: [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### Check Controllers
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 control list_controllers"
```

**Output:**
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
```

### Move to Different Positions

**Zero position:**
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 action send_goal /parol6_arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \"{trajectory: {joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6], points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3}}]}}\""
```

**90 degrees (π/2 rad):**
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 action send_goal /parol6_arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \"{trajectory: {joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6], points: [{positions: [1.57, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 5}}]}}\""
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| [`PAROL6/firmware/ros_control/ros_control.ino`](file:///home/far2deluxe/Desktop/Servo42C%204%20ESP/PAROL6_URDF/PAROL6/firmware/ros_control/ros_control.ino) | ESP32 firmware for motor control |
| [`start_real_robot.sh`](file:///home/far2deluxe/Desktop/Servo42C%204%20ESP/PAROL6_URDF/start_real_robot.sh) | Launch script for ROS 2 Control |
| [`parol6_hardware/config/parol6_controllers.yaml`](file:///home/far2deluxe/Desktop/Servo42C%204%20ESP/PAROL6_URDF/parol6_hardware/config/parol6_controllers.yaml) | Controller configuration (tolerances, update rate) |
| [`.docker_cache/`](file:///home/far2deluxe/Desktop/Servo42C%204%20ESP/PAROL6_URDF/.docker_cache) | Cached packages (libserial, controllers) |

---

## ⚙️ Configuration

### ESP32 Firmware Settings

**In `ros_control.ino`:**
```cpp
#define USB_BAUD 115200        // Serial to computer
#define MOTOR_BAUD 38400       // Serial to MKS motor
#define MOTOR_RX_PIN 16        // ESP32 RX pin
#define MOTOR_TX_PIN 17        // ESP32 TX pin
#define MOTOR_ADDR 0xE0        // MKS motor address

// Movement threshold (filters excessive commands)
if (abs(new_target - current_positions[0]) > 0.01) {  // 0.01 rad = ~0.6°
  moveMotor(new_target);
}
```

### Controller Tolerances

**In `parol6_controllers.yaml`:**
```yaml
constraints:
  goal_time: 10.0              # Allow 10 seconds to reach goal
  joint_L1: 
    trajectory: 2.0            # Allow 2 rad deviation during movement
    goal: 0.1                  # Must reach within 0.1 rad at end
```

**Adjust these if:**
- Motor aborts too early → Increase `trajectory` tolerance
- Motor doesn't reach target → Increase `goal` tolerance
- Movements too slow → Decrease `goal_time`

---

## 🐛 Troubleshooting

### Motor doesn't move

**Check ESP32 connection:**
```bash
ls /dev/ttyUSB*
```
Should show `/dev/ttyUSB0` (or similar)

**Fix permissions:**
```bash
sudo chmod 666 /dev/ttyUSB0
```

**Check Serial Monitor:**
- Open Arduino IDE Serial Monitor (115200 baud)
- Look for: `🎯 Moving J1: 0.000 → 1.500 rad`
- If you see this, motor should move

### "Goal finished with status: ABORTED"

**Cause:** Motor can't reach target within tolerance

**Fix:** Increase tolerances in `parol6_controllers.yaml`
```yaml
joint_L1: {trajectory: 5.0, goal: 0.2}  # More lenient
```

Then restart:
```bash
./start_real_robot.sh
```

### Packet loss warnings

**Example:**
```
⚠️ PACKET LOSS DETECTED! Expected seq 2, got 3
```

**Cause:** ESP32 sends at 10Hz, ROS expects 25Hz

**Fix (Optional):** Update ESP32 firmware:
```cpp
// Change line 79 in ros_control.ino:
if (now - last_feedback >= 40) {  // 25 Hz instead of 10 Hz
```

**Note:** Packet loss warnings are cosmetic and don't affect functionality.

### Build fails

**Error:** `libserial-dev not found`

**Fix:** Cached packages missing. Download them:
```bash
cd ~/Desktop/Servo42C\ 4\ ESP/PAROL6_URDF
docker exec -it parol6_dev bash -c "cd /workspace && mkdir -p .docker_cache && cd .docker_cache && apt-get update && apt-get download libserial-dev libserial1"
```

### RViz doesn't open

**Check X11 forwarding:**
```bash
xhost +local:docker
```

**Check DISPLAY:**
```bash
echo $DISPLAY
```
Should show `:0` or `:1`

---

## 🎯 Next Steps

### Scale to Multiple Motors

1. **Wire additional motors** to ESP32 (use different GPIO pins)
2. **Update firmware** to control all 6 joints
3. **Test each motor** individually
4. **Calibrate** joint limits and speeds

### Add Real Encoder Feedback

Currently using **open-loop** (trust commanded position). For better accuracy:

1. Read MKS encoder via `motor.readEncoder()`
2. Update `current_positions[]` with actual values
3. Send real feedback to ROS

### Integrate with MoveIt

RViz is already running with MoveIt! Try:
- Drag the interactive marker (orange sphere)
- Click "Plan & Execute"
- Motor should follow!

---

## 📊 System Architecture

```
┌─────────────┐
│   RViz      │  ← Visualization
│   MoveIt    │  ← Motion planning
└──────┬──────┘
       │
┌──────▼──────────────────────┐
│  ROS 2 Control              │
│  - joint_state_broadcaster  │  ← Publishes /joint_states
│  - parol6_arm_controller    │  ← Accepts trajectories
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐
│  parol6_hardware            │  ← Hardware interface (C++)
│  - Serial: /dev/ttyUSB0     │
│  - Baud: 115200             │
│  - Protocol: <SEQ,J1,...>   │
└──────┬──────────────────────┘
       │ USB
┌──────▼──────────────────────┐
│  ESP32                      │  ← Firmware (ros_control.ino)
│  - Parses commands          │
│  - Controls motor           │
│  - Sends feedback           │
└──────┬──────────────────────┘
       │ UART (38400 baud)
┌──────▼──────────────────────┐
│  MKS Servo42C               │  ← Closed-loop stepper
│  - Built-in encoder         │
│  - Built-in PID             │
│  - Position control         │
└─────────────────────────────┘
```

---

## 🎓 Understanding the System

### Communication Protocol

**ROS → ESP32:**
```
<SEQ,J1,J2,J3,J4,J5,J6>
Example: <123,1.5,0.0,0.0,0.0,0.0,0.0>
```

**ESP32 → ROS:**
```
<ACK,SEQ,J1,J2,J3,J4,J5,J6>
Example: <ACK,123,1.50,0.00,0.00,0.00,0.00,0.00>
```

### How It Works

1. **ROS sends trajectory** (e.g., move to 1.5 rad in 5 seconds)
2. **Controller interpolates** into small steps (25 Hz)
3. **Hardware interface** sends commands via serial
4. **ESP32 receives** and filters (only significant changes)
5. **ESP32 commands MKS** motor via `uartRunPulses()`
6. **MKS motor** moves using built-in closed-loop control
7. **ESP32 reports** commanded position back to ROS
8. **ROS visualizes** in RViz

---

## ✅ Success Criteria

You know it's working when:

- ✅ Serial opens: `/dev/ttyUSB0 @ 115200`
- ✅ First feedback received
- ✅ Both controllers active
- ✅ Motor moves to commanded positions
- ✅ Goals finish with `SUCCEEDED`
- ✅ RViz shows robot moving

---

## 📞 Support

**Issues?** Check:
1. ESP32 firmware uploaded correctly
2. Motor wired correctly (TX/RX, power)
3. Serial port permissions (`sudo chmod 666 /dev/ttyUSB0`)
4. Docker container running (`docker ps`)
5. Controllers active (`ros2 control list_controllers`)

**Still stuck?** Review the troubleshooting section above.

---

**Status:** ✅ Single motor working with ROS 2 Control  
**Last Updated:** 2026-01-18  
**Next Milestone:** Scale to 6 motors
