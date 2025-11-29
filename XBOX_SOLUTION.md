# Xbox Controller Integration - INDUSTRIAL EDITION 🏭

## 🎉 Status: ROBUST & FACTORY READY

The PAROL6 robot now runs on a **fixed-rate industrial controller** (`xbox_industrial_controller.py`).

## 🚀 Quick Start

### 1. Launch Simulation
```bash
./start_ignition.sh
```

### 2. Launch Controller
```bash
./start_xbox_action.sh
```

## 🎮 Controls (Standard)

| Control | Action |
|---------|--------|
| **Left Stick** | Base & Shoulder |
| **Right Stick** | Elbow & Wrist Pitch |
| **Triggers** | Wrist Roll |
| **A Button** | Reset to Zero |
| **B Button** | Home Position |

## ⚙️ Tuning & Customization (New!)

You can now adjust settings **while the robot is running** using ROS 2 parameters!

### Method 1: Command Line
```bash
# Change speed (default: 0.05)
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 param set /xbox_controller sensitivity 0.1"

# Change deadzone (default: 0.15)
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 param set /xbox_controller deadzone 0.05"
```

### Method 2: GUI (rqt)
1. Open `rqt` in your container (if GUI is enabled) or connect from host.
2. Go to **Plugins > Configuration > Dynamic Reconfigure** (or Parameter Reconfigure).
3. Select `/xbox_controller`.
4. Adjust sliders!

## 🔧 Technical Improvements

### Why is this "Industrial Grade"?

1.  **Fixed-Rate Loop (10Hz)**:
    *   Old: Sent commands whenever joystick moved (jittery, flood-prone).
    *   New: Sends commands exactly every 100ms. Smooth & predictable.

2.  **Independent Target Integration**:
    *   Old: Read current position → added delta → sent goal. If robot was stuck, target got stuck.
    *   New: Maintains internal "Target State". If robot gets stuck physically, the target can still move away, pulling the robot free.

3.  **Safety Clamping**:
    *   Strictly enforces URDF joint limits before sending commands.

4.  **Parameter Server**:
    *   All constants (speed, deadzone, rate) are now ROS 2 parameters.

## 📁 File Structure

*   `xbox_industrial_controller.py`: **The Main Brain**.
*   `start_xbox_action.sh`: Startup script.
*   `xbox_teleop.launch.py`: ROS 2 launch file (for future use).

---
**Maintained by**: DeepSeek (Handoff Complete)
