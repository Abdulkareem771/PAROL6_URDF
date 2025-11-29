# PAROL6 Project Handoff - Industrial Edition 🏭

## 🚨 CRITICAL STATUS
**Current State:** WORKING (Industrial Controller Implemented)
**Last Action:** Replaced basic controller with `xbox_industrial_controller.py` to fix lag and stability issues.

## 🎯 Your Mission
You are the Lead Robotics Engineer for the PAROL6 project. Your goal is to ensure this robot is **factory-ready**.

## 🏗️ Architecture: The "Industrial" Pattern

We moved away from simple "callback-based" control to a **Fixed-Rate Control Loop**.

### Why?
1.  **Stability**: Callbacks fire whenever messages arrive (jittery). A fixed loop (10Hz) ensures steady command flow.
2.  **Safety**: We integrate target position independently of current state during motion. This prevents "stuck" joints where the robot refuses to move because it thinks it's at a limit.
3.  **Responsiveness**: We use a "streaming" approach where we update the target continuously but only send goals at a controlled rate.

### Key File: `xbox_industrial_controller.py`
This is the **ONLY** controller you should be using.

**Logic Flow:**
1.  **Init**: Sync `target_joints` with `current_joints` (once).
2.  **Loop (10Hz)**:
    *   Read Joystick (`joy_cmd`).
    *   `target_joints += joy_cmd * sensitivity`.
    *   **Clamp** `target_joints` to URDF limits.
    *   Send `FollowJointTrajectory` goal (async).

## 🛠️ Environment & Tools

### Docker Container (`parol6_dev`)
Everything runs here.
*   **Source**: `source /opt/ros/humble/setup.bash`
*   **Workspace**: `/workspace` (mapped to host `~/Desktop/PAROL6_URDF`)

### Simulation (`Ignition Gazebo`)
*   **Launch**: `./start_ignition.sh`
*   **Check**: Ensure "Controllers loaded" message appears.

### Controller Launch
*   **Command**: `./start_xbox_action.sh`

## 🐛 Troubleshooting Guide (Advanced)

### 1. "Elbow Falls Down" / Gravity Collapse
*   **Cause**: Controller aborted goal and switched to idle (0 effort).
*   **Fix**: The new `xbox_industrial_controller.py` sends continuous goals to "hold" position even when sticks are released.
*   **Check**: `ros2 topic echo /parol6_arm_controller/follow_joint_trajectory/_action/status` - look for ABORTED (4) or REJECTED (5).

### 2. "Base Stuck"
*   **Cause**: Target position variable got synced to a "stuck" physical value.
*   **Fix**: The new controller **does not** sync to `joint_states` while moving. It trusts its internal `target_joints` integrator.

### 3. "Lagging"
*   **Cause**: Flooding the action server with 100Hz+ updates.
*   **Fix**: The new controller runs at a fixed **10Hz** (configurable via `control_rate`).

## 🎛️ Tuning Parameters (In Python Code)

You can adjust these in `xbox_industrial_controller.py`:

```python
self.declare_parameter('sensitivity', 0.05)      # Speed
self.declare_parameter('deadzone', 0.15)         # Stick threshold
self.declare_parameter('control_rate', 10.0)     # Hz
self.declare_parameter('trajectory_duration', 0.2) # Smoothness
```

## 🔮 Future Roadmap (For DeepSeek)

1.  **GUI Integration**:
    *   Create an `rqt` plugin or use `rqt_reconfigure` to adjust sensitivity at runtime.
    *   The current controller uses ROS parameters, so `ros2 param set` works!

2.  **Real Hardware**:
    *   This controller sends `FollowJointTrajectory`. It works for **both** Sim and Real Hardware (if the real robot driver exposes the same action server).

3.  **Inverse Kinematics (IK)**:
    *   Currently we do **Joint Space** control (moving individual joints).
    *   Next step: Implement **Cartesian Control** (moving X, Y, Z) using MoveIt's `Servo` package.

## 📝 Git & Handoff
*   **Branch**: `xbox-controller`
*   **Commit**: Ensure you commit `xbox_industrial_controller.py` and `start_xbox_action.sh`.

---
**You are ready. The factory is waiting.** 🏭
