# PAROL6 Project Handoff - Industrial Edition 🏭

## 🚨 CRITICAL STATUS
**Current State:** WORKING (Industrial Controller with Slew Rate Limiter)
**Last Action:** Implemented Slew Rate Limiter to fix "Home Button Crash".

## 🎯 Your Mission
You are the Lead Robotics Engineer for the PAROL6 project. Your goal is to ensure this robot is **factory-ready**.

## 🏗️ Architecture: The "Industrial" Pattern

We use a **Fixed-Rate Control Loop** with **Slew Rate Limiting**.

### Why?
1.  **Stability**: Callbacks fire whenever messages arrive (jittery). A fixed loop (20Hz) ensures steady command flow.
2.  **Safety**: We integrate target position independently of current state during motion.
3.  **Smoothness**: The **Slew Rate Limiter** ensures that even if the user requests a sudden jump (like pressing "Home"), the robot smoothly ramps to that position at `max_speed`.

### Key File: `xbox_industrial_controller.py`
This is the **ONLY** controller you should be using.

**Logic Flow:**
1.  **Init**: Sync `commanded_joints` with `current_joints` (once).
2.  **Loop (20Hz)**:
    *   **Input**: Update `commanded_joints` based on Joystick (velocity) or Buttons (step change).
    *   **Clamp**: Ensure `commanded_joints` are within URDF limits.
    *   **Slew Limit**: Move `trajectory_joints` towards `commanded_joints` by at most `max_speed * dt`.
    *   **Output**: Send `trajectory_joints` as a `FollowJointTrajectory` goal (async).

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
*   **Fix**: The new controller sends continuous goals to "hold" position.

### 2. "Stuck after Home"
*   **Cause**: Step input caused velocity limit violation.
*   **Fix**: Slew Rate Limiter ramps the value smoothly.

### 3. "Lagging"
*   **Cause**: Flooding the action server.
*   **Fix**: Fixed rate loop (20Hz).

## 🎛️ Tuning Parameters (In Python Code)

You can adjust these in `xbox_industrial_controller.py` or via ROS params:

```python
self.declare_parameter('sensitivity', 0.05)      # Joystick Speed
self.declare_parameter('deadzone', 0.15)         # Stick threshold
self.declare_parameter('max_speed', 0.8)         # Rad/s (Global limit)
self.declare_parameter('control_rate', 20.0)     # Hz
self.declare_parameter('trajectory_lookahead', 0.1) # Seconds
```

## 🔮 Future Roadmap (For DeepSeek)

1.  **MoveIt Servo Integration (The "Right Way")**:
    *   **Goal**: Use `moveit_servo` for collision avoidance and singularity handling.
    *   **Blocker**: `ros-humble-moveit-servo` is NOT installed in the container.
    *   **Action**: You need to update the Dockerfile or install it manually if permissions allow.

2.  **GUI Integration**:
    *   Use `rqt` > Dynamic Reconfigure to tune parameters live.

3.  **Real Hardware**:
    *   This controller sends `FollowJointTrajectory`. It works for **both** Sim and Real Hardware.

## 📝 Git & Handoff
*   **Branch**: `xbox-controller`
*   **Commit**: Ensure you commit `xbox_industrial_controller.py` and `start_xbox_action.sh`.

---
**You are ready. The factory is waiting.** 🏭
