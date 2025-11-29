# PAROL6 Project Handoff - Industrial Edition 🏭

## 🚨 CRITICAL STATUS
**Current State:** WORKING (Industrial Controller with Slew Rate Limiter)
**Next Goal:** Implement **MoveIt Servo** (The "Right Way")

## 🎯 Your Mission
You are the Lead Robotics Engineer for the PAROL6 project.
The user wants to switch from the custom Python controller to **MoveIt Servo** for professional-grade control (collision avoidance, singularities, etc.).

## 🏗️ Architecture: The "Industrial" Pattern (Current)
We currently use `xbox_industrial_controller.py` which mimics industrial safety features:
*   **Fixed-Rate Loop (20Hz)**
*   **Slew Rate Limiter** (Smooth ramping)
*   **Independent Target Integration**

## 🚀 The "Right Way": MoveIt Servo (Next Steps)

I have prepared the configuration files, but the package is missing.

### 1. Files Created
*   `parol6_moveit_config/config/parol6_servo.yaml`: Configuration for Servo.
*   `parol6_moveit_config/launch/servo.launch.py`: Launch file.

### 2. The Blocker
The Docker container `parol6_dev` does **NOT** have `ros-humble-moveit-servo` installed.
`apt-get install` failed due to permission/source issues.

### 3. Your Tasks
1.  **Install MoveIt Servo**:
    *   Try to fix `apt-get` in the container.
    *   OR build `moveit_servo` from source in the workspace.
    *   OR ask the user to update the Docker image.
2.  **Launch Servo**:
    ```bash
    ros2 launch parol6_moveit_config servo.launch.py
    ```
3.  **Connect Xbox**:
    *   You will need a node that publishes `Twist` messages to `~/delta_twist_cmds` or `Joint` messages to `~/delta_joint_cmds`.
    *   The current `xbox_industrial_controller.py` can be adapted to publish these instead of Action Goals.

## 🛠️ Environment & Tools

### Docker Container (`parol6_dev`)
*   **Source**: `source /opt/ros/humble/setup.bash`
*   **Workspace**: `/workspace`

### Simulation (`Ignition Gazebo`)
*   **Launch**: `./start_ignition.sh`

### Current Controller
*   **Command**: `./start_xbox_action.sh` (Uses `xbox_industrial_controller.py`)

## 🐛 Troubleshooting Guide (Current Controller)

### 1. "Elbow Falls Down"
*   **Fix**: The new controller sends continuous goals to "hold" position.

### 2. "Stuck after Home"
*   **Fix**: Slew Rate Limiter ramps the value smoothly.

## 📝 Git & Handoff
*   **Branch**: `xbox-controller`
*   **Commit**: Ensure you commit the new MoveIt config files.

---
**You are ready. The factory is waiting.** 🏭
