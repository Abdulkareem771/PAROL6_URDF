# PAROL6 Xbox Controller Integration - DeepSeek Handoff

## 🎯 Mission
You are taking over the PAROL6 robotic arm project. Your task is to maintain and improve the Xbox controller integration for real-time robot control in Gazebo simulation.

## 📁 Project Structure & Environment

### Working Directory
```
/home/kareem/Desktop/PAROL6_URDF/
```

This is the main workspace. All commands should be run from here unless specified otherwise.

### Docker Environment
**CRITICAL**: All ROS 2 commands MUST run inside the Docker container `parol6_dev`.

```bash
# To run ROS 2 commands:
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && <your_command>"

# To enter the container interactively:
docker exec -it parol6_dev bash
```

**Container details:**
- Image: `parol6-ultimate:latest`
- Container name: `parol6_dev`
- ROS 2 Distribution: Humble
- Workspace inside container: `/workspace` (mapped to `/home/kareem/Desktop/PAROL6_URDF`)

### Key Files & Directories

```
PAROL6_URDF/
├── xbox_action_controller.py          # MAIN CONTROLLER - Use this!
├── start_xbox_action.sh               # Startup script
├── start_ignition.sh                  # Launch Gazebo simulation
├── old_xbox_files/                    # Archive of old/broken versions
│   ├── xbox_trajectory_controller.py  # OLD - don't use
│   ├── xbox_controller_node.py        # OLD - don't use
│   └── test_movement.py               # OLD test script
├── PAROL6/                            # Robot description package
│   ├── urdf/                          # URDF files
│   ├── config/ros2_controllers.yaml   # Controller configuration
│   └── launch/                        # Launch files
└── parol6_moveit_config/              # MoveIt configuration
```

## 🤖 Robot Configuration

### Joint Names (CRITICAL - Must match exactly)
```python
joint_names = [
    'joint_L1',  # Base rotation (revolute)
    'joint_L2',  # Shoulder (revolute)
    'joint_L3',  # Elbow (revolute)
    'joint_L4',  # Wrist pitch (revolute)
    'joint_L5',  # Wrist roll (revolute)
    'joint_L6'   # Gripper (fixed/prismatic)
]
```

### Joint Limits (from URDF)
```python
joint_limits = {
    'joint_L1': (-3.05, 3.05),    # ±175°
    'joint_L2': (-1.91, 1.91),    # ±110°
    'joint_L3': (-2.53, 2.53),    # ±145°
    'joint_L4': (-2.70, 2.70),    # ±155°
    'joint_L5': (-6.28, 6.28),    # ±360° (continuous)
    'joint_L6': (0.0, 0.0)        # Fixed for now
}
```

### Controller Interface (CRITICAL)
The robot uses **ROS 2 actions**, NOT simple topics!

```python
# ✅ CORRECT - Action interface
ActionClient(self, FollowJointTrajectory, 
             '/parol6_arm_controller/follow_joint_trajectory')

# ❌ WRONG - Topic interface (doesn't work!)
Publisher(JointTrajectory, 
          '/parol6_arm_controller/joint_trajectory', 10)
```

**Why actions?**
- Provides feedback (accepted, executing, succeeded)
- Can be preempted/cancelled
- Proper trajectory execution with timing

## 🎮 Xbox Controller Setup

### Hardware
- Xbox 360 USB controller
- Device appears as `/dev/input/js0`

### ROS 2 Joy Node
```bash
# Launch joy node (inside container)
ros2 run joy joy_node
```

### Xbox Button/Axis Mapping
```python
# Axes (msg.axes[index])
axes[0]  # Left stick X (left/right)
axes[1]  # Left stick Y (up/down)
axes[2]  # Left trigger (LT): -1=released, 1=pressed
axes[3]  # Right stick X (left/right)
axes[4]  # Right stick Y (up/down)
axes[5]  # Right trigger (RT): -1=released, 1=pressed

# Buttons (msg.buttons[index])
buttons[0]  # A button
buttons[1]  # B button
buttons[2]  # X button
buttons[3]  # Y button
buttons[4]  # LB (left bumper)
buttons[5]  # RB (right bumper)
```

## 🚀 How to Launch Everything

### Step 1: Start Simulation
```bash
./start_ignition.sh
```
This launches Ignition Gazebo with the PAROL6 robot and starts the controllers.

**Wait for this message:** "Controllers loaded and started successfully"

### Step 2: Start Xbox Controller
```bash
./start_xbox_action.sh
```
This launches:
1. Joy node (reads Xbox controller)
2. Xbox action controller (sends commands to robot)

## 🔧 Common Development Tasks

### Check if Controller is Running
```bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 node list | grep xbox"
```
Expected output: `/xbox_action_controller`

### Monitor Joint States
```bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /joint_states --field position"
```
You should see 6 values changing as you move the controller.

### Check Action Server
```bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 action list"
```
Should show: `/parol6_arm_controller/follow_joint_trajectory`

### Debug Joy Messages
```bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /joy"
```
Move sticks and press buttons to see values.

### Kill Stuck Processes
```bash
docker exec parol6_dev bash -c "pkill -f xbox_action_controller"
docker exec parol6_dev bash -c "pkill -f joy_node"
```

## 🐛 Known Issues & Solutions

### Issue 1: Robot Moves Slowly/Laggily
**Cause:** `time_from_start` in trajectory point is too long
**Solution:** Reduce to 50ms (0.05 seconds) for real-time control
```python
point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms
```

### Issue 2: Controller Stops Responding
**Cause:** Goals being rejected or queued up
**Solution:** 
- Use `send_goal_async()` (non-blocking)
- Add goal response callback to check acceptance
- Only send goals when movement detected

### Issue 3: Joints Hit Limits and Get Stuck
**Cause:** Not clamping to joint limits
**Solution:** Always clamp target positions:
```python
def clamp_to_limits(self, joint_idx, value):
    joint_name = self.joint_names[joint_idx]
    min_val, max_val = self.joint_limits[joint_name]
    return max(min_val, min(max_val, value))
```

### Issue 4: Robot Drifts from Target
**Cause:** Position accumulation without feedback
**Solution:** Subscribe to `/joint_states` and update current position regularly

### Issue 5: Simulation Time Mismatch
**Cause:** Node not using simulation time
**Solution:**
```python
super().__init__('node_name',
    parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', 
                                 rclpy.Parameter.Type.BOOL, True)
    ])
```

## 📊 Performance Expectations

### Good Performance
- Immediate response to stick movement (<100ms)
- Smooth motion without jittering
- All 6 joints controllable independently
- Home/Reset buttons work instantly

### Signs of Problems
- Delay >500ms between input and movement
- Jerky/stuttering motion
- Joints not moving at all
- Goals being rejected (check logs)

## 🔍 Debugging Workflow

When something doesn't work:

1. **Check container is running:**
   ```bash
   docker ps | grep parol6_dev
   ```

2. **Check ROS 2 nodes:**
   ```bash
   docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 node list"
   ```

3. **Check topics:**
   ```bash
   docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 topic list"
   ```

4. **Check controller is loaded:**
   ```bash
   docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 control list_controllers"
   ```
   Should show `parol6_arm_controller` as **active**

5. **Check action server:**
   ```bash
   docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && ros2 action list"
   ```

6. **Read logs:**
   - Joy node terminal: Look for axis/button values
   - Controller terminal: Look for error messages
   - Container logs: `docker logs parol6_dev`

## 💡 Code Architecture

### Main Controller (`xbox_action_controller.py`)

```python
class XboxActionController(Node):
    def __init__(self):
        # Create action client for robot control
        # Subscribe to /joy for Xbox input
        # Subscribe to /joint_states for robot state
        
    def state_callback(self, msg):
        # Update current joint positions from robot
        
    def joy_callback(self, msg):
        # Read Xbox controller input
        # Calculate target positions
        # Send goals to action server
        
    def send_goal(self):
        # Create FollowJointTrajectory goal
        # Send asynchronously to action server
```

### Key Parameters to Tune

```python
deadzone = 0.15          # Ignore small stick movements
sensitivity = 0.08       # How fast joints move (radians/command)
max_speed = 0.5          # Maximum speed limit
command_duration = 0.05  # How long trajectory takes (50ms)
```

## 📝 Git Workflow

Current branch: `xbox-controller`

```bash
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "fix: improve controller responsiveness"

# Push to remote
git push origin xbox-controller
```

## 🎯 Your Responsibilities

1. **Maintain working Xbox controller**
2. **Improve responsiveness and user experience**
3. **Add new features** (gripper control, speed modes, etc.)
4. **Document changes** in markdown files
5. **Keep code clean** and well-commented

## 🚨 Critical Rules

1. ✅ **ALWAYS** run ROS 2 commands inside Docker container
2. ✅ **ALWAYS** use action interface, never topic publishing
3. ✅ **ALWAYS** set `use_sim_time=True` for nodes
4. ✅ **ALWAYS** clamp to joint limits
5. ✅ **ALWAYS** test in simulation before deploying
6. ❌ **NEVER** delete old files, move to `old_xbox_files/`
7. ❌ **NEVER** edit URDF without understanding impacts
8. ❌ **NEVER** commit log files or `__pycache__`

## 📚 Useful ROS 2 Commands Reference

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo topic data
ros2 topic echo /topic_name

# List controllers
ros2 control list_controllers

# List actions
ros2 action list

# Send action goal (for testing)
ros2 action send_goal /action_name action_type "goal_data"

# Kill node
ros2 node kill /node_name
```

## 🎓 Learning Resources

- ROS 2 Actions: https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html
- ros2_control: https://control.ros.org/humble/index.html
- Gazebo Integration: https://gazebosim.org/docs

---

**You are now the maintainer of this project. Good luck!** 🚀

## Quick Start Checklist

- [ ] Read this entire document
- [ ] Launch simulation: `./start_ignition.sh`
- [ ] Launch controller: `./start_xbox_action.sh`
- [ ] Move Xbox sticks and verify robot responds
- [ ] Check logs for errors
- [ ] Test all 6 joints individually
- [ ] Test A/B buttons for reset/home
- [ ] Make your first improvement!
