# PAROL6 Developer Guide

**For colleagues working with the real robot hardware**

---

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- PAROL6 workspace cloned: `/path/to/PAROL6_URDF/`
- ESP32 connected via USB (for real robot) OR `socat` virtual ports (for testing)

### Launch Real Robot
```bash
cd /path/to/PAROL6_URDF/
./bringup.sh real
```

**What happens:**
1. Docker container `parol6_dev` starts
2. Workspace builds (`parol6`, `parol6_driver`, `parol6_moveit_config`)
3. RViz opens with MoveIt interface
4. Driver connects to ESP32 (or virtual serial)

**Result**: You can plan and execute trajectories in RViz.

---

## 📝 Writing Code to Control the Robot

### Option 1: Python Script (Recommended for Beginners)

Create `my_robot_program.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
    def move_to_pose(self, x, y, z, roll, pitch, yaw):
        """Move robot end-effector to specified pose"""
        goal_msg = MoveGroup.Goal()
        # ... (configure goal)
        self._action_client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    node = RobotController()
    node.move_to_pose(0.3, 0.0, 0.4, 0, 0, 0)  # Example
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

**Run it:**
```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
python3 my_robot_program.py
```

---

### Option 2: MoveIt Python API (Advanced)

For complex motion planning:

```python
#!/usr/bin/env python3
import rclpy
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

def plan_and_execute():
    rclpy.init()
    
    # Initialize MoveIt
    moveit = MoveItPy(node_name="moveit_py_demo")
    arm = moveit.get_planning_component("parol6_arm")
    
    # Set target joint positions
    joint_goals = {
        'joint1': 0.0,
        'joint2': -0.5,
        'joint3': 0.5,
        'joint4': 0.0,
        'joint5': 0.5,
        'joint6': 0.0
    }
    arm.set_goal_state(configuration_name="home")  # Or use joint_goals
    
    # Plan
    plan_result = arm.plan()
    
    if plan_result:
        # Execute
        robot_trajectory = plan_result.trajectory
        moveit.execute(robot_trajectory, controllers=[])
        print("Motion executed successfully!")
    else:
        print("Planning failed!")

if __name__ == '__main__':
    plan_and_execute()
```

---

### Option 3: ROS 2 Service Call (Quick Commands)

For one-off commands:

```bash
# Inside container
docker exec -it parol6_dev bash
source /workspace/install/setup.bash

# Call MoveIt service to plan to named state
ros2 service call /plan_kinematic_path moveit_msgs/srv/GetMotionPlan "{
  motion_plan_request: {
    group_name: 'parol6_arm',
    goal_constraints: [{
      name: 'home'
    }]
  }
}"
```

---

## 🏗️ Code Structure

```
PAROL6_URDF/
├── parol6/                          # Robot description (URDF, meshes)
│   ├── urdf/PAROL6.urdf            # Robot model
│   ├── meshes/                     # STL files
│   └── launch/
│       └── ignition.launch.py      # Gazebo simulation
│
├── parol6_moveit_config/           # MoveIt configuration
│   ├── config/
│   │   ├── parol6.srdf            # Semantic robot description
│   │   ├── moveit_controllers.yaml # Controller config
│   │   └── kinematics.yaml        # IK solver config
│   └── rviz/moveit.rviz           # RViz layout
│
├── parol6_driver/                  # Real robot driver
│   ├── parol6_driver/
│   │   ├── real_robot_driver.py   # Main driver node
│   │   └── virtual_esp32.py       # Test firmware simulation
│   └── launch/
│       └── unified_bringup.launch.py  # Unified launcher
│
└── PAROL6/firmware/                # ESP32 firmware (Arduino)
    └── firmware.ino               # Microcontroller code
```

---

## 🔧 Troubleshooting

### Issue: Robot not visible in RViz

**Symptoms**: RViz opens but no robot model appears

**Solutions**:
1. Check Fixed Frame: `Global Options → Fixed Frame` should be `base_link` or `world`
2. Add RobotModel display: `Add → RobotModel`
3. Verify joint states:
   ```bash
   ros2 topic echo /joint_states
   ```
   Should show continuous updates

---

### Issue: "No controller available"

**Symptoms**: Planning works but execution fails

**Cause**: Driver not connected or action server not running

**Solution**:
```bash
# Check if driver is running
ros2 node list | grep real_robot_driver

# Check action server
ros2 action list | grep follow_joint_trajectory

# Restart driver if needed
docker exec -it parol6_dev bash -c "
  source /workspace/install/setup.bash && \
  ros2 run parol6_driver real_robot_driver
"
```

---

### Issue: Serial port not found

**Symptoms**: `[ERROR] Could not open serial port`

**Solutions**:

1. **Check port exists**:
   ```bash
   ls -la /dev/ttyACM*
   ls -la /dev/pts/*
   ```

2. **Give permission**:
   ```bash
   sudo chmod 666 /dev/ttyACM0
   ```

3. **Use virtual serial for testing**:
   ```bash
   # Terminal 1: Create virtual ports
   socat -d -d pty,raw,echo=0 pty,raw,echo=0
   # Note the pts numbers (e.g., /dev/pts/8 and /dev/pts/9)
   
   # Terminal 2: Run virtual ESP32
   python3 parol6_driver/parol6_driver/virtual_esp32.py /dev/pts/9
   
   # Terminal 3: Driver will auto-detect /dev/pts/8
   ./bringup.sh real
   ```

---

### Issue: Build fails

**Symptoms**: `colcon build` errors

**Common fixes**:
```bash
# Clean build
docker exec -it parol6_dev bash
cd /workspace
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select parol6 parol6_driver parol6_moveit_config

# If permissions error:
exit  # Exit container
sudo chown -R $USER:$USER .
```

---

## 🎯 Common Tasks

### Task: Move to Named Position

**Pre-defined positions** are in `parol6_moveit_config/config/parol6.srdf`:
- `home`: Default safe position
- `ready`: Ready for picking

```python
from moveit.planning import MoveItPy

moveit = MoveItPy(node_name="demo")
arm = moveit.get_planning_component("parol6_arm")

arm.set_goal_state(configuration_name="home")
plan = arm.plan()
moveit.execute(plan.trajectory, controllers=[])
```

---

### Task: Move to Cartesian Position

```python
from geometry_msgs.msg import PoseStamped

pose_goal = PoseStamped()
pose_goal.header.frame_id = "base_link"
pose_goal.pose.position.x = 0.3
pose_goal.pose.position.y = 0.0
pose_goal.pose.position.z = 0.4
pose_goal.pose.orientation.w = 1.0  # No rotation

arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="L6")
plan = arm.plan()
```

---

### Task: Execute Cartesian Path (Straight Line)

For welding/tracking applications:

```python
waypoints = []

# Start pose
wpose = arm.get_current_pose().pose

# Move down 10cm
wpose.position.z -= 0.1
waypoints.append(copy.deepcopy(wpose))

# Move right 5cm
wpose.position.y += 0.05
waypoints.append(copy.deepcopy(wpose))

# Compute Cartesian path
(plan, fraction) = arm.compute_cartesian_path(
    waypoints,
    0.01,  # 1cm step size
    0.0    # No jump threshold
)

if fraction > 0.95:  # Successfully planned 95%+
    moveit.execute(plan, controllers=[])
```

---

## 🤖 AI Assistant Prompts (For Future Development)

### Prompt 1: Add Vision Integration
```
I have the PAROL6 robot with MoveIt2 setup. I want to add object detection using a Kinect v2 camera. 
The camera driver is already installed (libfreenect2, kinect2_ros2).

Tasks:
1. Create a ROS2 node that subscribes to /kinect2/sd/image_color_rect and /kinect2/sd/points
2. Use YOLOv8 (ultralytics package) to detect objects in the RGB stream
3. For detected objects, find their 3D position from the point cloud
4. Publish object poses as visualization_msgs/MarkerArray
5. Provide code to make the robot move to pick up the detected object

Project structure is in /workspace/. Add files to parol6_vision/ package if needed.
```

---

### Prompt 2: Add Path Smoothing
```
I need to smooth a planned trajectory using B-splines before executing it on the robot.

Context:
- Using MoveIt2 with parol6_arm planning group
- Have scipy installed
- Want to reduce jerk for smooth welding motions

Tasks:
1. Take a MoveIt trajectory (list of JointTrajectoryPoints)
2. Fit a B-spline curve through the waypoints
3. Resample at higher frequency (e.g., 100 Hz)
4. Return smooth trajectory
5. Integrate into the execute pipeline

Provide Python code compatible with ROS 2 Humble.
```

---

### Prompt 3: Add Safety Limits
```
Add safety features to prevent robot from hitting workspace boundaries or moving too fast.

Requirements:
- Check if planned trajectory enters forbidden zones (define as YAML config)
- Limit maximum joint velocities to 50% of current limits
- Add emergency stop button monitoring (subscribe to /emergency_stop topic)
- If emergency stop pressed, halt all motion immediately

Modify real_robot_driver.py to include these checks.
```

---

## 📚 Additional Resources

- **MoveIt2 Tutorials**: https://moveit.picknik.ai/main/doc/tutorials/tutorials.html
- **ROS 2 Humble Docs**: https://docs.ros.org/en/humble/
- **PAROL6 Hardware Docs**: See `PROJECT_ARCHITECTURE_REPORT.md`
- **Integration Guide**: See `REAL_ROBOT_INTEGRATION.md`

---

## 🆘 Getting Help

1. **Check logs**:
   ```bash
   docker logs parol6_dev --tail 100
   ```

2. **List active nodes**:
   ```bash
   docker exec -it parol6_dev bash
   source /workspace/install/setup.bash
   ros2 node list
   ```

3. **Inspect topics**:
   ```bash
   ros2 topic list
   ros2 topic echo /joint_states
   ```

4. **Contact**: kareem@example.com (Update with actual contact)

---

**Good luck building!** 🚀
