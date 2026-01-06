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

## 🎯 Project Context: Vision-Guided Welding/Gluing

**Thesis Goal**: Automated welding/gluing system that:
1. **Vision**: Captures workspace image → Detects workpiece → Identifies ROI (seam/edge)
2. **Planning**: Generates smooth welding path from ROI → Applies B-spline smoothing
3. **Execution**: Follows path precisely with mkservo42c closed-loop servos

**Hardware**:
- **Servos**: mkservo42c (closed-loop FOC steppers)
- **Camera**: Kinect v2 (RGB + Depth)
- **End-Effector**: Welding torch / glue dispenser

**Key Requirements**:
- Smooth Cartesian paths (straight lines, no jerky motion)
- Precise trajectory following (closed-loop feedback)
- Acceleration profiles (smooth start/stop for quality)

**This guide helps colleagues**:
- Control the robot via MoveIt API
- Understand the driver <-> servo communication
- Extend the system for welding/vision applications

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

## 🔌 Understanding the Driver Architecture

### When to Use MoveIt API vs. Direct Driver Interaction

**Use MoveIt API** (Recommended 95% of the time):
- Planning trajectories
- Collision avoidance
- Inverse kinematics
- High-level robot control

**Modify the Driver** (Advanced use cases):
- Changing serial communication protocol
- Adding custom hardware interfaces (sensors, grippers)
- Implementing low-level safety checks
- Debugging communication issues

### Driver Node Anatomy

**File**: `parol6_driver/parol6_driver/real_robot_driver.py`

**What it does**:
1. **Connects to ESP32** via serial port (`/dev/ttyACM0` or `/dev/pts/8`)
2. **Implements Action Server** (`/parol6_arm_controller/follow_joint_trajectory`)
3. **Receives trajectories** from MoveIt
4. **Sends commands** to microcontroller in format: `<J1,J2,J3,J4,J5,J6>`
5. **Publishes feedback** as `/joint_states`

### Communication Flow

```
┌──────────────┐
│   MoveIt     │  Plans trajectory
│  (move_group)│
└──────┬───────┘
       │
       │ Action Goal: FollowJointTrajectory
       │ (list of waypoints with timestamps)
       ▼
┌───────────────────────┐
│  real_robot_driver.py │  Executes trajectory
└───────────────────────┘
       │
       │ Serial: <J1,J2,J3,J4,J5,J6>
       ▼
┌───────────────────────┐
│   ESP32 Firmware      │  Moves motors
└───────────────────────┘
```

### Communication Protocol

**To ESP32** (Commands):
```
<0.0,0.5,-0.3,0.0,0.2,0.0>\n
```
Format: `<joint1,joint2,joint3,joint4,joint5,joint6>\n`

**From ESP32** (Acknowledgment):
```
READY
MOVING
DONE
```

---

## 🛠️ Modifying the Driver

### Example 1: Add Gripper Control

```python
# In real_robot_driver.py

from std_srvs.srv import SetBool

class RealRobotDriver(Node):
    def __init__(self):
        # ... existing code ...
        
        # Add gripper service
        self.gripper_service = self.create_service(
            SetBool,
            'gripper/set_state',
            self.gripper_callback
        )
    
    def gripper_callback(self, request, response):
        """Open/close gripper"""
        if request.data:  # True = close
            self.ser.write(b"<GRIP_CLOSE>\n")
        else:  # False = open
            self.ser.write(b"<GRIP_OPEN>\n")
        
        response.success = True
        return response
```

**Usage**:
```bash
ros2 service call /gripper/set_state std_srvs/srv/SetBool "{data: true}"
```

**ESP32 Firmware** (add this):
```cpp
void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    
    if (cmd == "<GRIP_CLOSE>") {
      digitalWrite(GRIPPER_PIN, HIGH);
    }
    else if (cmd == "<GRIP_OPEN>") {
      digitalWrite(GRIPPER_PIN, LOW);
    }
  }
}
```

---

### Example 2: Add Force Sensor Reading

```python
from geometry_msgs.msg import WrenchStamped
import threading

class RealRobotDriver(Node):
    def __init__(self):
        # ... existing code ...
        
        # Add force sensor publisher
        self.force_pub = self.create_publisher(
            WrenchStamped, '/force_sensor', 10
        )
        
        # Start sensor reading thread
        self.sensor_thread = threading.Thread(
            target=self.read_sensor_loop, daemon=True
        )
        self.sensor_thread.start()
    
    def read_sensor_loop(self):
        """Continuously read force sensor from Arduino"""
        while rclpy.ok():
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode().strip()
                
                if line.startswith("FORCE:"):
                    # Parse: "FORCE:12.5"
                    force_z = float(line.split(":")[1])
                    
                    msg = WrenchStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.wrench.force.z = force_z
                    self.force_pub.publish(msg)
            
            time.sleep(0.01)  # 100Hz
```

**ESP32 Firmware**:
```cpp
void loop() {
  float force = analogRead(A0) * 0.0488;  // Convert to Newtons
  Serial.print("FORCE:");
  Serial.println(force);
  delay(10);
}
```

---

### Example 3: Add Safety Limit Checking

```python
def execute_callback(self, goal_handle):
    trajectory = goal_handle.request.trajectory
    
    # Check if trajectory is safe
    for point in trajectory.points:
        if not self.is_safe_position(point.positions):
            self.get_logger().error("Unsafe trajectory detected!")
            goal_handle.abort()
            return FollowJointTrajectory.Result()
    
    # ... rest of execution ...

def is_safe_position(self, joint_positions):
    """Check if position is within safety bounds"""
    safety_limits = {
        0: (-3.14, 3.14),   # joint1
        1: (-1.57, 1.57),   # joint2
        2: (-1.57, 1.57),   # joint3
        3: (-3.14, 3.14),   # joint4
        4: (-1.57, 1.57),   # joint5
        5: (-3.14, 3.14),   # joint6
    }
    
    for i, pos in enumerate(joint_positions):
        min_val, max_val = safety_limits[i]
        
        if not (min_val <= pos <= max_val):
            self.get_logger().warn(f"Joint {i+1} out of bounds: {pos}")
            return False
    
    return True
```

---

## 📊 Debugging the Driver

### Test driver directly

```python
# test_driver.py
import rclpy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

def send_test_command():
    rclpy.init()
    node = rclpy.create_node('test_driver')
    
    client = ActionClient(
        node, 
        FollowJointTrajectory,
        '/parol6_arm_controller/follow_joint_trajectory'
    )
    
    goal = FollowJointTrajectory.Goal()
    
    # Single waypoint
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point.time_from_start.sec = 2
    
    goal.trajectory.points = [point]
    goal.trajectory.joint_names = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
    ]
    
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    
    print(f"Goal accepted: {future.result().accepted}")

if __name__ == '__main__':
    send_test_command()
```

### Monitor Communication

```bash
# Check if driver is running
ros2 node list | grep real_robot_driver

# Check action server
ros2 action list
ros2 action info /parol6_arm_controller/follow_joint_trajectory

# Monitor joint states
ros2 topic echo /joint_states

# See trajectory commands
ros2 topic echo /joint_trajectory
```

---

## 🎯 Recommendation

**For most tasks**: Use the MoveIt Python API (Options 1 & 2 shown earlier)

**Only modify the driver when**:
- Adding new hardware (sensors, grippers, tool changers)
- Changing communication protocol
- Implementing robot-specific safety features
- Debugging low-level communication issues

**MoveIt handles**:
- Path planning & optimization
- Collision detection  
- Inverse kinematics
- Trajectory smoothing

**Driver handles**:
- Serial communication
- Hardware interface
- Real-time control loop
- Joint state feedback

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
