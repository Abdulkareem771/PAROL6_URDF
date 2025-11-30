# PAROL6 ROS 2 Project Explanation

Welcome! This document is designed to explain the PAROL6 ROS 2 project from the bottom up. It is written for beginners who want to understand how all the pieces fit together to control a robot arm using ROS 2.

## 🏗️ 1. The Foundation: Docker Environment

Before we touch any robot code, we need a computer environment to run it. We use **Docker** for this. Think of Docker as a "virtual computer" that lives inside your laptop. It ensures that everyone (you and your colleagues) has the exact same software versions, so "it works on my machine" is always true.

### 📄 File: `Dockerfile`
This file is the "recipe" for building that virtual computer.

```dockerfile
# 1. Base Image: Start with a version of Linux (Ubuntu) that has ROS 2 Humble pre-installed.
FROM osrf/ros:humble-desktop

# 2. Install Dependencies: Add extra tools we need, like Gazebo (simulator) and MoveIt (motion planning).
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \       # Connects ROS 2 to Gazebo
    ros-humble-gazebo-ros2-control \   # Allows ROS 2 to control Gazebo motors
    ros-humble-moveit \                # Motion planning library
    ros-humble-rviz2 \                 # 3D Visualization tool
    gazebo \                           # The simulator itself
    && rm -rf /var/lib/apt/lists/*     # Clean up to keep image small

# 3. Workspace: Create a folder where our code will live inside the container.
WORKDIR /workspace

# 4. Auto-Source: Make sure ROS 2 commands are available every time we open a terminal.
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 5. Default Command: When the container starts, give us a bash terminal.
CMD ["/bin/bash"]
```

---

## 🤖 2. The Body: Robot Description (URDF)

Now that we have an OS, we need to tell ROS what our robot looks like. We use **URDF** (Unified Robot Description Format), which is an XML file.

### 📄 File: `PAROL6/urdf/PAROL6.urdf`
This file defines the **Links** (solid parts) and **Joints** (moving hinges) of the robot.

#### A. Defining a Link (A solid part)
```xml
<link name="base_link">
  <!-- 1. Inertial: Mass and center of gravity (needed for physics simulation) -->
  <inertial>
    <mass value="0.812" />
    <inertia ixx="0.001" ... /> 
  </inertial>

  <!-- 2. Visual: What it looks like (3D mesh) -->
  <visual>
    <geometry>
      <mesh filename="package://parol6/meshes/base_link.STL" />
    </geometry>
  </visual>

  <!-- 3. Collision: The physical shape for bumping into things (often same as visual) -->
  <collision>
    <geometry>
      <mesh filename="package://parol6/meshes/base_link.STL" />
    </geometry>
  </collision>
</link>
```

#### B. Defining a Joint (A hinge)
```xml
<joint name="joint_L1" type="revolute">
  <!-- 1. Parent and Child: Connects base_link to Link 1 -->
  <parent link="base_link" />
  <child link="L1" />

  <!-- 2. Axis: Rotates around the Z axis (0 0 1) -->
  <axis xyz="0 0 1" />

  <!-- 3. Limits: Can rotate from -1.7 to +1.7 radians -->
  <limit lower="-1.7" upper="1.7" effort="300" velocity="3" />
</joint>
```

#### C. The Brain Interface: `ros2_control`
This special section tells ROS how to talk to the motors (or simulated motors).

```xml
<ros2_control name="IgnitionSystem" type="system">
  <hardware>
    <!-- Use the Ignition Gazebo plugin to simulate hardware -->
    <plugin>ign_ros2_control/IgnitionSystem</plugin>
  </hardware>
  
  <joint name="joint_L1">
    <!-- We can send POSITION commands to this joint -->
    <command_interface name="position">
      <param name="min">-1.7</param>
      <param name="max">1.7</param>
    </command_interface>
    <!-- We can read the current POSITION from this joint -->
    <state_interface name="position"/>
  </joint>
  ...
</ros2_control>
```

---

## 🚀 3. The Launcher: Starting the World

We have a computer (Docker) and a robot (URDF). Now we need to start the simulation.

### 📄 File: `start_ignition.sh`
This is a Bash script (like a batch file) that automates the complex startup process.

```bash
#!/bin/bash

# 1. Enable Graphics: Allow the Docker container to show windows on your screen.
xhost +local:docker > /dev/null 2>&1

# 2. Start Docker: Run the image we built, mounting our code folder (/workspace).
docker run -d --rm \
  --name parol6_dev \
  --network host \
  --privileged \
  -v /home/kareem/Desktop/PAROL6_URDF:/workspace \
  parol6-ultimate:latest \
  tail -f /dev/null

# 3. Build Code: Compile the ROS packages inside the container.
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build"

# 4. Launch ROS: Run the main launch file.
docker exec -it parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  ros2 launch parol6 ignition.launch.py
"
```

---

## 🎮 4. The Brain: Controlling the Robot

Finally, we want to move the robot with an Xbox controller. This is where our Python script comes in.

### 📄 File: `xbox_direct_control.py`
This script acts as a "bridge". It listens to the Xbox controller and sends commands to the robot.

#### A. Setup and Connections
```python
class XboxDirectControl(Node):
    def __init__(self):
        super().__init__('xbox_direct_control')
        
        # 1. Action Client: The way we send commands to the robot controller.
        # We send a "trajectory" (a path of movement).
        self._action_client = ActionClient(self, FollowJointTrajectory, '/parol6_arm_controller/follow_joint_trajectory')
        
        # 2. Subscriber: Listen to the Xbox controller inputs.
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        
        # 3. Subscriber: Listen to where the robot currently IS (Joint States).
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
```

#### B. The Logic Loop (`joy_callback`)
This function runs every time you move a joystick.

```python
    def joy_callback(self, msg):
        # 1. Map Joystick Axes to Robot Joints
        # msg.axes[0] is the Left Stick Horizontal
        # self.target_positions[0] is Joint 1
        self.target_positions[0] -= msg.axes[0] * self.speed_scale

        # 2. Safety Limits: Don't let the robot hit itself (Software Limit)
        self.target_positions[0] = max(-3.14, min(3.14, self.target_positions[0]))
        
        # 3. Send the Command
        self.send_goal()
```

#### C. Sending the Command (`send_goal`)
```python
    def send_goal(self):
        goal_msg = FollowJointTrajectory.Goal()
        
        # Create a trajectory point (where we want to be 0.2 seconds from now)
        point = JointTrajectoryPoint()
        point.positions = self.target_positions
        point.time_from_start = Duration(sec=0, nanosec=200000000) # 200ms
        
        # Send it!
        goal_msg.trajectory.points.append(point)
        self._action_client.send_goal_async(goal_msg)
```

---

## 🔗 How It All Connects

1.  **You** push the Xbox Joystick.
2.  **Linux** sees the input (`/dev/input/js0`).
3.  **`joy_node`** (ROS Driver) reads Linux input and publishes a `Joy` message to the topic `/joy`.
4.  **`xbox_direct_control.py`** receives the `/joy` message.
    *   It calculates new joint angles.
    *   It sends a `FollowJointTrajectory` action goal to `/parol6_arm_controller`.
5.  **`ros2_control`** (running in Gazebo) receives the goal.
6.  **Gazebo** simulates the motors moving to those angles.
7.  **You** see the robot move on screen!

## 📚 Next Steps for Learning

1.  **Modify the Speed**: Change `self.speed_scale` in `xbox_direct_control.py` to make it faster or slower.
2.  **Change Mappings**: Swap which joystick axis controls which joint.
3.  **Add a Button**: Make the "A" button (index 0 in `msg.buttons`) reset the robot to the home position (all zeros).

Happy Coding! 🚀
