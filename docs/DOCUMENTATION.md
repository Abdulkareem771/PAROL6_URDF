# PAROL6 Robot - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Architecture Overview](#architecture-overview)
5. [Configuration Files](#configuration-files)
6. [Usage Guide](#usage-guide)
7. [Programming Interface](#programming-interface)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)
10. [API Reference](#api-reference)

---

## 1. Introduction

### About PAROL6

PAROL6 is a 6-DOF (Degrees of Freedom) robotic arm designed for research, education, and light industrial applications. This documentation covers the complete ROS 2 + MoveIt integration for simulation and control.

### Features

- ✅ Full MoveIt 2 integration for motion planning
- ✅ Gazebo simulation support
- ✅ ros2_control framework integration
- ✅ 6-DOF kinematic chain
- ✅ Collision avoidance
- ✅ Multiple motion planning algorithms
- ✅ Python and C++ API support
- ✅ Docker-based development environment

### Robot Specifications

| Property | Value |
|----------|-------|
| **Degrees of Freedom** | 6 |
| **Joint Names** | L1, L2, L3, L4, L5, L6 |
| **Base Link** | base_link |
| **End Effector** | L6 |
| **Max Velocity** | 3.0 rad/s (all joints) |
| **Planning Group** | parol6_arm |

### Joint Limits

| Joint | Type | Min (rad) | Max (rad) | Max Vel (rad/s) |
|-------|------|-----------|-----------|-----------------|
| L1 | Revolute | -1.7 | 1.7 | 3.0 |
| L2 | Revolute | -0.98 | 1.0 | 3.0 |
| L3 | Revolute | -2.0 | 1.3 | 3.0 |
| L4 | Revolute | -2.0 | 2.0 | 3.0 |
| L5 | Revolute | -2.1 | 2.1 | 3.0 |
| L6 | Continuous | - | - | 3.0 |

---

## 2. System Requirements

### Hardware Requirements

- **CPU:** x86_64 processor (Intel/AMD)
- **RAM:** Minimum 4GB, Recommended 8GB+
- **Storage:** 10GB free space
- **GPU:** Optional (for Gazebo visualization)
- **Display:** X11 compatible display server

### Software Requirements

- **OS:** Ubuntu 22.04 LTS (or compatible)
- **Docker:** Version 20.10+
- **X11 Server:** For GUI applications
- **Git:** For version control (optional)

### Pre-installed Docker Image

The `parol6-robot:latest` Docker image includes:
- ROS 2 Humble
- MoveIt 2
- Gazebo Classic
- ros2_control
- All required dependencies

---

## 3. Installation & Setup

### 3.1 Quick Start

```bash
# Navigate to project directory
cd /home/kareem/Desktop/PAROL6_URDF

# Start Docker container
docker run -it --rm \
  --name parol6_dev \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/home/kareem/Desktop/PAROL6_URDF:/workspace" \
  parol6-robot:latest

# Inside container: Build workspace
source /opt/ros/humble/setup.bash
cd /workspace
colcon build --symlink-install
source install/setup.bash

# Launch MoveIt demo
ros2 launch parol6_moveit_config demo.launch.py
```

### 3.2 Using Helper Scripts

```bash
# Interactive launcher
./launch.sh

# Run automated tests
./test_setup.sh

# View quick reference
./QUICKREF.sh
```

### 3.3 Workspace Structure

```
/workspace/
├── PAROL6/                    # Robot description package
│   ├── urdf/                  # URDF files
│   ├── meshes/                # 3D models
│   ├── config/                # Controller configs
│   └── launch/                # Launch files
│
└── parol6_moveit_config/      # MoveIt configuration
    ├── config/                # MoveIt configs
    ├── launch/                # Launch files
    ├── rviz/                  # RViz configs
    └── scripts/               # Python examples
```

---

## 4. Architecture Overview

### 4.1 System Components

```
┌─────────────────────────────────────┐
│         User Interface              │
│  (RViz, Python Scripts, CLI)        │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         MoveIt Framework            │
│  - Motion Planning (OMPL)           │
│  - Kinematics (KDL)                 │
│  - Collision Checking               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      ros2_control Layer             │
│  - Controller Manager               │
│  - Joint Trajectory Controller      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│     Hardware Interface              │
│  - Gazebo (Simulation)              │
│  - Real Hardware (Future)           │
└─────────────────────────────────────┘
```

### 4.2 Data Flow

1. **User Input** → Goal pose/joint positions
2. **MoveIt Planning** → Computes collision-free trajectory
3. **Trajectory Execution** → Sends commands to controller
4. **Controller** → Interpolates and executes motion
5. **Hardware/Simulation** → Moves robot
6. **Feedback** → Joint states published back to system

### 4.3 ROS 2 Communication

**Key Topics:**
- `/joint_states` - Current joint positions/velocities
- `/robot_description` - URDF of the robot
- `/tf`, `/tf_static` - Transform tree
- `/display_planned_path` - Planned trajectory visualization
- `/parol6_arm_controller/joint_trajectory` - Trajectory commands

**Key Services:**
- `/compute_ik` - Inverse kinematics
- `/compute_fk` - Forward kinematics
- `/plan_kinematic_path` - Motion planning

**Key Actions:**
- `/parol6_arm_controller/follow_joint_trajectory` - Execute trajectory

---

## 5. Configuration Files

### 5.1 SRDF (parol6.srdf)

Defines semantic information about the robot:

```xml
<group name="parol6_arm">
  <chain base_link="base_link" tip_link="L6"/>
</group>

<group_state name="home" group="parol6_arm">
  <joint name="L1" value="0"/>
  <!-- ... all joints at 0 -->
</group_state>

<disable_collisions link1="L1" link2="L2" reason="Adjacent"/>
<!-- ... collision matrix -->
```

**Purpose:** Planning groups, named states, collision pairs

### 5.2 Kinematics (kinematics.yaml)

```yaml
parol6_arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
```

**Purpose:** Inverse kinematics solver configuration

### 5.3 OMPL Planning (ompl_planning.yaml)

Configures motion planning algorithms:

- **RRTConnect** - Fast, bidirectional sampling
- **RRT** - Basic rapidly-exploring random tree
- **RRTstar** - Optimal path planning
- **PRM/PRMstar** - Probabilistic roadmap methods
- **KPIECE** - Kinodynamic planning

**Purpose:** Motion planning algorithm selection and tuning

### 5.4 Controllers (moveit_controllers.yaml)

```yaml
moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager

moveit_simple_controller_manager:
  controller_names:
    - parol6_arm_controller
  
  parol6_arm_controller:
    type: FollowJointTrajectory
    joints: [L1, L2, L3, L4, L5, L6]
```

**Purpose:** Links MoveIt to ros2_control controllers

### 5.5 Joint Limits (joint_limits.yaml)

```yaml
joint_limits:
  L1:
    has_velocity_limits: true
    max_velocity: 3.0
    has_acceleration_limits: true
    max_acceleration: 2.0
```

**Purpose:** Velocity and acceleration constraints for planning

---

## 6. Usage Guide

### 6.1 Launching the System

#### Option 1: MoveIt Demo (No Gazebo)

```bash
ros2 launch parol6_moveit_config demo.launch.py
```

**What it does:**
- Starts MoveIt move_group node
- Launches RViz with MoveIt plugin
- Uses fake controllers (no physics)
- Best for: Quick motion planning tests

#### Option 2: Gazebo Simulation

```bash
ros2 launch parol6 gazebo.launch.py
```

**What it does:**
- Starts Gazebo simulator
- Spawns PAROL6 robot
- Loads physics-based controllers
- Best for: Realistic simulation

#### Option 3: Gazebo + MoveIt

```bash
# Terminal 1
ros2 launch parol6 gazebo.launch.py

# Terminal 2 (after Gazebo loads)
ros2 launch parol6 Movit_RViz_launch.py
```

**What it does:**
- Full integration of MoveIt with Gazebo
- Physics-based motion execution
- Best for: Complete system testing

### 6.2 Using RViz

1. **Start the system** (any option above)

2. **In RViz:**
   - Find "MotionPlanning" panel (left side)
   - Select planning group: `parol6_arm`
   - See interactive marker at end effector

3. **Plan a motion:**
   - Drag the orange/blue interactive marker
   - Click "Plan" button
   - Review planned path (ghost robot)
   - Click "Execute" to run motion

4. **Use named states:**
   - Under "Planning" tab
   - "Select Goal State" dropdown
   - Choose "home" or "ready"
   - Click "Plan & Execute"

5. **Adjust settings:**
   - Change planner (RRTConnect recommended)
   - Adjust planning time (default: 5s)
   - Enable/disable collision checking

### 6.3 Command Line Control

#### Check System Status

```bash
# List controllers
ros2 control list_controllers

# Expected output:
# parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
# joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active

# Monitor joint states
ros2 topic echo /joint_states

# List all topics
ros2 topic list
```

#### Send Manual Commands

```bash
# Move to specific joint positions
ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{
    joint_names: [L1, L2, L3, L4, L5, L6],
    points: [
      {
        positions: [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],
        time_from_start: {sec: 2}
      }
    ]
  }"
```

---

## 7. Programming Interface

### 7.1 Python API (MoveIt Commander)

```python
#!/usr/bin/env python3
import rclpy
from moveit.planning import MoveItPy

# Initialize
rclpy.init()
moveit = MoveItPy(node_name="parol6_controller")
arm = moveit.get_planning_component("parol6_arm")

# Move to named state
arm.set_goal_state(configuration_name="home")
plan = arm.plan()
if plan:
    moveit.execute(plan.trajectory, controllers=[])

# Move to joint positions
joint_positions = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
robot_state = RobotState(moveit.get_robot_model())
for i, pos in enumerate(joint_positions):
    robot_state.set_joint_positions(f"L{i+1}", [pos])
arm.set_goal_state(robot_state=robot_state)
plan = arm.plan()
if plan:
    moveit.execute(plan.trajectory, controllers=[])
```

### 7.2 Example Script

Run the included example:

```bash
cd /workspace/parol6_moveit_config/scripts
python3 example_controller.py
```

Features:
- Move to named states
- Custom joint positions
- Pose-based goals
- Interactive menu

### 7.3 C++ API

```cpp
#include <moveit/move_group_interface/move_group_interface.h>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("parol6_controller");
  
  moveit::planning_interface::MoveGroupInterface arm(node, "parol6_arm");
  
  // Move to named state
  arm.setNamedTarget("home");
  arm.move();
  
  // Move to joint positions
  std::vector<double> joint_positions = {0.0, -0.5, 0.5, 0.0, 0.0, 0.0};
  arm.setJointValueTarget(joint_positions);
  arm.move();
  
  rclcpp::shutdown();
  return 0;
}
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Package Not Found

**Problem:** `Package 'parol6' not found`

**Solution:**
```bash
source /workspace/install/setup.bash
```

#### Controllers Not Loading

**Problem:** Controllers fail to start

**Solution:**
```bash
# Check controller manager
ros2 control list_controllers

# Manually load controller
ros2 control load_controller parol6_arm_controller
ros2 control set_controller_state parol6_arm_controller active
```

#### Planning Fails

**Problem:** "No motion plan found"

**Solutions:**
1. Increase planning time in RViz
2. Try different planner (RRTConnect usually best)
3. Check if goal is reachable
4. Verify no collision at start/goal
5. Check joint limits

#### Gazebo Crashes

**Problem:** Gazebo window freezes or crashes

**Solutions:**
```bash
# Set environment variable
export QT_X11_NO_MITSHM=1

# Restart with verbose output
ros2 launch parol6 gazebo.launch.py --ros-args --log-level debug
```

#### No IK Solution

**Problem:** "No IK solution found"

**Solutions:**
1. Goal pose may be out of reach
2. Try different orientation
3. Check joint limits
4. Increase IK timeout in kinematics.yaml

### 8.2 Debugging Tools

```bash
# Check TF tree
ros2 run tf2_tools view_frames
# Creates frames.pdf with transform tree

# Monitor planning scene
ros2 topic echo /monitored_planning_scene

# Check node graph
ros2 run rqt_graph rqt_graph

# View logs
ros2 run rqt_console rqt_console
```

### 8.3 Performance Optimization

**Slow Planning:**
- Reduce `kinematics_solver_search_resolution` (less accurate, faster)
- Use simpler planners (RRT instead of RRTstar)
- Disable unnecessary collision checking

**High CPU Usage:**
- Reduce controller update rate in ros2_controllers.yaml
- Simplify collision meshes
- Reduce Gazebo physics update rate

---

## 9. Advanced Topics

### 9.1 Adding Custom Named States

Edit `parol6_moveit_config/config/parol6.srdf`:

```xml
<group_state name="custom_pose" group="parol6_arm">
  <joint name="L1" value="0.5"/>
  <joint name="L2" value="-0.8"/>
  <joint name="L3" value="1.0"/>
  <joint name="L4" value="0.0"/>
  <joint name="L5" value="0.5"/>
  <joint name="L6" value="0.0"/>
</group_state>
```

Rebuild and use:
```bash
colcon build --symlink-install
source install/setup.bash
```

### 9.2 Tuning Motion Planners

Edit `parol6_moveit_config/config/ompl_planning.yaml`:

```yaml
RRTConnect:
  type: geometric::RRTConnect
  range: 0.0  # 0 = auto, or set max step size
  
parol6_arm:
  longest_valid_segment_fraction: 0.005  # Smaller = more accurate
```

### 9.3 Adding Collision Objects

```python
from moveit.planning import PlanningSceneInterface

scene = PlanningSceneInterface(node)

# Add box obstacle
box_pose = PoseStamped()
box_pose.header.frame_id = "world"
box_pose.pose.position.x = 0.3
box_pose.pose.position.z = 0.2
scene.add_box("obstacle", box_pose, size=(0.1, 0.1, 0.1))
```

### 9.4 Custom Constraints

```python
from moveit_msgs.msg import Constraints, OrientationConstraint

# Keep end effector upright
constraints = Constraints()
orientation_constraint = OrientationConstraint()
orientation_constraint.link_name = "L6"
orientation_constraint.orientation.w = 1.0
orientation_constraint.absolute_x_axis_tolerance = 0.1
orientation_constraint.absolute_y_axis_tolerance = 0.1
orientation_constraint.absolute_z_axis_tolerance = 3.14
constraints.orientation_constraints.append(orientation_constraint)

arm.set_path_constraints(constraints)
```

### 9.5 Real Hardware Integration

To use with real hardware:

1. **Create hardware interface plugin:**
```cpp
#include <hardware_interface/system_interface.hpp>

class PAROL6Hardware : public hardware_interface::SystemInterface {
  // Implement read(), write(), etc.
};
```

2. **Update URDF:**
```xml
<ros2_control name="RealSystem" type="system">
  <hardware>
    <plugin>parol6_hardware/PAROL6Hardware</plugin>
  </hardware>
  <!-- joints... -->
</ros2_control>
```

3. **Build and test:**
```bash
colcon build
ros2 launch parol6 real_robot.launch.py
```

---

## 10. API Reference

### 10.1 Launch Files

| File | Description | Usage |
|------|-------------|-------|
| `demo.launch.py` | Full MoveIt demo | `ros2 launch parol6_moveit_config demo.launch.py` |
| `move_group.launch.py` | MoveIt planning only | `ros2 launch parol6_moveit_config move_group.launch.py` |
| `gazebo.launch.py` | Gazebo simulation | `ros2 launch parol6 gazebo.launch.py` |
| `Movit_RViz_launch.py` | MoveIt + RViz | `ros2 launch parol6 Movit_RViz_launch.py` |

### 10.2 ROS 2 Parameters

**MoveIt Parameters:**
- `planning_time` - Max time for planning (default: 5.0s)
- `planning_attempts` - Number of planning attempts (default: 10)
- `max_velocity_scaling_factor` - Scale max velocity (0.0-1.0)
- `max_acceleration_scaling_factor` - Scale max acceleration (0.0-1.0)

**Controller Parameters:**
- `update_rate` - Controller update frequency (Hz)
- `state_publish_rate` - State publishing rate (Hz)
- `action_monitor_rate` - Action monitoring rate (Hz)

### 10.3 Helper Scripts

| Script | Purpose |
|--------|---------|
| `launch.sh` | Interactive menu launcher |
| `test_setup.sh` | Automated system tests |
| `QUICKREF.sh` | Display quick reference |
| `example_controller.py` | Python API example |

---

## Appendix A: File Reference

### Configuration Files
- `parol6.srdf` - Semantic robot description
- `kinematics.yaml` - IK solver config
- `ompl_planning.yaml` - Motion planners
- `joint_limits.yaml` - Joint constraints
- `moveit_controllers.yaml` - Controller manager
- `ros2_controllers.yaml` - ros2_control config

### Documentation Files
- `README.md` - Quick start guide
- `DOCUMENTATION.md` - This file
- `SETUP_COMPLETE.md` - Setup summary
- `ARCHITECTURE.md` - System architecture
- `QUICKREF.sh` - Command reference

---

## Appendix B: Resources

### Official Documentation
- **MoveIt 2:** https://moveit.picknik.ai/humble/
- **ros2_control:** https://control.ros.org/humble/
- **ROS 2 Humble:** https://docs.ros.org/en/humble/
- **Gazebo:** https://gazebosim.org/

### Tutorials
- MoveIt Tutorials: https://moveit.picknik.ai/humble/doc/tutorials/tutorials.html
- ros2_control Demos: https://github.com/ros-controls/ros2_control_demos

### Community
- ROS Discourse: https://discourse.ros.org/
- MoveIt Discord: https://discord.gg/moveit

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-27  
**Author:** AntiGravity AI  
**License:** BSD
