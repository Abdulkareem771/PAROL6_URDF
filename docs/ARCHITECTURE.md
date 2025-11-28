# PAROL6 System Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PAROL6 ROBOT SYSTEM                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
├─────────────────────────────────────────────────────────────────────┤
│  RViz2                    │  Python Scripts  │  Command Line         │
│  - MotionPlanning Plugin  │  - MoveIt API    │  - ros2 topic/service │
│  - Interactive Markers    │  - Custom Logic  │  - Manual Control     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MOVEIT FRAMEWORK                              │
├─────────────────────────────────────────────────────────────────────┤
│  move_group Node                                                     │
│  ├── Motion Planning (OMPL)                                         │
│  │   ├── RRTConnect, RRT, RRTstar                                  │
│  │   ├── PRM, PRMstar                                              │
│  │   └── KPIECE, BKPIECE, etc.                                     │
│  ├── Kinematics (KDL)                                               │
│  │   ├── Forward Kinematics                                        │
│  │   └── Inverse Kinematics                                        │
│  ├── Collision Checking                                             │
│  │   └── Self-collision & Scene collision                          │
│  └── Trajectory Processing                                          │
│      └── Time parameterization                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CONTROLLER LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  Controller Manager (ros2_control)                                   │
│  ├── parol6_arm_controller                                          │
│  │   └── JointTrajectoryController                                 │
│  │       └── FollowJointTrajectory Action                          │
│  └── joint_state_broadcaster                                        │
│      └── Publishes /joint_states                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      HARDWARE INTERFACE                              │
├─────────────────────────────────────────────────────────────────────┤
│  ros2_control Hardware Interface                                     │
│  ├── Simulation: GazeboSystem                                       │
│  │   └── Gazebo Physics Engine                                     │
│  └── Real Robot: [Your Hardware Interface]                         │
│      └── Motor Controllers                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ROBOT HARDWARE                               │
├─────────────────────────────────────────────────────────────────────┤
│  PAROL6 6-DOF Robot Arm                                             │
│  ├── Joint L1 (Base rotation)        -1.7 to 1.7 rad               │
│  ├── Joint L2 (Shoulder)              -0.98 to 1.0 rad             │
│  ├── Joint L3 (Elbow)                 -2.0 to 1.3 rad              │
│  ├── Joint L4 (Wrist pitch)           -2.0 to 2.0 rad              │
│  ├── Joint L5 (Wrist roll)            -2.1 to 2.1 rad              │
│  └── Joint L6 (End effector rotation) continuous                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Planning Request (Goal Pose/Joints)
        │
        ▼
┌───────────────────┐
│   MoveIt          │
│   - Plan path     │──────┐
│   - Check collisions     │
│   - Optimize trajectory  │
└───────────────────┘      │
        │                   │
        ▼                   │
  Trajectory               │
  (Joint positions         │
   over time)              │
        │                   │
        ▼                   ▼
┌───────────────────────────────┐
│  Controller Manager           │
│  - Interpolate trajectory     │
│  - Send position commands     │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Hardware Interface           │
│  - Convert to motor commands  │
│  - Read joint states          │
└───────────────────────────────┘
        │
        ▼
   Robot Moves!
```

## Configuration Files Map

```
parol6_moveit_config/config/
│
├── parol6.srdf                  ← Defines planning groups, collision pairs
│   └── Used by: MoveIt planning
│
├── kinematics.yaml              ← IK solver configuration
│   └── Used by: MoveIt IK calculations
│
├── ompl_planning.yaml           ← Motion planning algorithms
│   └── Used by: MoveIt path planning
│
├── joint_limits.yaml            ← Velocity/acceleration limits
│   └── Used by: MoveIt trajectory generation
│
├── moveit_controllers.yaml      ← Controller manager config
│   └── Used by: MoveIt execution
│
└── ros2_controllers.yaml        ← Controller parameters
    └── Used by: ros2_control

PAROL6/urdf/PAROL6.urdf          ← Robot structure, physics, ros2_control
    └── Used by: Everything (robot description)

PAROL6/config/ros2_controllers.yaml  ← Gazebo controller config
    └── Used by: Gazebo simulation
```

## ROS 2 Topics & Services

### Key Topics
```
/joint_states                    ← Current joint positions/velocities
/robot_description               ← URDF of the robot
/tf, /tf_static                  ← Transform tree
/display_planned_path            ← Visualization of planned trajectory
/execute_trajectory/_action/*    ← Execute planned motion
/parol6_arm_controller/joint_trajectory  ← Direct trajectory commands
```

### Key Services
```
/compute_ik                      ← Inverse kinematics
/compute_fk                      ← Forward kinematics
/get_planning_scene              ← Current collision scene
/plan_kinematic_path             ← Plan a motion
```

### Key Actions
```
/move_action                     ← High-level move command
/execute_trajectory              ← Execute a trajectory
/parol6_arm_controller/follow_joint_trajectory  ← Low-level control
```

## Launch File Hierarchy

```
Option 1: MoveIt Demo (Standalone)
└── ros2 launch parol6_moveit_config demo.launch.py
    ├── Starts move_group node
    ├── Starts RViz2 with MoveIt plugin
    ├── Starts robot_state_publisher
    ├── Starts ros2_control with FakeSystem
    └── Loads controllers

Option 2: Gazebo Only
└── ros2 launch parol6 gazebo.launch.py
    ├── Starts Gazebo simulator
    ├── Spawns robot model
    ├── Starts robot_state_publisher
    └── Loads Gazebo controllers

Option 3: Gazebo + MoveIt
├── Terminal 1: ros2 launch parol6 gazebo.launch.py
│   └── (Same as Option 2)
└── Terminal 2: ros2 launch parol6 Movit_RViz_launch.py
    └── Includes demo.launch.py
        └── (Same as Option 1, but connects to Gazebo)
```

## Development Workflow

```
1. Design Phase
   ├── Create/modify URDF (robot structure)
   ├── Generate SRDF (planning groups)
   └── Configure kinematics solver

2. Configuration Phase
   ├── Set up motion planners (OMPL)
   ├── Configure controllers (ros2_control)
   └── Define joint limits

3. Testing Phase
   ├── Test in RViz (visualization only)
   ├── Test in Gazebo (physics simulation)
   └── Verify controller communication

4. Deployment Phase
   ├── Integrate real hardware interface
   ├── Calibrate robot
   └── Test on physical robot

5. Application Phase
   ├── Write Python/C++ control scripts
   ├── Implement task logic
   └── Deploy application
```

## Quick Reference Commands

```bash
# Build workspace
colcon build --symlink-install

# Source workspace
source install/setup.bash

# Launch MoveIt demo
ros2 launch parol6_moveit_config demo.launch.py

# Launch Gazebo
ros2 launch parol6 gazebo.launch.py

# List controllers
ros2 control list_controllers

# Check joint states
ros2 topic echo /joint_states

# View TF tree
ros2 run tf2_tools view_frames

# Test trajectory
ros2 topic pub --once /parol6_arm_controller/joint_trajectory ...
```

---

**This architecture enables:**
- ✅ Motion planning with collision avoidance
- ✅ Inverse kinematics for pose goals
- ✅ Trajectory optimization
- ✅ Simulation before deployment
- ✅ Easy transition to real hardware
- ✅ Programmatic control via Python/C++
