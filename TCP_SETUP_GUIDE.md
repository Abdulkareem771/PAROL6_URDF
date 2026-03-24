# PAROL6 Tool Center Point (TCP) Setup Guide

This guide explains the changes made to correctly define a Tool Center Point (TCP) at the tip of our robot gripper, and how you can manually adjust this point if you change your gripper in the future.

## Why Do We Need a TCP?
By default, the PAROL6 URDF ends at the wrist flange (`L6` in the URDF, or `link6` internally). When planning trajectories (straight lines, cartesian paths), MoveIt tracks the `tip_link` of the `parol6_arm` planning chain. Without explicitly defining a dummy tool frame, MoveIt mathematically paths the physical wrist to the setpoint, rather than the tip of the gripper, resulting in confusing translations.

We fixed this by appending a blank, invisible coordinate frame (`tcp_link`) extended out from the physical wrist, and pointing MoveIt towards it.

## Where Are the Changes Made?
These changes were originally applied on **March 25, 2026** and involve three files. If you ever need to apply these changes to a fresh branch, you will need to cleanly reproduce the edits below:

1. **`PAROL6/urdf/PAROL6.urdf`**: (The core robot URDF generated from CAD)
2. **`parol6_hardware/urdf/parol6.urdf.xacro`**: (The software-in-the-loop mock URDF used in simulation)
3. **`parol6_moveit_config/config/parol6.srdf`**: (The semantic definitions used by MoveIt 2 to track endpoints and check self-collisions)

### 1. Adding `tcp_link` to the URDF
In both `PAROL6.urdf` and `parol6.urdf.xacro`, we inserted an empty link definition named `tcp_link` connected rigidly to the wrist with a parameter-based translation offset (`xyz`).

*(For `parol6.urdf.xacro`, substituting `L6` for `link6`)*
```xml
  <link name="tcp_link"/>
  <joint name="tcp_joint" type="fixed">
    <origin xyz="0 0 0.1" rpy="0 0 0" />
    <parent link="L6"/>
    <child link="tcp_link"/>
  </joint>
```

### 2. Updating MoveIt Group Chains
Inside of `parol6_moveit_config/config/parol6.srdf`, we updated the robot kinematic group named `parol6_arm` to end its calculations at `tcp_link` instead of `L6`:

```xml
    <group name="parol6_arm">
        <chain base_link="base_link" tip_link="tcp_link"/>
    </group>
```

We also bound the `<end_effector>` logical unit (named `hand`) to originate from the new tool frame:

```xml
    <end_effector name="hand" parent_link="tcp_link" group="parol6_arm"/>
```

---

## How to Adjust TCP Coordinates For a New Gripper

Whenever you swap physical grippers, or if tracking is inaccurate because the gripper depth is misjudged, you must manually edit the `<origin>` offset. 

**Steps for fine-tuning the TCP Point:**
1. Open up `/PAROL6/urdf/PAROL6.urdf` (and `/parol6_hardware/urdf/parol6.urdf.xacro`).
2. Navigate near the bottom of the file (ctrl+f for `tcp_joint` or `tcp_link`).
3. Locate the origin translation attributes:
   `<origin xyz="X Y Z" rpy="R P Y" />`

### Translations (`xyz`)
In `xyz="0 0 0.1"`, the dimensions are in **meters** relative to the rotating geometry of `L6` (the joint 6 wrist flange).
* **Z-axis:** Usually pointing straight **out** perpendicularly away from the wrist. If your new gripper extends `15.5cm`, you would change `Z` to `0.155`.
* **X and Y axes:** If your effector attaches asymmetrically, is offset horizontally, or curves drastically sideways, adjust X/Y accordingly (e.g. `xyz="0 0.05 0.1"` if sitting 5cm off-center to one side).

### Rotations (`rpy`)
Determines the pitch and yaw attitude of the tracking frame. `rpy` represents Roll, Pitch, and Yaw in **radians**.
* Currently it evaluates to `0 0 0`, meaning it faces exactly outwards aligned perfectly parallel with the `L6` face.
* A 90-degree twist offset (e.g., if you orient the camera 90° sideways) can be encoded by shifting the corresponding axis by `1.5708` or `-1.5708` (+/- pi/2 radians).

### Recompiling & Verify
Once the numbers have been tweaked and files saved:
1. Since we modified purely coordinate text logic (not C++ code) you only need to rebuild the workspace variables if your environment requires it: `colcon build --symlink-install`.
2. Re-source your terminal: `source install/setup.bash`.
3. Re-launch `ros2 launch parol6_moveit_config demo.launch.py`.
4. Command a linear point-to-point path in standard coordinate testing via code, or manipulate the sphere arrows manually in RViz and ensure they hover correctly at the edge of the new gripper.
