# RViz Setup & Troubleshooting Guide

**Common RViz issues and how to fix them**

---

## Issue 1: Robot Not Visible in RViz

### Symptoms
- RViz opens but shows empty scene
- Only grid visible, no robot

### Solution

**Method 1: Check RobotModel Display**
1. Left panel → "Displays"
2. Look for "MotionPlanning" → Expand it
3. Find "Robot Description" section
4. Check: "Show Robot Visual" is enabled
5. Set "Robot Alpha" to 1.0

**Method 2: Reset Camera**
1. Top menu → Views → Reset View  
2. Or scroll down in left panel to "Views"
3. Set "Distance" to 3-4 meters
4. Set "Yaw" to 0
5. Click "Zero" button

**Method 3: Check FixedFrame**
1. Top of left panel: "Fixed Frame"
2. Should be set to `world` or `base_link`
3. Try switching between them

---

## Issue 2: Interactive Markers Not Visible (The "3 Lines")

### Symptoms
- Robot is visible
- Can't drag end-effector
- No orange/blue sphere with arrows

### Debug: Are They Running?

Interactive markers ARE running if you see these topics:
```bash
docker exec -it parol6_real bash
ros2 topic list | grep marker
```

Should show:
```
/rviz_.../robot_interaction_interactive_marker_topic/update
/rviz_.../robot_interaction_interactive_marker_topic/feedback
```

**If you see these topics, markers exist but are just invisible!**

### Solution: Camera Position

**The markers are probably there, just out of view!**

**Fix 1: Reset Camera to Robot**
1. In RViz bottom panel, find "MotionPlanning"
2. Click **"Reset"** or **"Home"** button  
3. Camera should center on robot
4. The interactive marker sphere should now be visible at end-effector!

**Fix 2: Manual Camera Adjustment**
1. Hold middle mouse button and drag to pan camera
2. Scroll to zoom in
3. Look for orange/blue sphere at robot's gripper
4. It might be hidden behind robot - rotate view with right mouse button

**Fix 3: Increase Marker Size**
1. Left panel → Expand "MotionPlanning"
2. Find "Planning Request" section  
3. Set "Interactive Marker Size" to `0.3` (or higher)
4. Markers become more visible

### Solution: Enable Query Goal State

1. Left panel → "MotionPlanning" → "Planning Request"
2. Enable: **"Query Goal State"** ✓
3. Disable: "Query Start State" (to avoid confusion)
4. Check: **"Allow External Comm."** ✓ (if available)

---

## Issue 3: Can Execute But No Visual Feedback

### Symptoms
- Can plan and execute
- ESP32 receives commands
- Robot in RViz doesn't move

### Solution

This is **normal for real robot mode!** In real mode:
- RViz shows **planning result** (ghost trajectory)
- Real robot (ESP32) executes actual motion
- RViz model doesn't update dynamically

**This is correct behavior** - you're testing the hardware pipeline, not simulation.

---

### Method 3: Restart with Query Markers

If above doesn't work, the config might need updating:

**Stop current session (Ctrl+C)**

**Edit RViz config:**
```bash
# The file is at: parol6_moveit_config/rviz/moveit.rviz
# Look for "MoveIt_Goal_Tolerance" and ensure it's > 0
# Or just delete the config and let RViz recreate defaults
```

**Or use demo.launch.py** which definitely has markers:
```bash
docker exec -it parol6_real bash
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch real robot driver separately
ros2 run parol6_driver real_robot_driver &

# Launch demo (has markers enabled by default)
ros2 launch parol6_moveit_config demo.launch.py
```

---

## What You Should See

Once enabled, you'll see:
- **Orange/blue sphere** at the end-effector (gripper)
- **3 colored arrows**:
  - Red arrow = X axis (move left/right)
  - Green arrow = Y axis (move forward/back)
  - Blue arrow = Z axis (move up/down)
- **3 colored rings**:
  - For rotation around each axis

**To move:**
- **Click and drag arrows** → Translate (move)
- **Click and drag rings** → Rotate
- Click **"Plan"** → MoveIt calculates path
- Click **"Execute"** → Sends to ESP32!

---

## Troubleshooting

### Markers appear but are tiny
- Increase "Interactive Marker Size" in Planning Request section

### Markers don't move the goal
- Make sure "Query Goal State" is enabled
- Make sure "Query Start State" is disabled

### Can't grab the markers
- They might be behind the robot - rotate camera
- Try zooming in closer

---

**Your system is working! ESP32 is receiving commands. Once you enable markers, you can manually test the full pipeline!**
