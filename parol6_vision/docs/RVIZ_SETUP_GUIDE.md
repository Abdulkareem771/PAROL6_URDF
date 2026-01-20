# RViz Camera Visualization Guide

**For Teammates: Setting up and Testing the Vision System**

This guide walks you through launching RViz to visualize the PAROL6 robot with the Kinect camera positioned next to it.

---

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ Docker container running (`./start_container.sh`)
- ✅ Workspace built (`colcon build --symlink-install`)
- ✅ Kinect v2 camera (optional for initial visualization, required for camera feed)

---

## 🚀 Quick Start

### Step 1: Enter Docker Container and Build

```bash
docker exec -it parol6_dev bash
cd /workspace
```

### Step 2: Build the Workspace

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

**Important:** The camera visualization requires these packages to be built:
- `parol6` - Contains robot URDF
- `parol6_moveit_config` - Contains MoveIt configuration and SRDF
- `parol6_msgs` - Custom message definitions
- `parol6_vision` - Vision pipeline and launch files

**If you only want to build vision-related packages:**
```bash
colcon build --packages-up-to parol6_vision --symlink-install
```

This automatically builds `parol6_vision` and all its dependencies.

**Note:** First build may take 2-5 minutes.

### Step 3: Launch Camera Visualization

```bash
ros2 launch parol6_vision camera_setup.launch.py
```

**Expected Behavior:**
- RViz window opens
- Robot model appears in the center
- Camera frame (colored axes) visible next to the robot
- Orange interactive marker (ball) for motion planning

---

## 🔍 What You Should See in RViz

### 1. **Displays Panel** (Left Side)

You should see these displays automatically configured:

- ✅ **Grid**: Reference grid on the ground plane
- ✅ **MotionPlanning**: Shows the robot model and interactive controls
- ✅ **TF**: Displays coordinate frames (robot + camera)
- ✅ **Camera**: Panel for Kinect video feed (shows black until camera is connected)

### 2. **3D View** (Center)

- **Robot Model**: PAROL6 arm in default (zero) position
- **World Frame**: Origin at robot base
- **Camera Frame**: Colored axes at `(x=0.5, y=0.0, z=1.0)` relative to robot base

### 3. **Interactive Marker (Orange Ball)**

- Should appear automatically at the end-effector
- Drag it to move the robot to new positions
- Click "Plan" then "Execute" in the MotionPlanning panel to move

---

## 📷 Testing with Real Kinect Camera

### Step 1: Connect Kinect Hardware

1. Plug Kinect v2 into USB 3.0 port
2. Ensure power adapter is connected

### Step 2: Launch Kinect Driver

**In a NEW terminal:**

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge.launch.py
```

### Step 3: Verify Camera Feed in RViz

- The **Camera** panel (bottom right in RViz) should now show live video
- Image topic: `/kinect2/qhd/image_color_rect`

### Step 4: Check Camera Frame Alignment

In RViz, enable the **TF** display and verify:
- `kinect2_rgb_optical_frame` appears at the expected position (0.5m to the side, 1m up)
- Camera is "looking" toward the robot workspace

---

## 🎯 Testing Vision Pipeline Components

Once the camera is working, you can test individual vision nodes:

### Test 1: Red Line Detector

**What it does:** Detects red tape/markers in the camera view

```bash
# In a new terminal
ros2 run parol6_vision red_line_detector
```

**Verify:**
```bash
ros2 topic echo /vision/weld_lines_2d
```

### Test 2: Full Vision Pipeline

```bash
ros2 launch parol6_vision test_integration.launch.py
```

**Check these topics are publishing:**
- `/vision/weld_lines_2d` - Detected red lines (2D)
- `/vision/weld_lines_3d` - 3D projected lines
- `/vision/welding_path` - Generated trajectory

---

## 🔧 Troubleshooting


### Issue: Package Not Found Error

**Error message:**
```
PackageNotFoundError: "package 'parol6_moveit_config' not found"
```

**Cause:** You only built `parol6_vision` but the launch file requires the full workspace.

**Solution:**
```bash
cd /workspace
source /opt/ros/humble/setup.bash
colcon build --packages-up-to parol6_vision --symlink-install
source install/setup.bash
```

This builds `parol6_vision` and all required dependencies including:
- `parol6` (URDF)
- `parol6_moveit_config` (MoveIt configs)
- `parol6_msgs` (message definitions)

### Issue: Robot Model Not Visible

**Solution 1: Check Joint States**
```bash
ros2 topic echo /joint_states
```
Should show positions for `joint_L1` through `joint_L6`.

**Solution 2: Manually Add MotionPlanning Display**
1. Click "Add" in RViz Displays panel
2. Select "MotionPlanning" 
3. Set Planning Group to `parol6_arm`

### Issue: Camera Panel Shows Black Screen

**Check if Kinect is publishing:**
```bash
ros2 topic list | grep kinect
ros2 topic hz /kinect2/qhd/image_color_rect
```

**Expected output:** ~15-30 Hz

**If no topics:**
- Verify USB 3.0 connection (blue port)
- Check Kinect power LED
- Restart `kinect2_bridge` launch

### Issue: No Interactive Marker (Orange Ball)

**Verify MoveGroup is running:**
```bash
ros2 topic list | grep move_group
```

**Should see:**
- `/move_group/display_planned_path`
- `/move_group/status`

**If missing:**
- The `move_group` node may have failed to start
- Check terminal output for errors

### Issue: "Permission denied" on Serial Port

```bash
sudo usermod -a -G dialout $USER
# Then logout and login
```

### Issue: RViz Crashes on Startup

**X11 Display Error:**
```bash
# On HOST machine (not Docker):
xhost +local:docker
```

---

## 📊 Expected System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   RViz Visualization                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐│
│  │ Robot Model  │  │ Camera View  │  │ TF Frames ││
│  └──────────────┘  └──────────────┘  └───────────┘│
└─────────────────────────────────────────────────────┘
                          ▲
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│ Joint State  │  │   Kinect    │  │   Static    │
│  Publisher   │  │   Driver    │  │  TF (Camera)│
└──────────────┘  └─────────────┘  └─────────────┘
```

---

## 📚 Next Steps

After verifying RViz visualization:

1. **Calibrate Camera**: Follow [CAMERA_CALIBRATION_GUIDE.md](../../docs/CAMERA_CALIBRATION_GUIDE.md)
2. **Test Detection**: Run unit tests - [TESTING_GUIDE.md](TESTING_GUIDE.md)
3. **Run Full Pipeline**: Launch complete vision system

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review terminal output for error messages
3. Verify all prerequisites are met
4. Check [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed test procedures

**Common Log Files:**
- `/root/.ros/log/` - ROS 2 logs
- `/workspace/logs/` - Hardware interface logs (if configured)

---

## ✅ Verification Checklist

Before proceeding with vision development:

- [ ] RViz launches without errors
- [ ] Robot model visible in 3D view
- [ ] Interactive marker (orange ball) appears
- [ ] Camera frame (TF axes) visible at expected position
- [ ] Camera panel shows live feed (when Kinect connected)
- [ ] Joint states publishing at ~10 Hz
- [ ] No console errors or warnings

**Once all checked**, you're ready to start vision pipeline development!

---

**Last Updated:** 2026-01-21  
**Maintainer:** PAROL6 Vision Team
