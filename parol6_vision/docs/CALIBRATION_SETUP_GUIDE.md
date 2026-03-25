# Camera Calibration Setup Guide

## Summary

Your Kinect v2 calibration files have been successfully converted! Here's what's ready and what you still need to provide.

---

## ✅ What's Already Calibrated (From Your Files)

### 1. **Intrinsic Parameters** (Camera Internal Geometry)
These were extracted from `parol6_vision/data/calib_color.yaml`:

```yaml
camera_intrinsics:
  fx: 1059.95  # Focal length X ✅
  fy: 1053.93  # Focal length Y ✅
  cx: 954.88   # Principal point X ✅
  cy: 523.74   # Principal point Y ✅
  distortion: [0.0563, -0.0742, 0.0014, -0.0017, 0.0241]  # ✅
```

**Status:** ✅ **DONE** - Your Kinect is already calibrated!

### 2. **Depth→Color Transform** (Internal Sensor Alignment)
This was extracted from `parol6_vision/data/calib_pose.yaml`:

```yaml
depth_to_color_transform:  # For reference only
  translation: {x: -0.052, y: -0.0005, z: 0.0009}
  rotation: {x: -0.0009, y: -0.0054, z: 0.0084, w: 0.9999}
```

**Status:** ✅ **DONE** - Depth and RGB sensors are aligned

---

## ⚠️ What You MUST Provide (Camera→Robot Calibration)

### **Extrinsic Parameters** (Camera Position Relative to Robot)

This is **THE CRITICAL MISSING PIECE** for the vision pipeline to work correctly. You need to measure/calibrate where the Kinect is positioned relative to your robot's base.

**Current values (PLACEHOLDERS):**
```yaml
camera_to_base_transform:
  translation:
    x: 0.5   # ⚠️ DEFAULT - NEEDS CALIBRATION
    y: 0.0   # ⚠️ DEFAULT - NEEDS CALIBRATION
    z: 1.0   # ⚠️ DEFAULT - NEEDS CALIBRATION
  rotation:
    x: -0.5  # ⚠️ DEFAULT - NEEDS CALIBRATION
    y: 0.5   # ⚠️ DEFAULT - NEEDS CALIBRATION
    z: -0.5  # ⚠️ DEFAULT - NEEDS CALIBRATION
    w: 0.5   # ⚠️ DEFAULT - NEEDS CALIBRATION
```

---

## 📏 How to Get Camera→Robot Transform

You have **3 options** (from easiest to most accurate):

### **Option 1: Manual Measurement** (±10-20mm accuracy)

**Steps:**
1. **Measure translation** (in meters):
   - **x**: Forward/backward from robot base
   - **y**: Left/right from robot base  
   - **z**: Up/down from robot base (floor to camera)

2. **Estimate rotation**:
   - Use a compass app or protractor to measure camera orientation
   - Convert angles to quaternion using online calculator
   - Or use this Python snippet:

```python
from scipy.spatial.transform import Rotation

# Example: Camera rotated 90° around Y axis
roll, pitch, yaw = 0, 90, 0  # degrees
r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
quat = r.as_quat()  # [x, y, z, w]
print(f"Quaternion: x={quat[0]:.3f}, y={quat[1]:.3f}, z={quat[2]:.3f}, w={quat[3]:.3f}")
```

### **Option 2: Semi-Automatic with ArUco Marker** (±5mm accuracy)

1. Print an ArUco marker and attach it to robot end-effector
2. Move robot to known position
3. Run detection script to calculate transform
4. See: `CAMERA_CALIBRATION_GUIDE.md` (Section: ArUco Method)

### **Option 3: Automated Hand-Eye Calibration** (±2-3mm accuracy)

Use ROS2 package `easy_handeye2`:
```bash
ros2 launch easy_handeye2 calibrate.launch.py
```

This requires:
- Robot must be movable via MoveIt
- Multiple robot poses (8-15 positions)
- ArUco marker visible to camera

---

## 🛠️ How to Apply Your Camera→Robot Transform

Once you have the measurements:

### Step 1: Edit the Calibrated File

```bash
nano parol6_vision/config/camera_params_calibrated.yaml
```

### Step 2: Update the Values

Replace the placeholder values:

```yaml
camera_to_base_transform:
  translation:
    x: YOUR_MEASURED_X  # Example: 0.45 (45cm forward)
    y: YOUR_MEASURED_Y  # Example: 0.20 (20cm to right)
    z: YOUR_MEASURED_Z  # Example: 1.05 (105cm up)
  rotation:
    x: YOUR_CALCULATED_QX  # From quaternion
    y: YOUR_CALCULATED_QY
    z: YOUR_CALCULATED_QZ
    w: YOUR_CALCULATED_QW
```

### Step 3: Activate the Calibration

```bash
# Backup original
cp parol6_vision/config/camera_params.yaml parol6_vision/config/camera_params_original.yaml

# Replace with calibrated version
cp parol6_vision/config/camera_params_calibrated.yaml parol6_vision/config/camera_params.yaml
```

### Step 4: Rebuild (if needed)

If using symlink install, the changes are instant. Otherwise:
```bash
colcon build --packages-select parol6_vision --symlink-install
source install/setup.bash
```

---

## ✅ Validation

After applying your camera→robot transform, validate it:

### Test 1: Visual Check in RViz

```bash
ros2 launch parol6_vision camera_setup.launch.py
```

**What to verify:**
- Camera frame (colored axes) appears at correct position relative to robot
- Camera is "looking" at the robot workspace

### Test 2: Known Object Position

1. Place a red marker at a known position (e.g., robot base origin)
2. Run vision pipeline:
   ```bash
   ros2 launch parol6_vision vision_pipeline.launch.py
   ```
3. Check detected 3D coordinates:
   ```bash
   ros2 topic echo /vision/weld_lines_3d
   ```
4. Compare detected position with actual position
5. Error should be < 10mm for good calibration

---

## 📦 Files Overview

| File | Purpose | Status |
|------|---------|--------|
| `data/calib_color.yaml` | Kinect intrinsic calibration | ✅ Source data |
| `data/calib_pose.yaml` | Depth→RGB alignment | ✅ Source data |
| `config/camera_params_calibrated.yaml` | **Generated** ROS config | ⚠️ Extrinsic needs update |
| `config/camera_params.yaml` | **Active** ROS config | ⚠️ Awaiting calibrated version |
| `convert_kinect_calibration.py` | Conversion script | ✅ Already ran |

---

## 🆘 Quick Reference: Common Setups

### Camera Mounted to the Side (Looking at Robot)

```yaml
# Camera 50cm to right, 1m up, looking at robot
translation: {x: 0.0, y: 0.5, z: 1.0}
rotation: {x: -0.5, y: 0.5, z: -0.5, w: 0.5}  # 90° rotation
```

### Camera Mounted Above (Looking Down)

```yaml
# Camera directly above robot, 1.5m up
translation: {x: 0.0, y: 0.0, z: 1.5}
rotation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}  # Looking down
```

### Camera Mounted in Front (Looking Forward)

```yaml
# Camera 60cm in front, 80cm up
translation: {x: 0.6, y: 0.0, z: 0.8}
rotation: {x: 0.0, y: 0.383, z: 0.0, w: 0.924}  # Tilted down 45°
```

---

## 📚 Related Documentation

- [CAMERA_CALIBRATION_GUIDE.md](../../docs/CAMERA_CALIBRATION_GUIDE.md) - Full calibration procedures
- [DEPTH_MATCHER_GUIDE.md](DEPTH_MATCHER_GUIDE.md) - How calibration affects 3D reconstruction
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Validation procedures

---

**Next Action:** Measure your camera position and update `camera_params_calibrated.yaml`!
