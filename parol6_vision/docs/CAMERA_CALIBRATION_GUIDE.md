# Camera–Robot Calibration and Validation Guide

**Purpose:** Ensure accurate 3D reconstruction for vision-guided welding by properly calibrating camera intrinsics and the camera-to-robot transformation.

---

## Overview

This calibration process consists of three phases:

1. **Intrinsic Calibration:** Determine camera's internal parameters (focal length, principal point, distortion)
2. **Extrinsic Calibration:** Determine camera's position and orientation relative to robot base
3. **Validation:** Verify accuracy using known test objects

---

## Phase 1: Intrinsic Calibration

### Required Equipment
- Checkerboard calibration pattern (recommended: 9x6 squares, 25mm square size)
- Kinect v2 camera
- Good lighting conditions

### Procedure

#### Step 1: Print Calibration Pattern
```bash
# Download standard pattern
wget https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
# Print on A3 paper, measure actual square size
```

#### Step 2: Collect Calibration Images
```bash
# Launch Kinect
ros2 launch iai_kinect2 kinect2_bridge.launch.py

# Record calibration images (20-30 images from different angles)
mkdir -p ~/calibration_data
ros2 run image_view image_saver \
  --ros-args \
  -r image:=/kinect2/qhd/image_color_rect \
  -p filename_format:=~/calibration_data/calib_%04d.png
```

**Capture guidelines:**
- Move checkerboard to different positions and orientations
- Cover full field of view
- Include corners and edges of image
- Ensure checkerboard is fully visible in each image
- Aim for 20-30 high-quality images

#### Step 3: Run OpenCV Calibration
```bash
# Install calibration tools
pip3 install opencv-contrib-python

# Run calibration script
cd ~/PAROL6_URDF/parol6_vision/scripts
python3 calibrate_camera.py \
  --images ~/calibration_data/*.png \
  --pattern-size 9x6 \
  --square-size 0.025 \
  --output ~/calibration_data/camera_intrinsics.yaml
```

#### Step 4: Verify Calibration Quality

Good calibration metrics:
- **RMS reprojection error:** < 0.5 pixels (excellent), < 1.0 pixels (acceptable)
- **Individual image errors:** All < 1.0 pixels
- **Distortion coefficients:** Reasonable values (not extreme)

```bash
# Check results
cat ~/calibration_data/camera_intrinsics.yaml
```

Expected output format:
```yaml
camera_matrix:
  fx: 1081.372
  fy: 1081.372
  cx: 959.5
  cy: 539.5
distortion_coefficients:
  k1: 0.0862
  k2: -0.1234
  p1: 0.0012
  p2: -0.0005
  k3: 0.0234
image_size: [1920, 1080]
rms_error: 0.423
```

---

## Phase 2: Extrinsic Calibration (Camera-to-Robot Transform)

### Method A: Manual Measurement (Quick but less accurate)

#### Equipment Needed
- Measuring tape or ruler (±1mm accuracy)
- Level (to check alignment)

#### Procedure

1. **Measure Translation:**
   ```
   Camera optical center to robot base origin:
   - X (forward): _____ mm
   - Y (left): _____ mm  
   - Z (up): _____ mm
   ```

2. **Measure Rotation:**
   - Use level to check if camera is aligned with robot axes
   - Estimate roll, pitch, yaw angles
   - Convert to quaternion using online tool

3. **Record in config:**
   ```yaml
   # parol6_vision/config/camera_params.yaml
   camera_to_base_transform:
     translation: [0.500, 0.000, 0.600]  # meters
     rotation: [0.0, 0.0, 0.0, 1.0]      # quaternion [x,y,z,w]
   ```

**Expected Accuracy:** ±10-20mm

---

### Method B: Semi-Automatic Calibration (Recommended)

Uses robot to touch known points visible by camera.

#### Equipment Needed
- Small calibration marker/pointer attached to robot end-effector
- Calibration board with known 3D points

#### Procedure

1. **Setup Calibration Board:**
   - Place board with 4-6 distinct markers in camera view
   - Markers should be visible and easily detectable
   - Measure marker positions relative to robot base manually

2. **Collect Correspondences:**
   ```bash
   # Launch vision system
   ros2 launch parol6_vision camera_robot_calibration.launch.py
   
   # For each marker:
   # 1. Move robot tip to touch marker (use MoveIt in RViz)
   # 2. Record robot position from /joint_states → forward kinematics
   # 3. Record marker position in image
   # 4. Click "Add Point" in calibration GUI
   ```

3. **Compute Transform:**
   ```bash
   # After collecting 6+ point pairs
   # Run PnP solver to estimate camera pose
   python3 scripts/compute_camera_transform.py \
     --correspondences ~/calibration_data/point_pairs.yaml \
     --intrinsics ~/calibration_data/camera_intrinsics.yaml \
     --output ~/calibration_data/camera_extrinsics.yaml
   ```

4. **Update Configuration:**
   Copy results to `parol6_vision/config/camera_params.yaml`

**Expected Accuracy:** ±5mm

---

### Method C: Automatic Calibration (Most Accurate)

Uses ArUco markers and robot motion.

#### Procedure

1. **Attach ArUco Marker to Robot:**
   - Print ArUco marker (ID 0, 50mm size)
   - Attach to known position on end-effector
   - Measure marker offset from TCP precisely

2. **Automated Data Collection:**
   ```bash
   # Launch calibration routine
   ros2 launch parol6_vision auto_calibration.launch.py
   
   # Robot will automatically move to 10-15 poses
   # Camera detects ArUco marker at each pose
   # System collects robot_pose ↔ marker_in_camera pairs
   ```

3. **Solve Hand-Eye Calibration:**
   ```bash
   # Uses AX=XB formulation
   python3 scripts/hand_eye_calibration.py \
     --data ~/calibration_data/hand_eye_data.yaml \
     --method Tsai-Lenz \
     --output ~/calibration_data/camera_extrinsics.yaml
   ```

**Expected Accuracy:** ±2-3mm

---

## Phase 3: Validation

### Test 1: Known Object Position

**Setup:**
1. Place object (e.g., red marker) at known position
2. Measure position with ruler: X, Y, Z from robot base
3. Record values

**Validation:**
```bash
# Launch vision system
ros2 launch parol6_vision vision_pipeline.launch.py

# Echo 3D detection
ros2 topic echo /vision/weld_lines_3d --once

# Compare detected position to measured position
# Compute error: sqrt((x_det - x_meas)^2 + (y_det - y_meas)^2 + (z_det - z_meas)^2)
```

**Acceptance Criteria:**
- Position error < 10mm (good)
- Position error < 5mm (excellent)

---

### Test 2: Distance Measurement

**Setup:**
1. Place two markers at known distance apart (e.g., 100mm)
2. Measure with calipers

**Validation:**
```bash
# Detect both markers
ros2 topic echo /vision/weld_lines_3d

# Compute distance between detected positions
# Compare to measured distance
```

**Acceptance Criteria:**
- Distance error < 2% (e.g., ±2mm for 100mm distance)

---

### Test 3: Robot Touch Test

**Setup:**
1. Detect object with vision
2. Command robot to move to detected position
3. Observe if robot reaches correct location

**Validation:**
```bash
# Automatic test script
ros2 run parol6_vision test_vision_accuracy.py
```

**Acceptance Criteria:**
- Robot reaches within 10mm of target

---

## Calibration Maintenance

### When to Re-Calibrate

**Intrinsic calibration:**
- Camera is moved or bumped
- Focus ring is adjusted (if applicable)
- Temperature changes significantly
- Recommendation: Check every 3 months

**Extrinsic calibration:**
- Camera mount is adjusted
- Robot base is moved
- After any physical changes to setup
- Recommendation: Check every 2 weeks during development

### Quick Validation Test

Run this before each major demo/test:

```bash
# Places test marker at known position
# Runs detection
# Reports error
ros2 run parol6_vision quick_calibration_check.py
```

If error > 15mm, re-calibration recommended.

---

## Troubleshooting

### High Reprojection Error in Intrinsics
- **Cause:** Poor quality calibration images
- **Fix:** Recapture images with better lighting, sharper focus

### Large Position Errors (>20mm)
- **Cause:** Incorrect TF chain or wrong frame_id
- **Fix:** Verify TF tree with `ros2 run tf2_tools view_frames`

### Inconsistent Results
- **Cause:** Camera or robot vibration
- **Fix:** Ensure rigid mounting, average multiple measurements

### Depth Noise
- **Cause:** Infrared interference or poor surface reflectivity
- **Fix:** Improve lighting, avoid shiny surfaces

---

## Thesis Documentation

Include in thesis:

### Section: Camera–Robot Calibration
- Calibration method used (A/B/C)
- RMS reprojection error
- Validation test results (position errors)

### Figure: Calibration Setup
- Photo showing camera, robot, calibration board
- Annotated with coordinate frames

### Table: Validation Results
| Test Object | Measured Position (mm) | Detected Position (mm) | Error (mm) |
|-------------|----------------------|----------------------|------------|
| Marker 1    | (400, 0, 300)       | (398, 2, 302)        | 3.6        |
| Marker 2    | (500, 100, 300)     | (497, 98, 298)       | 4.1        |

This demonstrates scientific rigor and builds examiner confidence.

---

**Questions?** See troubleshooting or contact team for assistance.
