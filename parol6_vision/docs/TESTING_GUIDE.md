# Vision Pipeline Testing Guide

This guide provides step-by-step instructions for testing the PAROL6 vision pipeline, from basic setup to full integration testing.

---

## Prerequisites

### 1. Environment Setup
Ensure you have access to the Docker development environment:

```bash
# Navigate to project directory
cd ~/Desktop/PAROL6_URDF

# Start Docker container (if not already running)
docker start parol6_dev

# Enter the container
docker exec -it parol6_dev bash
```

### 2. Build the Workspace
Inside the Docker container:

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build packages
cd /workspace
colcon build --packages-select parol6_msgs parol6_vision

# Source the workspace
source install/setup.bash
```

**Verification:**
```bash
# Check if packages are visible
ros2 pkg list | grep parol6

# Expected output:
# parol6_msgs
# parol6_vision
```

---

## Test Level 1: Unit Tests

Unit tests validate individual components in isolation.

### Run All Unit Tests
```bash
cd /workspace
colcon test --packages-select parol6_vision --event-handlers console_direct+
```

### Run Specific Test Suites

**Red Line Detector Tests:**
```bash
colcon test --packages-select parol6_vision --pytest-args test/test_detector.py -v
```

**Depth Matcher Tests:**
```bash
colcon test --packages-select parol6_vision --pytest-args test/test_depth_matcher.py -v
```

**Path Generator Tests:**
```bash
colcon test --packages-select parol6_vision --pytest-args test/test_path_generator.py -v
```

**Expected Results:**
- All tests should pass with `OK` status
- Some tests may be skipped (marked with `SKIP`) - this is normal for tests requiring precise tuning

---

## Test Level 2: Integration Test (Mock Camera)

Integration tests validate the entire pipeline using synthetic data.

### Step 1: Launch Integration Test
```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Set required environment variables
export AMENT_PREFIX_PATH=/workspace/install/parol6_vision:/workspace/install/parol6_msgs:$AMENT_PREFIX_PATH
export PYTHONPATH=/workspace/install/parol6_vision/lib/python3.10/site-packages:/workspace/install/parol6_msgs/local/lib/python3.10/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/workspace/install/parol6_msgs/lib:$LD_LIBRARY_PATH

# Launch the integration test
ros2 launch parol6_vision test_integration.launch.py
```

### Step 2: Monitor Outputs
In a **new terminal** (inside Docker):

```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Watch for detected lines
ros2 topic echo /vision/weld_lines_2d
```

**Expected Output:**
```yaml
header:
  stamp:
    sec: <timestamp>
  frame_id: "kinect2_rgb_optical_frame"
lines:
- id: "red_line_0"
  confidence: 1.28  # Value > 0.5 indicates good detection
  pixels: [...]      # Array of detected points
```

### Step 3: Verify Detection Logs
Look for these log messages in the launch terminal:

```
[red_line_detector]: Segmentation found 9317 pixels
[red_line_detector]: Morphology output: 11881 pixels
[red_line_detector]: Skeleton pixels: 401
[red_line_detector]: Extracted 1 contours
[red_line_detector]: Frame X: Detected 1 line(s), confidences: ['1.28']
```

✅ **Success Criteria:**
- Detector finds ~9000-12000 segmented pixels
- Skeleton extraction yields ~400 pixels
- At least 1 contour is extracted
- Confidence score > 0.5

---

## Test Level 3: Red Line Detector (Standalone)

Test the detector node in isolation to diagnose issues.

### Step 1: Launch Detector Only
```bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

ros2 launch parol6_vision red_detector_only.launch.py
```

### Step 2: Verify Debug Visualization
In another terminal:

```bash
# Check available topics
ros2 topic list | grep red_line

# Expected topics:
# /red_line_detector/debug_image
# /vision/weld_lines_2d
```

### Step 3: Visualize in RViz (Optional)
If you have RViz access:
```bash
rviz2 -d /workspace/src/parol6_vision/config/vision_debug.rviz
```

---

## Test Level 4: Live Camera Testing

### Prerequisites
- Kinect v2 camera connected
- `freenect2` or equivalent driver running

### Step 1: Verify Camera Stream
```bash
# Check if camera topics are publishing
ros2 topic list | grep kinect

# Expected topics:
# /kinect2/qhd/image_color_rect
# /kinect2/qhd/image_depth_rect
# /kinect2/qhd/camera_info

# Check image stream
ros2 topic hz /kinect2/qhd/image_color_rect
# Expected: ~30 Hz
```

### Step 2: Launch Vision Pipeline
```bash
ros2 launch parol6_vision vision_pipeline.launch.py
```

### Step 3: Place Red Test Object
- Use red tape, marker, or painted line
- Ensure good lighting (avoid shadows)
- Place object in camera's field of view (~0.5-2m from camera)

### Step 4: Monitor Detections
```bash
# In another terminal
ros2 topic echo /vision/weld_lines_2d

# If detections appear, check downstream:
ros2 topic echo /vision/weld_lines_3d
ros2 topic echo /vision/welding_path
```

---

## Troubleshooting

### Issue 1: Package Not Visible After Build ⚠️

**Symptom:**
```bash
ros2 pkg list | grep parol6
# Only shows: parol6_msgs
# Missing: parol6_vision
```

**Root Cause:**
Stale build artifacts preventing proper package registration in `AMENT_PREFIX_PATH`.

**Solution:**
```bash
# Inside Docker container
cd /workspace

# Clean the problematic package completely
rm -rf build/parol6_vision install/parol6_vision

# Rebuild with symlink-install (recommended for development)
colcon build --packages-select parol6_vision --symlink-install

# Re-source the workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

# Verify both packages now appear
ros2 pkg list | grep parol6
# Expected:
# parol6_msgs
# parol6_vision
```

**Verify AMENT_PREFIX_PATH:**
```bash
echo $AMENT_PREFIX_PATH | tr ':' '\n'
# Should include:
# /workspace/install/parol6_vision
# /workspace/install/parol6_msgs
```

---

### Issue 2: "ModuleNotFoundError: No module named 'rclpy'" ❌

**Symptom:**
```
Traceback (most recent call last):
  File "red_line_detector.py", line 74, in <module>
    import rclpy
ModuleNotFoundError: No module named 'rclpy'
```

**Cause:** Trying to run the Python file directly outside the ROS2 environment.

**Solution:** Always use `ros2 run` or `ros2 launch`:
```bash
# ❌ Wrong (runs outside ROS2 context):
python3 /workspace/src/parol6_vision/parol6_vision/red_line_detector.py

# ✅ Correct (uses ROS2 environment):
ros2 run parol6_vision red_line_detector

# ✅ Or use launch file:
ros2 launch parol6_vision red_detector_only.launch.py
```

---

### Issue 3: "No detections" (Empty `/vision/weld_lines_2d`) 🔍

**Check 1: Verify image topic is publishing**
```bash
ros2 topic hz /kinect2/qhd/image_color_rect
# Expected: ~30 Hz (if camera is running)
# If 0 Hz or error → Camera driver isn't running
```

**Check 2: Inspect the detection mask**
```bash
ros2 run rqt_image_view rqt_image_view
# Select topic: /red_line_detector/debug_image
# Red objects should appear WHITE, everything else BLACK
```

**Check 3: Review detector logs**
```bash
# In the terminal running the detector, look for:
[red_line_detector]: Segmentation found X pixels
# If X = 0 → HSV color range needs tuning
# If X > 0 but no detections → Check min_line_length parameter
```

**Check 4: Tune HSV parameters**
Edit `config/detection_params.yaml`:
```yaml
red_line_detector:
  ros__parameters:
    hsv_lower_1: [0, 100, 100]     # Adjust based on your red marker
    hsv_upper_1: [10, 255, 255]
    hsv_lower_2: [170, 100, 100]
    hsv_upper_2: [180, 255, 255]
    min_line_length: 50             # Lower if detecting short lines
```

---

### Issue 4: Test Collection Error 🧪

**Symptom:**
```
ERROR collecting test session
ModuleNotFoundError: No module named 'test_integration'
```

**Cause:** Test discovery is finding a file it can't import (e.g., launch files in test/ directory).

**Quick Solution (Skip problematic tests):**
```bash
# Run specific test files only
colcon test --packages-select parol6_vision --pytest-args test/test_detector.py -v
colcon test --packages-select parol6_vision --pytest-args test/test_depth_matcher.py -v
```

**Permanent Solution:**
Add `pytest.ini` to exclude launch files:
```bash
# Create /workspace/src/parol6_vision/pytest.ini
[pytest]
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

---

### Issue 5: "TF Lookup Failed" in Depth Matcher ⚠️

**Symptom:**
```
[depth_matcher]: Could not transform kinect2_rgb_optical_frame to base_link: 
Lookup would require extrapolation into the future
```

**Cause:** Static transform from camera to robot base not published or timing mismatch.

**Solution 1: Verify transform is published**
```bash
# Check if transform exists
ros2 run tf2_ros tf2_echo base_link kinect2_rgb_optical_frame

# If error → Static transform publisher not running
# Check if test_integration.launch.py or vision_pipeline.launch.py includes:
# static_transform_publisher node
```

**Solution 2: Check launch file**
Ensure your launch file includes:
```python
Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=['0.5', '0.0', '1.0', '0', '0.707', '0', '0.707',
               'base_link', 'kinect2_rgb_optical_frame']
)
```

---

### Issue 6: Build Fails with "LD_LIBRARY_PATH" Error 🔧

**Symptom:**
```
ImportError: libparol6_msgs__rosidl_generator_py.so: cannot open shared object file
```

**Cause:** Custom message shared libraries not in library path.

**Solution:**
```bash
# Add to your build/launch script:
export LD_LIBRARY_PATH=/workspace/install/parol6_msgs/lib:$LD_LIBRARY_PATH

# Or add to ~/.bashrc inside Docker:
echo 'export LD_LIBRARY_PATH=/workspace/install/parol6_msgs/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

### Issue 7: "Extracted 0 contours" Despite Segmentation Success 🐛

**Symptom:**
```
[red_line_detector]: Segmentation found 9317 pixels
[red_line_detector]: Morphology output: 11881 pixels
[red_line_detector]: Skeleton pixels: 401
[red_line_detector]: Extracted 0 contours  ← Problem here
```

**Cause:** Contour filtering logic rejecting valid lines (e.g., by area instead of length).

**Diagnosis:**
Check if skeleton exists but contours are filtered:
- If `Skeleton pixels > 0` but `Extracted = 0` → Filtering bug
- If `Skeleton pixels = 0` → Skeletonization failed

**Solution (already implemented in current code):**
Filter skeletons by **length** not **area**, since 1-pixel wide lines have zero area:
```python
# ❌ Wrong (kills skeleton contours):
contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

# ✅ Correct (checks point count):
contours = [cnt for cnt in contours if len(cnt) >= min_line_length]
```

---

### Issue 8: Permission Denied Errors in Docker 🔒

**Symptom:**
```
Permission denied: '/workspace/build/parol6_vision/...'
```

**Cause:** Build directories owned by root inside Docker.

**Solution:**
```bash
# Inside Docker container
cd /workspace
sudo chown -R $USER:$USER build install log

# Or rebuild with current user:
colcon build --packages-select parol6_vision
```

---

### General Debugging Tips 💡

**Enable Verbose Logging:**
```bash
# Set ROS2 log level to DEBUG
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
ros2 run parol6_vision red_line_detector --ros-args --log-level debug
```

**Check Topic Connections:**
```bash
# See who's publishing/subscribing
ros2 topic info /vision/weld_lines_2d

# Monitor message flow
ros2 topic hz /vision/weld_lines_2d
ros2 topic echo /vision/weld_lines_2d --once
```

**Isolate Components:**
Test components individually:
1. Mock camera only
2. Mock camera + detector only
3. Mock camera + detector + depth matcher
4. Full pipeline

**Clean Rebuild:**
```bash
# Nuclear option - rebuild everything fresh
cd /workspace
rm -rf build install log
colcon build --packages-select parol6_msgs parol6_vision
source install/setup.bash
```


---

## Performance Benchmarks

Expected processing rates (on typical hardware):

| Component | Target Rate | Notes |
|-----------|-------------|-------|
| Red Line Detector | 5-10 Hz | Limited by HSV conversion + skeletonization |
| Depth Matcher | 10-15 Hz | Lightweight back-projection |
| Path Generator | 1-2 Hz | Spline fitting is slower, but triggered on-demand |

**Measure actual rates:**
```bash
ros2 topic hz /vision/weld_lines_2d
ros2 topic hz /vision/weld_lines_3d
ros2 topic hz /vision/welding_path
```

---

## Quick Validation Checklist

Use this checklist for rapid validation:

- [ ] Packages build without errors
- [ ] Unit tests pass (at least detector tests)
- [ ] Mock camera integration test shows detections
- [ ] Live camera stream is visible
- [ ] Red object is detected in `/weld_lines_2d`
- [ ] 3D points appear in `/weld_lines_3d` (if depth matcher enabled)
- [ ] Path published to `/welding_path` (if path generator enabled)

---

## Next Steps After Testing

Once all tests pass:
1. **Calibrate Camera**: Follow [`CAMERA_CALIBRATION_GUIDE.md`](file:///home/osama/Desktop/PAROL6_URDF/docs/CAMERA_CALIBRATION_GUIDE.md)
2. **Tune Detection Parameters**: Adjust HSV ranges for your specific markers
3. **Test with MoveIt**: Validate motion planning integration
4. **Create Demo Recording**: Document successful end-to-end execution

For detailed algorithm explanations, see [`RED_LINE_DETECTOR_GUIDE.md`](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/docs/RED_LINE_DETECTOR_GUIDE.md).
