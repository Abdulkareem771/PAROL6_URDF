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

### Issue: "No detections" (Empty `/vision/weld_lines_2d`)

**Check 1: Verify image topic**
```bash
ros2 topic hz /kinect2/qhd/image_color_rect
# If no data, camera driver isn't running
```

**Check 2: Inspect debug image**
```bash
ros2 run rqt_image_view rqt_image_view
# Select /red_line_detector/debug_image
# You should see the mask overlay
```

**Check 3: Tune HSV parameters**
- If lighting conditions differ, adjust HSV thresholds in `config/detection_params.yaml`
- Red objects appear white in the mask; everything else should be black

### Issue: "ModuleNotFoundError: No module named 'rclpy'"

**Cause:** Trying to run the Python file directly outside ROS2 environment.

**Solution:** Always use `ros2 run` or `ros2 launch`:
```bash
# Wrong:
python3 red_line_detector.py

# Correct:
ros2 run parol6_vision red_line_detector
```

### Issue: "TF Lookup Failed" in Depth Matcher

**Cause:** Static transform from camera to robot base not published.

**Solution:** Verify transform publisher:
```bash
ros2 run tf2_ros tf2_echo base_link kinect2_rgb_optical_frame
# Should show the transform matrix
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
