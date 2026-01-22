# Depth Matcher Node - Developer Guide

## 1. Overview

The `depth_matcher` node is the **3D reconstruction component** of the PAROL6 vision pipeline. It takes 2D weld line detections from the `red_line_detector` and projects them into 3D space using synchronized depth data from the Kinect camera.

**Node Name:** `depth_matcher`  
**Package:** `parol6_vision`  
**Source:** `parol6_vision/depth_matcher.py`

**Purpose:**
- Convert 2D pixel coordinates to 3D points in robot workspace
- Synchronize RGB detections with depth measurements
- Transform camera coordinates to robot base frame
- Filter outliers and invalid depth readings

---

## 2. Architecture & Data Flow

### Input Sources (Synchronized)
The node uses `message_filters.ApproximateTimeSynchronizer` to align three streams:

1. **2D Detections** (`/vision/weld_lines_2d`) - From red_line_detector
2. **Depth Image** (`/kinect2/qhd/image_depth_rect`) - Aligned depth map (mm)
3. **Camera Info** (`/kinect2/qhd/camera_info`) - Intrinsic parameters

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Message Synchronization                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │
│  │ 2D Lines │  │  Depth   │  │  Camera Info     │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────────────┘     │
│       └─────────────┼─────────────┘                     │
└─────────────────────┼───────────────────────────────────┘
                      ▼
         ┌────────────────────────┐
         │  Parse Intrinsics      │
         │  (fx, fy, cx, cy)      │
         └────────┬───────────────┘
                  ▼
         ┌────────────────────────┐
         │  For Each Pixel (u,v): │
         │  1. Sample depth d     │
         │  2. Validate range     │
         │  3. Back-project to 3D │
         └────────┬───────────────┘
                  ▼
         ┌────────────────────────┐
         │  TF Transform          │
         │  Camera → Base Frame   │
         └────────┬───────────────┘
                  ▼
         ┌────────────────────────┐
         │  Statistical Outlier   │
         │  Removal (Filtering)   │
         └────────┬───────────────┘
                  ▼
         ┌────────────────────────┐
         │  Quality Check         │
         │  (min points, quality) │
         └────────┬───────────────┘
                  ▼
    ┌────────────────────────────────┐
    │  Publish WeldLine3DArray       │
    │  /vision/weld_lines_3d         │
    └────────────────────────────────┘
```

---

## 3. Algorithm Details

### 3.1 Pinhole Camera Back-Projection

For each pixel `(u, v)` in a detected line:

1. **Sample Depth:** `d = depth_image[v, u]` (in millimeters)
2. **Validate Depth:** Check if `min_depth ≤ d ≤ max_depth` and `d ≠ 0`
3. **Convert to Meters:** `Z = d / 1000.0`
4. **Back-Project to Camera Frame:**
   ```
   X_camera = (u - cx) × Z / fx
   Y_camera = (v - cy) × Z / fy
   Z_camera = Z
   ```

**Intrinsic Parameters:**
- `fx, fy`: Focal lengths (pixels)
- `cx, cy`: Principal point (image center, pixels)

These are automatically extracted from `/kinect2/qhd/camera_info`.

### 3.2 Coordinate Transformation

Points are transformed from **camera optical frame** to **robot base frame** using TF2:

```
Point_base = TF_transform(Point_camera, 'kinect2_rgb_optical_frame' → 'base_link')
```

This requires a **static transform** to be published (typically in launch file):
```python
static_transform_publisher --x 0.5 --y 0.0 --z 1.0 \
  --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 \
  --frame-id base_link --child-frame-id kinect2_rgb_optical_frame
```

### 3.3 Outlier Filtering

**Statistical Outlier Removal:**
1. Compute mean position of all 3D points
2. Calculate distance of each point from mean
3. Compute mean and standard deviation of distances
4. Remove points beyond `mean_dist + (threshold × std_dist)`

**Default threshold:** 2.0 (adjustable via `outlier_std_threshold` parameter)

### 3.4 Quality Metrics

Each 3D line includes quality indicators:
- **depth_quality:** Ratio of valid depth readings `(valid_points / total_pixels)`
- **num_points:** Count of 3D points after filtering
- **confidence:** Inherited from 2D detection

---

## 4. ROS API

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | 2D detections from red_line_detector |
| `/kinect2/qhd/image_depth_rect` | `sensor_msgs/Image` | Rectified depth image (16UC1, mm) |
| `/kinect2/qhd/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsic parameters |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | 3D weld lines in base frame |
| `/depth_matcher/markers` | `visualization_msgs/MarkerArray` | RViz visualization (3D points) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_frame` | string | `'base_link'` | Target coordinate frame for output |
| `depth_scale` | float | `1.0` | Depth scaling factor (usually 1.0 for Kinect) |
| `outlier_std_threshold` | float | `2.0` | Std dev multiplier for outlier removal |
| `min_valid_points` | int | `10` | Minimum 3D points to publish a line |
| `max_depth` | float | `2000.0` | Maximum valid depth (mm) |
| `min_depth` | float | `300.0` | Minimum valid depth (mm) |
| `min_depth_quality` | float | `0.6` | Minimum ratio of valid depth readings |
| `sync_time_tolerance` | float | `0.1` | Time sync tolerance (seconds) |
| `sync_queue_size` | int | `10` | Message synchronizer queue size |

---

## 5. Running the Node

### Standalone Mode

```bash
# Terminal 1: Start Kinect driver
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# Terminal 2: Start red line detector
ros2 run parol6_vision red_line_detector

# Terminal 3: Start depth matcher
ros2 run parol6_vision depth_matcher
```

### With Vision Pipeline

```bash
ros2 launch parol6_vision vision_pipeline.launch.py
```

This launches all nodes together: detector → depth_matcher → path_generator

### Verify Output

```bash
# Check if 3D lines are being published
ros2 topic echo /vision/weld_lines_3d

# Check publishing rate
ros2 topic hz /vision/weld_lines_3d

# Visualize in RViz
ros2 topic echo /depth_matcher/markers
```

---

## 6. Parameter Tuning Guide

### 6.1 Depth Range Adjustment

**Scenario:** Working distance from camera changed

**Parameters to adjust:**
```yaml
depth_matcher:
  ros__parameters:
    min_depth: 300.0    # Closest valid depth (mm)
    max_depth: 2000.0   # Farthest valid depth (mm)
```

**Recommended values:**
- **Close work (< 1m):** `min_depth: 200`, `max_depth: 1000`
- **Standard (1-2m):** `min_depth: 300`, `max_depth: 2000`
- **Far work (> 2m):** `min_depth: 500`, `max_depth: 3000`

### 6.2 Outlier Filtering

**Scenario:** Too many noisy points OR good points being removed

**Parameter to adjust:**
```yaml
outlier_std_threshold: 2.0  # Higher = more lenient, Lower = stricter
```

**Tuning guide:**
- **Noisy depth data:** Decrease to `1.5` or `1.0` (stricter filtering)
- **Clean but sparse data:** Increase to `2.5` or `3.0` (more lenient)
- **Visualize effect:** Check `/depth_matcher/markers` in RViz while adjusting

### 6.3 Quality Thresholds

**Scenario:** Lines with poor depth coverage are being published

**Parameters to adjust:**
```yaml
min_valid_points: 10      # Minimum 3D points required
min_depth_quality: 0.6    # Minimum % of valid depth readings
```

**Example adjustments:**
- **Stricter quality:** `min_valid_points: 20`, `min_depth_quality: 0.8`
- **More permissive:** `min_valid_points: 5`, `min_depth_quality: 0.4`

**When to relax thresholds:**
- Short weld lines (small objects)
- Challenging lighting conditions
- Material with poor depth reflectivity (shiny/dark surfaces)

### 6.4 Synchronization Tolerance

**Scenario:** "Could not synchronize messages" warnings

**Parameter to adjust:**
```yaml
sync_time_tolerance: 0.1  # Seconds
```

**Increase if:**
- Using slower camera frame rates
- Running on resource-constrained hardware
- Network delays between nodes

**Recommended values:**
- **Fast systems (30 Hz camera):** `0.05` seconds
- **Standard systems (15-30 Hz):** `0.1` seconds
- **Slow systems (< 15 Hz):** `0.2` seconds

---

## 7. Troubleshooting

### Issue 1: No 3D Lines Published

**Check 1: Verify 2D detections exist**
```bash
ros2 topic echo /vision/weld_lines_2d --once
# Should show at least one line
```

**Check 2: Verify depth image is publishing**
```bash
ros2 topic hz /kinect2/qhd/image_depth_rect
# Should show ~15-30 Hz
```

**Check 3: Check synchronization**
```bash
# Look for warnings in depth_matcher logs
# "Could not synchronize..." means timing mismatch
```

**Solution:** Increase `sync_time_tolerance` parameter

### Issue 2: "TF Lookup Failed"

**Error message:**
```
[depth_matcher]: Could not transform kinect2_rgb_optical_frame to base_link
```

**Cause:** Static transform not published

**Solution:**
```bash
# Verify transform exists
ros2 run tf2_ros tf2_echo base_link kinect2_rgb_optical_frame

# If error, publish static transform manually
ros2 run tf2_ros static_transform_publisher \
  --x 0.5 --y 0.0 --z 1.0 \
  --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 \
  --frame-id base_link --child-frame-id kinect2_rgb_optical_frame
```

### Issue 3: Points Have Wrong Scale/Position

**Symptom:** 3D points are tiny or huge, or in wrong location

**Possible causes:**

1. **Wrong depth scale:**
   - Kinect v2 uses millimeters → `depth_scale: 1.0`
   - If using meters → `depth_scale: 0.001`

2. **Incorrect camera transform:**
   - Verify actual camera position relative to robot
   - Use calibration tools (see CAMERA_CALIBRATION_GUIDE.md)

3. **Wrong target frame:**
   - Check `target_frame` parameter matches your robot base

### Issue 4: Too Few Points After Filtering

**Check depth quality in logs:**
```bash
# Look for "Published X 3D weld lines" message
# Check depth_quality field in published messages
ros2 topic echo /vision/weld_lines_3d
```

**Solutions:**
- Lower `min_depth_quality` threshold
- Increase `outlier_std_threshold` (less aggressive filtering)
- Decrease `min_valid_points`
- Improve lighting conditions for depth camera

### Issue 5: High Latency / Slow Processing

**Symptoms:** Delayed 3D output, low publishing rate

**Optimizations:**
1. Reduce `sync_queue_size` (default: 10 → try 5)
2. Simplify 2D detections (fewer points per line in detector)
3. Decrease camera resolution if supported

---

## 8. Visualization in RViz

### Setup RViz Display

1. **Add MarkerArray display:**
   - Click "Add" → "By topic"
   - Select `/depth_matcher/markers` → `MarkerArray`

2. **Configure display:**
   - Make sure "Fixed Frame" is set to `base_link`
   - 3D points appear as **blue spheres**
   - Line connectivity shown as **cyan lines**

### What You Should See

- **Blue spheres:** Individual 3D sample points along weld line
- **Cyan line:** Connects points to show line continuity
- **TF frame:** `kinect2_rgb_optical_frame` shows camera position

### Debugging with Visualization

- **No markers?** Depth matcher not publishing (check topics)
- **Markers far from robot?** Check TF transform correctness
- **Sparse points?** Increase `outlier_std_threshold` or decrease quality thresholds
- **Too many noisy points?** Decrease `outlier_std_threshold`

---

## 9. Integration with Pipeline

The depth_matcher sits between detection and planning:

```
red_line_detector → depth_matcher → path_generator → moveit_controller
     (2D pixels)      (3D points)      (Path)         (Trajectory)
```

**Key considerations:**

1. **Coordinate frames must match:**
   - Output frame (`base_link`) must match path_generator expectations
   
2. **Quality propagation:**
   - `confidence` from 2D detector is preserved
   - `depth_quality` is added as additional metric
   
3. **Timing:**
   - Synchronization tolerance affects end-to-end latency
   - Balance between accuracy and responsiveness

---

## 10. Performance Benchmarks

Expected performance on typical hardware:

| Metric | Target | Notes |
|--------|--------|-------|
| Processing Rate | 10-15 Hz | Limited by depth image rate |
| Latency | < 100ms | Including synchronization |
| 3D Points/Line | 50-200 | Depends on 2D detection density |
| Depth Quality | > 0.7 | For reliable 3D reconstruction |

**Measure actual performance:**
```bash
ros2 topic hz /vision/weld_lines_3d
ros2 topic delay /vision/weld_lines_3d
```

---

## 11. Best Practices

### For Development

1. **Test with mock data first:** Use `test_integration.launch.py`
2. **Visualize always:** Keep RViz open during development
3. **Log depth quality:** Monitor `depth_quality` field to catch issues early
4. **Tune incrementally:** Change one parameter at a time

### For Deployment

1. **Calibrate camera properly:** See CAMERA_CALIBRATION_GUIDE.md
2. **Validate transform accuracy:** Use known object positions
3. **Set conservative thresholds:** Better to reject bad data than accept noise
4. **Monitor performance:** Log processing times and quality metrics

### For Teammates

1. **Start with default parameters:** They're tuned for standard use
2. **Document your changes:** If you adjust params, note why
3. **Test in target environment:** Lighting and reflectivity matter
4. **Share findings:** If you find better param sets, update team

---

## 12. Related Documentation

- [Red Line Detector Guide](RED_LINE_DETECTOR_GUIDE.md) - Understanding 2D input
- [Testing Guide](TESTING_GUIDE.md) - How to test the full pipeline
- [Camera Calibration Guide](../../docs/CAMERA_CALIBRATION_GUIDE.md) - TF setup
- [RViz Setup Guide](RVIZ_SETUP_GUIDE.md) - Visualization configuration

---

**Last Updated:** 2026-01-23  
**Maintainer:** PAROL6 Vision Team
