# Depth Matcher Node - Developer Guide

## 1. Overview

The `depth_matcher` node is the **3D reconstruction component** of the PAROL6 vision pipeline. It takes 2D weld line detections from an upstream detector (`path_optimizer` or `red_line_detector`) and projects them into 3D space using synchronized depth data from the Kinect camera.

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

### Input Sources (Cache-Based — No Timestamp Sync)

Depth image and camera info are cached on arrival and used when a `weld_lines_2d` message triggers processing:

1. **2D Detections** (`/vision/weld_lines_2d`) — From `path_optimizer` or `red_line_detector` — triggers processing
2. **Depth Image** (`/vision/captured_image_depth`) — TRANSIENT_LOCAL subscriber; cached into `_cached_depth`
3. **Camera Info** (`/vision/captured_camera_info`) — TRANSIENT_LOCAL subscriber; cached into `_cached_info`

> **Why cache-based?** The user draws weld lines *minutes after* capturing the depth frame. `ApproximateTimeSynchronizer` failed because the timestamps never matched. The cache-based approach decouples timing: depth is stored once at capture, and used whenever a line arrives.

A **0.5 s rate-limit gate** in `_on_lines()` prevents the continuous stream of `weld_lines_2d` (fired at camera rate) from flooding the pipeline with duplicate 3D projections.

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│            Cache-Based Acquisition (no sync)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │
│  │ 2D Lines │  │  Depth   │  │  Camera Info     │     │
│  │(trigger) │  │(cached)  │  │  (cached)        │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────────────┘     │
│       └─────────────┼─────────────┘ (used from cache)  │
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

These are automatically extracted from `/vision/captured_camera_info`.

### 3.2 Coordinate Transformation

Points are transformed from **camera optical frame** to **robot base frame** using TF2:

```
Point_base = TF_transform(Point_camera, 'kinect2_rgb_optical_frame' → 'base_link')
```

> **Note:** The node uses `rclpy.time.Time()` (i.e. "latest available") for TF
> lookups — not the message timestamp — so that replayed bag data (which carries
> old timestamps) works correctly with a live TF tree.

This requires a **static transform** to be published (typically in launch file):
```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.5 --y 0.0 --z 1.0 \
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

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | VOLATILE | 2D detections from `path_optimizer` or `red_line_detector` — triggers processing |
| `/vision/captured_image_depth` | `sensor_msgs/Image` | **TRANSIENT_LOCAL** | Captured aligned depth image (16UC1, mm); cached into `_cached_depth` |
| `/vision/captured_camera_info` | `sensor_msgs/CameraInfo` | **TRANSIENT_LOCAL** | Camera intrinsic parameters; cached into `_cached_info` |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | 3D weld lines in base frame |
| `/depth_matcher/markers` | `visualization_msgs/MarkerArray` | RViz visualization (3D points; DELETEALL sent first to prevent accumulation) |

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
| `depth_topic` | string | `/kinect2/sd/image_depth_rect` | Override depth source topic |
| `camera_info_topic` | string | `/kinect2/sd/camera_info` | Override camera info source topic |

---

## 5. Upstream 2D Detectors

The `depth_matcher` node can receive its 2D input from either of the two detectors below. Both publish to `/vision/weld_lines_2d`.

### 5.1 `path_optimizer` (Recommended)

**Source:** `parol6_vision/path_optimizer.py`

A streamlined detector designed for **variant-length weld lines**:
- Accepts lines of **all lengths** — no minimum length/contour-area filter
- Publishes **exactly one line per frame** (highest-confidence contour)
- Uses `/path_optimizer/` prefix for debug topics

#### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` | Annotated image from vision processing mode |

#### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | Best detected weld line (0 or 1 per frame) |
| `/path_optimizer/debug_image` | `sensor_msgs/Image` | Colour-coded detection overlay |
| `/path_optimizer/markers` | `visualization_msgs/MarkerArray` | RViz LINE_STRIP markers |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hsv_lower_1` | int[] | `[0, 70, 50]` | Lower red HSV range (0–10°) |
| `hsv_upper_1` | int[] | `[10, 255, 255]` | Upper bound of low-red range |
| `hsv_lower_2` | int[] | `[170, 70, 50]` | Lower red HSV range (160–180°) |
| `hsv_upper_2` | int[] | `[180, 255, 255]` | Upper bound of high-red range |
| `morphology_kernel_size` | int | `3` | Kernel size for morphological ops |
| `erosion_iterations` | int | `0` | Number of erosion passes |
| `dilation_iterations` | int | `0` | Number of dilation passes |
| `douglas_peucker_epsilon` | float | `2.0` | Polyline simplification tolerance (px) |
| `min_confidence` | float | `0.5` | Minimum confidence to publish a line |
| `publish_debug_images` | bool | `True` | Enable debug visualisation topics |

#### Detection Algorithm

```
Input Image (BGR)
      │
      ▼
┌─────────────────────────────┐
│  HSV Color Segmentation     │  Dual-range mask (0–10° ∪ 160–180°)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Morphological Processing   │  Erosion → Dilation
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Skeletonization            │  scikit-image, 1-px centerline
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Contour Extraction         │  ALL contours (no length filter)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  PCA Point Ordering         │  Start → end along principal axis
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Douglas-Peucker            │  Polyline simplification
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Confidence Scoring         │  retention × continuity ∈ [0,1]
└──────────────┬──────────────┘
               ▼
  Publish best line (highest point count above min_confidence)
```

**Confidence formula:**
```
confidence = retention × continuity
retention   = pixels_after_morphology / pixels_before_morphology
continuity  = exp(−angle_variance × 5)   ∈ [0, 1]
```

**Line selection:** The contour with the **most skeleton points** is chosen (not the highest confidence score), provided its confidence exceeds `min_confidence`.

---

### 5.2 `red_line_detector` (Legacy)

**Source:** `parol6_vision/red_line_detector.py`

The original multi-line detector. Key differences from `path_optimizer`:

| Feature | `path_optimizer` | `red_line_detector` |
|---------|-----------------|---------------------|
| Lines per frame | 1 (best only) | Up to `max_lines_per_frame` (default 5) |
| Minimum length filter | None | `min_line_length` (default 100 px) |
| Debug topic prefix | `/path_optimizer/` | `/red_line_detector/` |
| Kernel size default | 3 px | 5 px |

Both nodes subscribe to `/vision/processing_mode/annotated_image` and publish to `/vision/weld_lines_2d`.

---

## 6. Running the Node

### Standalone Mode

```bash
# Terminal 1: Start Kinect driver and capture images node
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# Terminal 2: Start capture images node (publishes depth + camera info)
ros2 run parol6_vision capture_images_node

# Terminal 3: Start upstream 2D detector (choose one)
ros2 run parol6_vision path_optimizer       # recommended
# ros2 run parol6_vision red_line_detector  # legacy alternative

# Terminal 4: Start depth matcher
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

## 7. Parameter Tuning Guide

### 7.1 Depth Range Adjustment

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

### 7.2 Outlier Filtering

**Scenario:** Too many noisy points OR good points being removed

**Parameter to adjust:**
```yaml
outlier_std_threshold: 2.0  # Higher = more lenient, Lower = stricter
```

**Tuning guide:**
- **Noisy depth data:** Decrease to `1.5` or `1.0` (stricter filtering)
- **Clean but sparse data:** Increase to `2.5` or `3.0` (more lenient)
- **Visualize effect:** Check `/depth_matcher/markers` in RViz while adjusting

### 7.3 Quality Thresholds

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

### 7.4 Rate Limiting

The node has a built-in **0.5 s rate-limit gate** (`_min_process_interval`). If `weld_lines_2d` arrives faster than 2 Hz (e.g., `manual_line_node` fires at camera rate), the gate drops the excess calls. This is correct behaviour — the weld line has not changed, so no new 3D projection is needed.

---

## 8. Troubleshooting

### Issue 1: No 3D Lines Published

**Check 1: Verify 2D detections exist**
```bash
ros2 topic echo /vision/weld_lines_2d --once
# Should show at least one line
```

**Check 2: Verify depth image is publishing**
```bash
ros2 topic hz /vision/captured_image_depth
# Should show a non-zero rate
```

**Check 3: Verify camera info is publishing**
```bash
ros2 topic hz /vision/captured_camera_info
# Should show a non-zero rate
```

**Check 4: Check if depth was captured first**
```bash
# Verify depth topic has TRANSIENT_LOCAL QoS (should show 1 publisher)
ros2 topic info /vision/captured_image_depth --verbose
# Look for: Durability: TRANSIENT_LOCAL
```

**Solution:** Capture an image first (`Press 's'` or use the GUI Capture button) before drawing weld lines. If `depth_matcher` starts after capture and topics have matching `TRANSIENT_LOCAL` QoS, it will receive the cached depth immediately.

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

### Issue 6: 2D Detector Publishes 0 Lines

**If using `path_optimizer`:**
```bash
# Check if red pixels are detected at all
ros2 topic echo /path_optimizer/debug_image  # view in RViz

# Verify HSV thresholds — use HSV_color_test.py utility:
# venvs/vision_venvs/ultralytics_cpu_env/YOLO_resources/HSV_color_test.py
```

**If using `red_line_detector`:**
- Lower `min_line_length` (default 100 px may be too restrictive for short seams)
- Check `min_confidence` threshold

---

## 9. Visualization in RViz

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

## 10. Integration with Pipeline

The `depth_matcher` sits between detection and planning:

```
capture_images_node ──────────────────────────────────┐
  (publishes depth + camera_info)                      │
                                                       ▼
path_optimizer ──► /vision/weld_lines_2d ──► depth_matcher ──► /vision/weld_lines_3d ──► path_generator ──► moveit_controller
  (2D pixels)                                (3D points)            (Path)                  (Trajectory)

─── OR ───

red_line_detector ──► /vision/weld_lines_2d ──► depth_matcher ...
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
   - The node uses TF "latest" time — avoid using bag timestamps directly

---

## 11. Performance Benchmarks

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

## 12. Best Practices

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

## 13. Related Documentation

- [Red Line Detector Guide](RED_LINE_DETECTOR_GUIDE.md) - Understanding the legacy 2D detector
- [Testing Guide](TESTING_GUIDE.md) - How to test the full pipeline
- [Camera Calibration Guide](../../docs/CAMERA_CALIBRATION_GUIDE.md) - TF setup
- [RViz Setup Guide](RVIZ_SETUP_GUIDE.md) - Visualization configuration

---

**Last Updated:** 2026-03-22  
**Maintainer:** PAROL6 Vision Team
