# Path Generator Node - Developer Guide

## 1. Overview

The `path_generator` node is the **trajectory planning component** of the PAROL6 vision pipeline. It transforms unordered 3D weld line points into smooth, kinematically feasible welding paths with proper end-effector orientations.

**Node Name:** `path_generator`  
**Package:** `parol6_vision`  
**Source:** `parol6_vision/path_generator.py`

**Purpose:**
- Convert raw 3D point clouds into ordered, smooth trajectories
- Generate consistent waypoint spacing for uniform welding speed
- Compute end-effector orientations based on path geometry
- Publish ROS-standard `nav_msgs/Path` messages for MoveIt execution

---

## 2. Architecture & Data Flow

### Input
- **3D Weld Lines** (`/vision/weld_lines_3d`) - From depth_matcher
- Type: `parol6_msgs/WeldLine3DArray`
- Contains: Unordered 3D points in base_link frame

### Processing Pipeline

```
┌────────────────────────────────────────────────────────┐
│  Receive WeldLine3DArray (3D points in base_link)     │
└──────────────┬─────────────────────────────────────────┘
               ▼
     ┌─────────────────────┐
     │  1. PCA ORDERING    │  Sort points along principal direction
     │  (order_points_pca) │
     └──────────┬──────────┘
                ▼
     ┌─────────────────────┐
     │  2. DEDUPLICATION   │  Remove identical/near-duplicate points
     │  (remove_duplicates)│
     └──────────┬──────────┘
                ▼
     ┌─────────────────────────┐
     │  3. B-SPLINE FITTING    │  Fit cubic curve (degree=3)
     │  (scipy.interpolate)    │  Smoothing factor s=0.005
     └──────────┬──────────────┘
                ▼
     ┌──────────────────────────┐
     │  4. UNIFORM RESAMPLING   │  Resample at fixed distance
     │  (Arc-length param)      │  Default: 5mm spacing
     └──────────┬───────────────┘
                ▼
     ┌──────────────────────────┐
     │  5. ORIENTATION GEN      │  Compute 6-DOF poses
     │  (tangent + pitch angle) │  Default: 45° approach
     └──────────┬───────────────┘
                ▼
    ┌────────────────────────────┐
    │  Publish nav_msgs/Path     │
    │  /vision/welding_path      │
    └────────────────────────────┘
```

### Output
- **Welding Path** (`/vision/welding_path`) - nav_msgs/Path
- **Visualization** (`/path_generator/markers`) - MarkerArray (orientation arrows)

---

## 3. Algorithm Details

### 3.1 Point Ordering with PCA

**Problem:** 3D points from depth_matcher are unordered (random sequence from image processing).

**Solution:** Use Principal Component Analysis to find the primary direction of the weld line.

**Algorithm:**
```python
pca = PCA(n_components=1)
projected = pca.fit_transform(points)  # Project to 1D
sorted_indices = np.argsort(projected.flatten())
ordered_points = points[sorted_indices]
```

**Why PCA?**
- Finds the direction of maximum variance (the weld line direction)
- Robust to outliers compared to simple distance-based sorting
- Works for curved lines (projects onto best-fit principal axis)

### 3.2 B-Spline Smoothing

**Purpose:** Eliminate high-frequency noise from depth sensor while preserving shape.

**Mathematical Model:**
- **Cubic B-spline** (k=3): Guarantees C² continuity (smooth velocity and acceleration)
- **Smoothing parameter** s: Controls fit vs. smoothness trade-off

**How it works:**
```python
tck, u = interpolate.splprep(points.T, s=s, k=k)
# tck = (knot vector, coefficients, degree)
# u = parameter values [0, 1]
```

**Smoothing Parameter (s):**
- `s = 0.000`: Interpolating spline (passes through all points exactly)
- `s = 0.005`: Default (5mm allowed deviation) - balanced
- `s = 0.020`: High smoothing (ignores minor variations)

**Visual Effect:**
```
Raw Points (noisy):     Smoothed Spline:
    ●                        ───────
  ●   ●                    ╱         ╲
 ●     ●        →        ─           ─
  ●   ●                              
    ●
```

### 3.3 Arc-Length Reparameterization

**Problem:** Spline parameter `u` is NOT proportional to physical distance along the curve.

**Solution:** Resample at equal arc-length intervals for consistent welding speed.

**Algorithm Steps:**
1. **Generate fine-grained samples** on spline (u ∈ [0,1], 10× points)
2. **Compute cumulative arc length** using Euclidean distances
3. **Target waypoint distances:** Linearly spaced from 0 to total_length
4. **Inverse mapping:** For each target distance, find corresponding `u` value
5. **Evaluate spline** at optimized `u` values

**Result:** Waypoints spaced exactly 5mm apart (configurable) → constant velocity execution

### 3.4 Orientation Generation

**Scoping Assumption (Thesis):** Planar welding surfaces (non-complex 3D geometry).

**Coordinate Frame Definition:**
- **X-axis (Forward):** Path tangent vector (direction of welding)
- **Y-axis (Sideways):** Perpendicular to tangent and world Z
- **Z-axis (Down/Approach):** Torch approach vector

**Algorithm:**
```python
tangent = ∂spline/∂u (normalized)  # Forward direction
down = [0, 0, -1]                   # World down vector

y_axis = cross(down, tangent)       # Sideways (right-handed)
z_axis = cross(tangent, y_axis)     # Re-orthogonalize

R_base = [tangent, y_axis, z_axis]  # Rotation matrix

# Apply pitch rotation around Y (torch angle)
R_pitch = rotation_y(45°)
R_final = R_base @ R_pitch

quaternion = matrix_to_quat(R_final)
```

**Pitch Angle:** Default 45° (torch tilted toward weld direction for optimal penetration)

---

## 4. ROS API

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | 3D weld line points from depth_matcher |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/welding_path` | `nav_msgs/Path` | Smooth trajectory with 6-DOF poses |
| `/path_generator/markers` | `visualization_msgs/MarkerArray` | Orientation arrows (RViz visualization) |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `~/trigger_path_generation` | `std_srvs/Trigger` | Manually trigger path generation from latest weld line |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spline_degree` | int | `3` | B-spline polynomial degree (3 = cubic) |
| `spline_smoothing` | float | `0.005` | Smoothing factor (meters) |
| `waypoint_spacing` | float | `0.005` | Distance between waypoints (5mm) |
| `approach_angle_deg` | float | `45.0` | Torch pitch angle (degrees) |
| `auto_generate` | bool | `true` | Auto-publish on new detection |
| `min_points_for_path` | int | `5` | Minimum 3D points required |

---

## 5. Running the Node

### Standalone Mode

```bash
# Terminal 1: Prerequisites
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# Terminal 2: Red line detector
ros2 run parol6_vision red_line_detector

# Terminal 3: Depth matcher
ros2 run parol6_vision depth_matcher

# Terminal 4: Path generator
ros2 run parol6_vision path_generator
```

### With Full Pipeline

```bash
ros2 launch parol6_vision vision_pipeline.launch.py
```

### Manual Triggering

```bash
# Disable auto-generation
ros2 param set /path_generator auto_generate false

# Trigger manually when ready
ros2 service call /path_generator/trigger_path_generation std_srvs/srv/Trigger
```

### Verify Output

```bash
# Check if path is being published
ros2 topic echo /vision/welding_path

# Check waypoint count
ros2 topic echo /vision/welding_path --once | grep -c "PoseStamped"

# Visualize in RViz
ros2 run rviz2 rviz2
# Add display: /vision/welding_path (Path)
# Add display: /path_generator/markers (MarkerArray)
```

---

## 6. Parameter Tuning Guide

### 6.1 Smoothing Factor (`spline_smoothing`)

**What it controls:** How closely the spline fits the raw 3D points.

**Effect:**
- **Lower (0.001):** Tight fit, follows noise → jagged path
- **Default (0.005):** Balanced, removes sensor noise
- **Higher (0.020):** Very smooth, may miss small features

**Tuning procedure:**
```yaml
path_generator:
  ros__parameters:
    spline_smoothing: 0.005  # Start here
```

**Scenarios:**
| Scenario | Recommended Value | Reasoning |
|----------|-------------------|-----------|
| Clean depth data | 0.001 - 0.003 | Preserve detail |
| Noisy Kinect v2 | 0.005 - 0.010 | Filter noise |
| Very long weld (>1m) | 0.010 - 0.020 | Prevent overfitting |
| Sharp corners | 0.001 - 0.002 | Preserve geometry |

**Debug tip:** Visualize raw 3D points vs. smoothed path in RViz to see the effect.

### 6.2 Waypoint Spacing (`waypoint_spacing`)

**What it controls:** Physical distance between consecutive waypoints.

**Effect:**
- **Smaller (2mm):** More waypoints → smoother MoveIt trajectory, slower planning
- **Default (5mm):** Good balance for welding (typical bead width)
- **Larger (10mm):** Fewer waypoints → faster planning, may miss details

**Welding-specific considerations:**
```
Weld bead width: 3-8mm (typical)
→ Waypoint spacing: 5mm (optimal)

For fine work (< 3mm beads):
→ Waypoint spacing: 2-3mm

For coarse beads (> 8mm):
→ Waypoint spacing: 8-10mm
```

**MoveIt planning impact:**
- More waypoints → Higher Cartesian success rate
- Fewer waypoints → Easier collision checking

### 6.3 Approach Angle (`approach_angle_deg`)

**What it controls:** Torch tilt relative to the surface normal.

**Default:** 45° (industry standard for GMAW welding)

**Typical ranges:**
- **Flat surface:** 40° - 50°
- **Vertical seam:** 90° (perpendicular)
- **Overhead:** 30° - 40°

**Adjust for welding process:**
| Process | Recommended Angle |
|---------|-------------------|
| GMAW (MIG) | 45° (push) or 135° (pull) |
| GTAW (TIG) | 40° - 50° |
| Flux-core | 45° - 55° |

```yaml
path_generator:
  ros__parameters:
    approach_angle_deg: 45.0  # Adjust as needed
```

### 6.4 Minimum Points (`min_points_for_path`)

**What it controls:** Minimum 3D points required to generate a path.

**Default:** 5 points (minimum for cubic spline with deduplication)

**Adjust based on line length:**
```yaml
# Short weld lines (< 5cm):
min_points_for_path: 3

# Standard (5-20cm):
min_points_for_path: 5

# Long welds (> 20cm):
min_points_for_path: 10  # Ensure enough data
```

**Safety consideration:** Too few points → unreliable path → MoveIt may fail

---

## 7. Troubleshooting

### Issue 1: No Path Published

**Symptoms:** `/vision/welding_path` topic is silent

**Diagnostic steps:**
```bash
# 1. Check if 3D lines are being received
ros2 topic echo /vision/weld_lines_3d --once

# 2. Check path_generator logs
ros2 node info /path_generator

# 3. Check if auto_generate is enabled
ros2 param get /path_generator auto_generate
```

**Common causes:**
- No 3D weld lines detected
- Not enough points (< min_points_for_path)
- Spline fitting failed

**Solutions:**
- Lower `min_points_for_path` to 3
- Check depth_matcher is publishing
- Review logs for spline fitting errors

### Issue 2: "Spline fitting failed"

**Error message:**
```
[path_generator]: Spline fitting failed: Rank(B) (1) < deg+1 (4)
```

**Cause:** Not enough unique points after deduplication.

**Solutions:**
1. Lower `spline_degree` from 3 to 2:
   ```yaml
   spline_degree: 2  # Quadratic instead of cubic
   ```

2. Increase `waypoint_spacing` (generates fewer waypoints, less strict):
   ```yaml
   waypoint_spacing: 0.010  # 10mm instead of 5mm
   ```

3. Check for duplicate 3D points in depth_matcher output

### Issue 3: Path Too Jagged

**Symptom:** RViz shows zig-zag/noisy trajectory

**Cause:** Smoothing factor too low, following depth noise

**Solution:**
```yaml
spline_smoothing: 0.010  # Increase from 0.005
```

**Verify:** Compare raw 3D points with smoothed path in RViz

### Issue 4: Path Missing Details

**Symptom:** Sharp corners or features are rounded off

**Cause:** Smoothing factor too high

**Solution:**
```yaml
spline_smoothing: 0.002  # Decrease from 0.005
waypoint_spacing: 0.003   # Also increase density
```

### Issue 5: Wrong End-Effector Orientation

**Symptom:** Torch pointing in wrong direction in RViz

**Diagnostic:**
```bash
# Check orientation markers
ros2 topic echo /path_generator/markers
```

**Causes & Solutions:**

| Problem | Cause | Solution |
|---------|-------|----------|
| Torch pointing up | Negative pitch angle | Use positive angle (45°) |
| Torch sideways | Tangent calculation error | Check 3D line direction |
| Inconsistent orientation | Singularity (vertical line) | Add singularity handling |

**Adjust approach angle:**
```yaml
approach_angle_deg: -45.0  # Flip torch direction
```

### Issue 6: Too Many / Too Few Waypoints

**Symptom:** MoveIt planning very slow or path too coarse

**Check waypoint count:**
```bash
ros2 topic echo /vision/welding_path --once | grep -c "position:"
```

**Adjust spacing:**
```yaml
# For faster planning (fewer waypoints):
waypoint_spacing: 0.010  # 10mm

# For smoother execution (more waypoints):
waypoint_spacing: 0.003  # 3mm
```

**Rule of thumb:** 1 waypoint per 5mm of weld length is optimal

---

## 8. Visualization in RViz

### Setup RViz Displays

1. **Add Path Display:**
   - Click "Add" → "By topic"
   - Select `/vision/welding_path` → `Path`
   - Set color: Green
   - Adjust line width: 0.01m

2. **Add Orientation Markers:**
   - Click "Add" → "By topic"
   - Select `/path_generator/markers` → `MarkerArray`
   - Magenta arrows show end-effector orientation

3. **Compare with 3D Points:**
   - Add `/depth_matcher/markers` (blue spheres = raw points)
   - Add `/vision/welding_path` (green line = smoothed path)

### What to Look For

**Healthy Path:**
- ✅ Smooth green line following general shape of blue points
- ✅ Magenta arrows aligned with path direction
- ✅ Arrows tilted at ~45° (approach angle)
- ✅ Uniform spacing between arrows (every 5th waypoint shown)

**Problem Indicators:**
- ❌ Jagged/zig-zag green line → Increase smoothing
- ❌ Path deviates significantly from points → Decrease smoothing
- ❌ Arrows pointing random directions → Orientation calculation error
- ❌ Very few waypoints → Increase waypoint density

---

## 9. Integration with MoveIt

The path_generator output is designed to feed directly into MoveIt's Cartesian path planner.

### Data Flow

```
path_generator → /vision/welding_path (nav_msgs/Path)
                         ↓
              moveit_controller (reads Path)
                         ↓
           MoveIt Cartesian Planning
                         ↓
           JointTrajectory Execution
```

### Path Quality for MoveIt

**Good paths have:**
- ✅ Uniform waypoint spacing (5mm)
- ✅ Smooth orientation changes (< 5° per waypoint)
- ✅ Consistent tangent direction (no reversals)
- ✅ All poses within robot workspace

**Common MoveIt failures:**
- ❌ Inconsistent waypoint spacing → Solution: Use arc-length resampling (already implemented)
- ❌ Large orientation jumps → Solution: Decrease `waypoint_spacing` to 3mm
- ❌ Unreachable poses → Solution: Adjust camera position or workspace

---

## 10. Performance Benchmarks

Expected performance on typical hardware:

| Metric | Target | Notes |
|--------|--------|-------|
| Processing Time | < 100ms | For 50-100 point input |
| Path Generation Rate | 1-2 Hz | Triggered mode |
| Waypoint Count | 20-200 | Depends on weld length |
| Orientation Smoothness | < 5° change | Between consecutive waypoints |

**Measure actual performance:**
```bash
# Monitor processing latency
ros2 topic delay /vision/welding_path

# Check generation rate
ros2 topic hz /vision/welding_path
```

**Bottlenecks:**
- Spline fitting: O(n²) - slow for > 200 points
- Arc-length computation: O(n) - fast

---

## 11. Thesis-Ready Statements

Use these in your thesis documentation:

> *"To mitigate sensor noise and ensure kinematic smoothness, raw 3D points are fitted with a cubic B-spline. The curve is then re-parameterized by arc length to generate equidistant waypoints, critical for maintaining constant heat input during welding."*

> *"End-effector orientation is derived from the curve tangent vector, assuming a fixed torch parameterization relative to a planar workspace. This simplifies the orientation planning problem while remaining sufficient for linear seam welding tasks."*

> *"The use of Principal Component Analysis for point ordering ensures robustness to varying detection sequences, as image-based feature extraction does not guarantee spatially ordered output."*

**Scientific Contributions:**
- Separation of smoothing (B-spline) and discretization (arc-length sampling)
- Explicit handling of orientation singularities in path tangent computation
- Configurable trade-off between fit accuracy and noise rejection via smoothing parameter

---

## 12. Best Practices

### For Development

1. **Visualize always:** Keep RViz open to see raw points vs. smoothed path
2. **Test with mock data first:** Use synthetic straight/curved lines
3. **Tune incrementally:** Change one parameter at a time
4. **Log waypoint count:** Monitor if it's reasonable for path length

### For Deployment

1. **Set conservative smoothing:** Better to follow noise slightly than miss features
2. **Match spacing to bead width:** 5mm is standard for most welding
3. **Validate orientation:** Check first/last waypoint orientations make sense
4. **Test full pipeline:** Don't tune in isolation - test with MoveIt execution

### For Teammates

1. **Start with defaults:** They're tuned for standard use cases
2. **Document parameter changes:** Note why you adjusted values
3. **Share findings:** If you find better params, update team
4. **Test on real welds:** Simulation ≠ real-world performance

---

## 13. Related Documentation

- [Depth Matcher Guide](DEPTH_MATCHER_GUIDE.md) - Understanding 3D input data
- [Testing Guide](TESTING_GUIDE.md) - How to test the full pipeline
- [Implementation Plan](../../.gemini/antigravity/brain/.../implementation_plan.md) - Original design decisions

---

**Last Updated:** 2026-01-28  
**Maintainer:** PAROL6 Vision Team
