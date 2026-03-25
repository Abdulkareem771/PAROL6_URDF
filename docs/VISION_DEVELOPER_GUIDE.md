# Vision System Developer Guide

**System:** PAROL6 Red Line Welding Path Detection  
**Purpose:** Complete developer reference for vision-guided welding implementation

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Development Environment Setup](#development-environment-setup)
4. [Package Structure](#package-structure)
5. [Node-by-Node Implementation](#node-by-node-implementation)
6. [Message Flow](#message-flow)
7. [Testing Strategy](#testing-strategy)
8. [Debugging Guide](#debugging-guide)
9. [Performance Optimization](#performance-optimization)
10. [Thesis Documentation](#thesis-documentation)

---

## System Overview

### Purpose

This system implements a complete perception → planning → execution pipeline for vision-guided robotic welding:

1. **Detect** red marker lines indicating welding seams
2. **Project** 2D detections into 3D using calibrated depth data
3. **Generate** smooth welding trajectories
4. **Execute** paths using MoveIt2 Cartesian planning

### Key Features

- ✅ **Custom semantic messages** (`parol6_msgs`) for weld-specific data
- ✅ **Modular architecture** - swap detectors without changing pipeline
- ✅ **Depth image processing** - faster than PointCloud2
- ✅ **MoveIt fallback strategies** - robust Cartesian planning
- ✅ **Comprehensive calibration** - thesis-grade validation
- ✅ **Full ROS2 Humble compatibility**

### Scientific Contribution

*"We defined custom semantic messages that encode welding seam geometry, rather than overloading object-detection message types, demonstrating clean system architecture and research maturity."*

---

## Architecture

### System Diagram

```
┌─────────────┐
│ Kinect v2   │
│ Camera      │
└──┬──────┬───┘
   │      │
   │ RGB  │ Depth+Info
   ▼      ▼
┌──────────────────┐
│ Node 1:          │
│ Red Line         │──── WeldLineArray ────┐
│ Detector         │                       │
└──────────────────┘                       ▼
                                ┌──────────────────┐
                                │ Node 2:          │
                                │ Depth Matcher    │──── WeldLine3DArray ────┐
                                └──────────────────┘                         │
                                                                             ▼
                                                                  ┌──────────────────┐
                                                                  │ Node 3:          │
                                                                  │ Path Generator   │──── Path ────┐
                                                                  └──────────────────┘              │
                                                                                                    ▼
                                                                                         ┌──────────────────┐
                                                                                         │ Node 4:          │
                                                                                         │ MoveIt Controller│
                                                                                         └────────┬─────────┘
                                                                                                  │
                                                                                                  ▼
                                                                                            [Robot Execution]
```

### Message Flow

| Stage | Input | Output | Message Type |
|-------|-------|--------|-------------|
| **Detection** | RGB Image | 2D Lines | `parol6_msgs/WeldLineArray` |
| **3D Projection** | 2D Lines + Depth | 3D Points | `parol6_msgs/WeldLine3DArray` |
| **Path Planning** | 3D Points | Waypoints | `nav_msgs/Path` |
| **Execution** | Waypoints | Motion | `control_msgs/FollowJointTrajectory` |

### Topic Naming Convention

**Semantic Outputs** (used by pipeline):
- `/vision/weld_lines_2d` - 2D detections
- `/vision/weld_lines_3d` - 3D projections
- `/vision/welding_path` - Generated path

**Debug Topics** (development only):
- `/red_line_detector/debug_image` - Visualization overlay
- `/red_line_detector/markers` - RViz markers
- `/depth_matcher/markers` - 3D point visualization

---

## Development Environment Setup

### Prerequisites

- PAROL6 Docker container running
- ROS2 Humble sourced
- Kinect v2 camera available

### Initial Setup

```bash
# 1. Enter container
docker exec -it parol6_dev bash

# 2. Navigate to workspace
cd /workspace

# 3. Create vision package (if not exists)
# This will be done during implementation

# 4. Build custom messages first
colcon build --packages-select parol6_msgs
source install/setup.bash

# 5. Verify messages
ros2 interface list | grep parol6_msgs
```

### Python Environment

```bash
# Inside container, install additional dependencies
pip3 install opencv-python numpy scipy scikit-image

# Verify installations
python3 << EOF
import cv2
import numpy as np
import scipy
from skimage import morphology
print("All dependencies OK")
EOF
```

---

## Package Structure

### parol6_msgs (Custom Messages)

```
parol6_msgs/
├── CMakeLists.txt          # Build configuration
├── package.xml             # Package dependencies
├── README.md               # Package documentation
└── msg/
    ├── WeldLine.msg        # 2D weld line
    ├── WeldLineArray.msg   # Array of 2D lines
    ├── WeldLine3D.msg      # 3D weld line
    └── WeldLine3DArray.msg # Array of 3D lines
```

**Key Files:**
- [README.md](file:///home/osama/Desktop/PAROL6_URDF/parol6_msgs/README.md) - Complete package documentation
- [WeldLine.msg](file:///home/osama/Desktop/PAROL6_URDF/parol6_msgs/msg/WeldLine.msg) - 2D message definition
- [WeldLine3D.msg](file:///home/osama/Desktop/PAROL6_URDF/parol6_msgs/msg/WeldLine3D.msg) - 3D message definition

### parol6_vision (Vision Pipeline)

```
parol6_vision/
├── package.xml
├── setup.py
├── setup.cfg
├── README.md               # Usage guide
├── DEVELOPER_GUIDE.md      # This file
├── config/
│   ├── camera_params.yaml
│   ├── detection_params.yaml
│   └── path_params.yaml
├── launch/
│   ├── vision_pipeline.launch.py
│   ├── red_detector_only.launch.py
│   └── full_system.launch.py
├── parol6_vision/
│   ├── __init__.py
│   ├── red_line_detector.py      # Node 1
│   ├── depth_matcher.py           # Node 2
│   ├── path_generator.py          # Node 3
│   ├── moveit_controller.py       # Node 4
│   └── utils/
│       ├── __init__.py
│       ├── cv_utils.py            # Computer vision helpers
│       ├── path_utils.py          # Path processing utilities
│       └── tf_utils.py            # Transform utilities
├── test/
│   ├── test_detector.py
│   ├── test_depth_matcher.py
│   └── test_path_generator.py
└── scripts/
    ├── calibrate_camera.py        # Camera calibration script
    └── test_pipeline.py           # Integration test
```

---

## Node-by-Node Implementation

### Node 1: Red Line Detector

**File:** `parol6_vision/red_line_detector.py`

**Purpose:** Detect red marker lines in images using computer vision

**Algorithm Overview:**
1. RGB → HSV color space conversion
2. Red color masking (two HSV ranges for wraparound)
3. Morphological operations (erosion/dilation)
4. Skeletonization to find centerlines
5. Contour detection and ordering
6. Douglas-Peucker polyline approximation

**Key Functions:**

```python
class RedLineDetector(Node):
    def image_callback(self, msg):
        """Main processing pipeline"""
        # 1. Convert to OpenCV
        # 2. HSV conversion
        # 3. Color masking
        # 4. Morphology
        # 5. Skeletonization
        # 6. Contour detection
        # 7. Publish WeldLineArray
    
    def compute_confidence(self, pixels_before, pixels_after, continuity):
        """Calculate confidence score"""
        return (pixels_after / pixels_before) * continuity
    
    def compute_continuity(self, points):
        """Measure line smoothness"""
        # Douglas-Peucker ratio + angle variance
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hsv_lower_1` | int[3] | [0,100,100] | Lower HSV for red (range 1) |
| `hsv_upper_1` | int[3] | [10,255,255] | Upper HSV for red (range 1) |
| `hsv_lower_2` | int[3] | [170,100,100] | Lower HSV for red (range 2) |
| `hsv_upper_2` | int[3] | [180,255,255] | Upper HSV for red (range 2) |
| `morphology_kernel_size` | int | 5 | Kernel size for morphology |
| `min_line_length` | int | 50 | Minimum line length (pixels) |
| `douglas_peucker_epsilon` | float | 2.0 | Polyline approximation tolerance |

**Topics:**

| Type | Topic | Message | Rate |
|------|-------|---------|------|
| Sub | `/kinect2/qhd/image_color_rect` | `sensor_msgs/Image` | 30 Hz |
| Pub | `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | 10 Hz |
| Pub | `/red_line_detector/debug_image` | `sensor_msgs/Image` | 10 Hz |
| Pub | `/red_line_detector/markers` | `visualization_msgs/MarkerArray` | 10 Hz |

**Development Notes:**
- Use `cv_bridge` for ROS↔OpenCV conversion
- Publish debug images for tuning HSV parameters
- Use RViz markers to visualize detections in 2D
- Log confidence scores for quality monitoring

---

### Node 2: Depth Matcher

**File:** `parol6_vision/depth_matcher.py`

**Purpose:** Project 2D pixel coordinates to 3D using depth camera data

**Algorithm Overview:**
1. Synchronize WeldLineArray + DepthImage + CameraInfo
2. For each pixel along line: lookup depth value
3. Back-project using camera intrinsics: `(u,v,d) → (X,Y,Z)`
4. Filter outliers (invalid depth, statistical)
5. Transform from camera frame → robot base frame
6. Publish WeldLine3DArray

**Key Functions:**

```python
class DepthMatcher(Node):
    def __init__(self):
        # Setup message_filters for synchronization
        self.sync = ApproximateTimeSynchronizer(...)
    
    def synchronized_callback(self, lines_msg, depth_msg, info_msg):
        """Process synchronized messages"""
        # 1. Extract camera intrinsics
        # 2. For each line in lines_msg
        # 3.   For each pixel in line
        # 4.     Lookup depth
        # 5.     Back-project to 3D
        # 6.   Filter outliers
        # 7.   Transform to base_link
        # 8. Publish WeldLine3DArray
    
    def backproject_pixel(self, u, v, depth, fx, fy, cx, cy):
        """Convert (u,v,depth) to (X,Y,Z)"""
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return Point(x=X, y=Y, z=Z)
    
    def filter_outliers(self, points):
        """Remove outliers beyond 2σ from median"""
        # Statistical filtering
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_frame` | string | "base_link" | Target coordinate frame |
| `outlier_std_threshold` | float | 2.0 | Outlier rejection threshold (σ) |
| `min_valid_points` | int | 10 | Minimum points for valid line |
| `max_depth` | float | 2000.0 | Max depth (mm, Kinect limit) |
| `sync_time_tolerance` | float | 0.05 | Sync tolerance (seconds) |

**Topics:**

| Type | Topic | Message | Rate |
|------|-------|---------|------|
| Sub | `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | 10 Hz |
| Sub | `/kinect2/qhd/image_depth_rect` | `sensor_msgs/Image` | 30 Hz |
| Sub | `/kinect2/qhd/camera_info` | `sensor_msgs/CameraInfo` | 30 Hz |
| Pub | `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | 10 Hz |
| Pub | `/depth_matcher/markers` | `visualization_msgs/MarkerArray` | 10 Hz |

**Development Notes:**
- Use `message_filters.ApproximateTimeSynchronizer` for sync
- TF lookups may fail initially - add retries with timeout
- Visualize 3D points in RViz to verify projection
- Monitor `depth_quality` field for calibration issues

---

### Node 3: Path Generator

**File:** `parol6_vision/path_generator.py`

**Purpose:** Convert 3D points into smooth welding trajectories

**Algorithm Overview:**
1. Order points using PCA (principal direction)
2. Fit B-spline curve (degree 3)
3. Resample at uniform spacing (e.g., 5mm)
4. Compute orientations from tangent + fixed angle
5. Publish as nav_msgs/Path

**Key Functions:**

```python
class PathGenerator(Node):
    def weld_lines_3d_callback(self, msg):
        """Generate path from 3D lines"""
        # 1. Extract points from WeldLine3D
        # 2. Order using PCA
        # 3. Fit B-spline
        # 4. Resample uniformly
        # 5. Generate orientations
        # 6. Publish Path
    
    def order_points_pca(self, points):
        """Order points along principal direction"""
        # PCA to find line direction
        # Project and sort
    
    def fit_bspline(self, points, degree=3, smoothing=0.01):
        """Fit B-spline curve"""
        # scipy.interpolate.splprep
    
    def generate_orientation(self, tangent, approach_angle_deg=45):
        """Compute quaternion from tangent and fixed angle"""
        # Tangent → pitch angle → quaternion
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spline_degree` | int | 3 | B-spline degree (cubic) |
| `spline_smoothing` | float | 0.01 | Smoothing factor |
| `waypoint_spacing` | float | 0.005 | Spacing (meters, 5mm) |
| `approach_angle_deg` | float | 45.0 | Torch angle (degrees) |
| `auto_generate` | bool | true | Auto-generate on detection |
| `min_waypoints` | int | 10 | Minimum waypoints required |

**Topics:**

| Type | Topic | Message | Rate |
|------|-------|---------|------|
| Sub | `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | 10 Hz |
| Pub | `/vision/welding_path` | `nav_msgs/Path` | 1-2 Hz |
|Pub | `/path_generator/markers` | `visualization_msgs/MarkerArray` | 1-2 Hz |

**Services:**
- `~/trigger_path_generation` (std_srvs/Trigger) - Manual trigger
- `~/get_path_statistics` (custom) - Path info

**Development Notes:**
- Planar surface assumption: orientation = tangent + fixed pitch
- Log path statistics (length, waypoint count, curvature)
- Visualize path in RViz with orientation arrows
- Validate waypoint spacing in tests

---

### Node 4: MoveIt Controller

**File:** `parol6_vision/moveit_controller.py`

**Purpose:** Execute welding paths using MoveIt2

**Algorithm Overview:**
1. Subscribe to welding path
2. Validate path (min waypoints, spacing)
3. Cartesian planning with 3-tier fallback:
   - Try 1: 2mm steps, 95% success required
   - Try 2: 5mm steps, 95% success required
   - Try 3: 10mm steps, 90% success required
4. Execute via FollowJointTrajectory action
5. Monitor feedback and handle errors

**Key Functions:**

```python
class MoveItController(Node):
    def path_callback(self, msg):
        """Receive and validate path"""
        if self.validate_path(msg):
            self.execute_welding_sequence(msg)
    
    def execute_welding_sequence(self, path):
        """Full welding sequence with pre/post phases"""
        # 1. Pre-weld: Move to approach point
        # 2. Plan Cartesian path with fallback
        # 3. Execute trajectory
        # 4. Post-weld: Retract
    
    def plan_cartesian_with_fallback(self, path):
        """Try multiple resolutions"""
        for step_size, threshold in zip([0.002, 0.005, 0.010], 
                                         [0.95, 0.95, 0.90]):
            success_rate, traj = self.compute_cartesian_path(path, step_size)
            if success_rate > threshold:
                return traj
        return None  # Planning failed
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planning_group` | string | "parol6_arm" | MoveIt planning group |
| `end_effector_link` | string | "link_6" | End effector frame |
| `cartesian_step_sizes` | float[] | [0.002, 0.005, 0.010] | Fallback steps |
| `min_success_rates` | float[] | [0.95, 0.95, 0.90] | Success thresholds |
| `approach_distance` | float | 0.05 | Pre-weld approach (m) |
| `weld_velocity` | float | 0.01 | Welding speed (m/s) |

**Topics:**

| Type | Topic | Message | Rate |
|------|-------|---------|------|
| Sub | `/vision/welding_path` | `nav_msgs/Path` | 1-2 Hz |

**Action Clients:**
- `/move_group` (moveit_msgs/action/MoveGroup)
- `/parol6_arm_controller/follow_joint_trajectory` (control_msgs/action/FollowJointTrajectory)

**Services:**
- `~/execute_welding_path` (std_srvs/Trigger) - Manual execution
- `~/abort_execution` (std_srvs/Trigger) - Emergency stop

**Development Notes:**
- Fallback strategy prevents random planning failures
- Log which fallback level was used
- Implement proper action preemption for safety
- Monitor execution feedback for errors

---

## Message Flow

### Complete Pipeline Example

**Step 1: Detection (Node 1)**
```python
# RedLineDetector detects line
line = WeldLine()
line.id = "red_line_0"
line.confidence = 0.95
line.pixels = [Point32(320, 240, 0), ...]  # 150 points

array = WeldLineArray()
array.lines = [line]

# Publish
pub.publish(array)  # → /vision/weld_lines_2d
```

**Step 2: 3D Projection (Node 2)**
```python
# DepthMatcher receives WeldLineArray
for line_2d in weld_lines_2d.lines:
    line_3d = WeldLine3D()
    line_3d.id = line_2d.id
    
    # Project each pixel
    for pixel in line_2d.pixels:
        depth = depth_image[int(pixel.y), int(pixel.x)]
        point_3d = backproject(pixel.x, pixel.y, depth)
        line_3d.points.append(point_3d)
    
    line_3d.depth_quality = 0.92
    line_3d.num_points = len(line_3d.points)

# Publish
pub.publish(WeldLine3DArray(lines=[line_3d]))  # → /vision/weld_lines_3d
```

**Step 3: Path Generation (Node 3)**
```python
# PathGenerator receives WeldLine3DArray
points = weld_line_3d.lines[0].points

# Order and smooth
ordered = order_points_pca(points)
spline = fit_bspline(ordered)
waypoints = resample_uniform(spline, spacing=0.005)

# Create Path
path = Path()
for wp in waypoints:
    pose = PoseStamped()
    pose.pose.position = wp
    pose.pose.orientation = compute_orientation(tangent_at(wp))
    path.poses.append(pose)

# Publish
pub.publish(path)  # → /vision/welding_path
```

**Step 4: Execution (Node 4)**
```python
# MoveItController receives Path
if validate_path(path):
    # Pre-weld
    move_to_approach(path.poses[0])
    
    # Plan with fallback
    trajectory = plan_cartesian_with_fallback(path)
    
    # Execute
    execute_trajectory(trajectory)
    
    # Post-weld
    retract()
```

---

## Testing Strategy

### Unit Tests

**Test 1: Message Creation**
```bash
cd /workspace
colcon test --packages-select parol6_msgs
```

**Test 2: Node Functionality**
```bash
# Test red line detector
colcon test --packages-select parol6_vision --pytest-args -k test_detector

# Test depth matcher
colcon test --packages-select parol6_vision --pytest-args -k test_depth_matcher
```

### Integration Tests

**Full Pipeline Test:**
```bash
# 1. Record test bag
ros2 bag record \
  /kinect2/qhd/image_color_rect \
  /kinect2/qhd/image_depth_rect \
  /kinect2/qhd/camera_info \
  -o test_weld_line

# 2. Run pipeline
ros2 launch parol6_vision vision_pipeline.launch.py

# 3. Play bag
ros2 bag play test_weld_line

# 4. Verify outputs
ros2 topic echo /vision/weld_lines_2d --once
ros2 topic echo /vision/weld_lines_3d --once
ros2 topic echo /vision/welding_path --once
```

### Manual Validation

See [Implementation Plan - Verification Section](file:///home/osama/.gemini/antigravity/brain/d2ceb7f4-79cd-48c2-8f8b-a2b6c9627e7c/implementation_plan.md#verification-plan) for detailed procedures.

---

## Debugging Guide

### Common Issues

**Issue 1: No detections**
```bash
# Check camera feed
ros2 topic echo /kinect2/qhd/image_color_rect --once

# View debug image
ros2 run rqt_image_view rqt_image_view /red_line_detector/debug_image

# Adjust HSV parameters in real-time
ros2 param set /red_line_detector hsv_lower_1 "[0, 80, 80]"
```

**Issue 2: 3D projection errors**
```bash
# Check TF tree
ros2 run tf2_tools view_frames
# View frames.pdf

# Verify camera calibration
ros2 topic echo /kinect2/qhd/camera_info --once

# Check depth quality
ros2 topic echo /vision/weld_lines_3d --field lines[0].depth_quality
```

**Issue 3: MoveIt planning fails**
```bash
# Check MoveIt logs
ros2 run rqt_console rqt_console

# Visualize in RViz
ros2 launch parol6_moveit_config moveit_rviz.launch.py

# Try manual planning in RViz first
```

### Logging Best Practices

```python
# In your nodes, use appropriate log levels:
self.get_logger().debug("Detailed variable: value")
self.get_logger().info("Normal operation event")
self.get_logger().warn("Unusual but handleable")
self.get_logger().error("Failure requiring attention")
```

---

## Performance Optimization

### Target Performance

| Metric | Target | Measured |
|--------|--------|----------|
| Detection rate | 10 Hz | TBD |
| 3D projection rate | 10 Hz | TBD |
| Path generation | 1-2 Hz | TBD |
| End-to-end latency | <200ms | TBD |

### Profiling

```python
# Add timing to nodes
import time

class MyNode(Node):
    def callback(self, msg):
        start = time.time()
        # ... processing ...
        elapsed = time.time() - start
        self.get_logger().info(f"Processing took {elapsed*1000:.1f}ms")
```

### Optimization Tips

1. **Use NumPy vectorization** instead of Python loops
2. **Downsample images** if real-time performance issues
3. **Cache TF lookups** when possible
4. **Profile with `cProfile`** to find bottlenecks

---

## Thesis Documentation

### Sections to Include

**1. System Architecture**
- Block diagram (use the one from this guide)
- Message flow diagram
- Justify custom messages

**2. Algorithms**
- Red line detection algorithm (HSV + morphology)
- 3D projection mathematics
- B-spline smoothing rationale

**3. Calibration**
- Camera intrinsic calibration procedure
- Camera-robot extrinsic calibration
- Validation accuracy (±Xmm)

**4. Results**
- Detection accuracy
- 3D reconstruction error
- Execution success rate
- Performance metrics

**5. Discussion**
- Planar surface assumption (scoping)
- Future work: YOLO integration
- Future work: Complex surface normals

---

## Related Documentation

- **[Implementation Plan](file:///home/osama/.gemini/antigravity/brain/d2ceb7f4-79cd-48c2-8f8b-a2b6c9627e7c/implementation_plan.md)** - Complete system plan
- **[parol6_msgs README](file:///home/osama/Desktop/PAROL6_URDF/parol6_msgs/README.md)** - Message package docs
- **[Camera Calibration Guide](file:///home/osama/Desktop/PAROL6_URDF/docs/CAMERA_CALIBRATION_GUIDE.md)** - Calibration procedures
- **[ROS System Architecture](file:///home/osama/Desktop/PAROL6_URDF/docs/ROS_SYSTEM_ARCHITECTURE.md)** - Overall PAROL6 system

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-18  
**Maintainer:** PAROL6 Vision Team
