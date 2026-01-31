# Depth Matcher Node - Developer Guide

## 1. Quick Start 🚀

**What is this?**  
This node takes 2D red lines detected in the camera image and converts them into 3D lines in the robot's coordinate system using the depth camera.

**How to Run:**
```bash
# 1. Start the camera driver (if not running)
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# 2. Start the vision pipeline (recommended)
ros2 launch parol6_vision vision_pipeline.launch.py

# OR Run standalone (for debugging)
ros2 run parol6_vision depth_matcher --ros-args -p target_frame:=base_link
```

**Verify it's working:**
```bash
# Check for 3D output
ros2 topic echo /vision/weld_lines_3d --once
```

---

## 2. Overview

The `depth_matcher` node is the **3D reconstruction component** of the PAROL6 vision pipeline.

**Node Name:** `depth_matcher`  
**Package:** `parol6_vision`  
**Source:** `parol6_vision/depth_matcher.py`

**Purpose:**
- Convert 2D pixel coordinates (u, v) → 3D points (x, y, z).
- Synchronize RGB detections with Depth maps.
- Transform points from Camera Frame → Robot Base Frame.
- Filter out noisy depth data.

---

## 3. Architecture & Data Flow

### Input Sources (Synchronized)
The node uses `message_filters` to ensuring the Depth image matches the exact moment the 2D line was detected.

| Source | Topic | Description |
|--------|-------|-------------|
| **2D Lines** | `/vision/weld_lines_2d` | From `red_line_detector` |
| **Depth** | `/kinect2/qhd/image_depth_rect` | 16-bit Depth Image (mm) |
| **Intrinsics**| `/kinect2/qhd/camera_info` | Pinhole camera parameters |

### Data Processing Pipeline
1. **Sync:** Wait for all 3 messages to arrive with matching timestamps.
2. **Back-Project:** For every pixel in the 2D line, look up the Depth value `Z`.
   - `X = (u - cx) * Z / fx`
   - `Y = (v - cy) * Z / fy`
3. **TF Transform:** Convert point from `camera_optical_frame` → `base_link`.
4. **Filter:** Remove "flying pixels" (noise) using statistical analysis.
5. **Publish:** Send the clean 3D line to the Path Generator.

---

## 4. Algorithm Details

### 4.1 Pinhole Camera Back-Projection
Mathematically, we reverse the camera projection:
- **Input:** Pixel `(u, v)` and Depth `d` (mm).
- **Camera Intrinsics:** Focal lengths `(fx, fy)` and Center `(cx, cy)`.
- **Output:** Point `(X, Y, Z)` in camera frame.

### 4.2 Outlier Filtering
Depth cameras (like Kinect) are noisy at edges. We filter points that don't belong:
- **Range Filter:** Ignore points closer than 0.3m or further than 2.0m.
- **Statistical Filter:** Calculate the mean position of the line points. Remove any point that represents a spike (more than 2 standard deviations away).

---

## 5. ROS Parameters

You can tune these in `parol6_vision/config/camera_params.yaml` or via command line.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_frame` | `base_link` | Output coordinate frame (robot base). |
| `min_depth` | `300.0` | Minimum valid depth in mm. |
| `max_depth` | `2000.0` | Maximum valid depth in mm. |
| `outlier_std_threshold` | `2.0` | Strictness of noise filter. Lower (1.0) is stricter. |
| `min_valid_points` | `10` | Min points required to form a valid 3D line. |
| `min_depth_quality` | `0.6` | Min ratio of valid/total pixels (e.g. 60%). |
| `sync_time_tolerance` | `0.1` | Max delay (sec) between RGB and Depth messages. |

---

## 6. Troubleshooting

### "Could not synchronize messages"
- **Cause:** Camera is lagging or network is slow.
- **Fix:** Increase `sync_time_tolerance` to `0.2`.

### "Could not transform..."
- **Cause:** The ROS Transform (TF) tree is broken. The system doesn't know where the camera is relative to the robot.
- **Fix:** Ensure static transform publisher is running:
  ```bash
  ros2 run tf2_ros static_transform_publisher 0.5 0 0.6 0 0 0 1 base_link kinect2_rgb_optical_frame
  ```

### No Output (3D Lines)
- **Check:** Is the object too close (<30cm) or too far (>2m)?
- **Fix:** Adjust `min_depth` or `max_depth`.

---

## 7. Visualization
Open RViz2 and add:
- **MarkerArray:** Topic `/depth_matcher/markers`
  - **Blue Dots:** Raw 3D points.
  - **Cyan Lines:** The connected weld seam.
