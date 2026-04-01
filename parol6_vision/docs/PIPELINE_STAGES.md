# PAROL6 Vision Pipeline — Stage-by-Stage Documentation

**Package:** `parol6_vision` (ROS 2 Humble)  
**Date:** April 2026  
**Reference Diagram:** See `PIPELINE_DIAGRAM.png`

---

## Overview

The PAROL6 Vision Pipeline is a 7-stage, fully automated, vision-guided welding path detection and execution system. Each stage is an independent ROS 2 node that communicates exclusively through typed topics and services. The final output is a smooth 3D welding trajectory executed by the PAROL6 6-DOF robot arm via MoveIt2.

![PAROL6 Final Pipeline Diagram](/home/osama/.gemini/antigravity/brain/a85f6842-c918-4ff2-8886-572f893ecdfd/parol6_vision_pipeline_diagram_v3_final_1775079429707.png)

---

## Stage 1 — Image Capture (`capture_images_node`)

**File:** `parol6_vision/capture_images_node.py`  
**Node name:** `capture_images`

### Role

This is the **entry point** of the entire vision pipeline. It subscribes to three Kinect v2 camera topics and synchronizes the RGB color stream with the aligned depth stream using `message_filters.ApproximateTimeSynchronizer`. When a trigger event fires, a synchronized color+depth pair is atomically published to the pipeline's internal topics.

### Why it matters

Precise synchronization at this stage is critical. If the color image and depth image are not aligned in time, the 2D-to-3D back-projection in Stage 5 will produce incorrect 3D coordinates. The `TRANSIENT_LOCAL` QoS on the depth and camera info outputs ensures that `depth_matcher` always receives the last captured depth frame — even if it starts or restarts minutes after the capture event.

### Trigger Modes

| Mode | Mechanism |
|------|-----------|
| `keyboard` | Press `'s' + Enter` on the terminal; implemented via a background daemon thread reading `stdin`. |
| `timed` | A ROS timer fires every `frame_time` seconds (default 10 s) for automatic capture. |
| Topic trigger | `/vision/capture_trigger` (`std_msgs/Empty`) — used by the GUI. |

### Key Topics

| Direction | Topic | QoS |
|-----------|-------|-----|
| Sub | `/kinect2/sd/image_color_rect` | default |
| Sub | `/kinect2/sd/image_depth_rect` | default |
| Sub | `/kinect2/sd/camera_info` | default |
| Pub | `/vision/captured_image_raw` | VOLATILE |
| Pub | `/vision/captured_image_depth` | **TRANSIENT_LOCAL** |
| Pub | `/vision/captured_camera_info` | **TRANSIENT_LOCAL** |

---

## Stage 2 — Image Relay / Crop (`crop_image_node`)

**File:** `parol6_vision/crop_image_node.py`  
**Node name:** `crop_image`

### Role

An **always-active relay** node that sits between `capture_images` and the rest of the pipeline. It optionally applies a spatial polygon mask or rectangular crop to the captured color frame before forwarding it downstream. Configuration is read from `~/.parol6/crop_config.json` and can be reloaded live without restarting the node.

### Why it matters

Removing irrelevant background regions from the camera view before processing dramatically reduces false positives in color thresholding and YOLO segmentation (Stages 3 and beyond). The preferred **mask mode** preserves the original image resolution, which is essential for the depth map pixel-coordinate alignment used in Stage 5.

### Operating Modes

| Mode | Behavior |
|------|----------|
| **Mask** (recommended) | Fills pixels outside a polygon with black; image dimensions unchanged. Depth alignment preserved. |
| **Crop** (legacy) | Crops to a bounding box, changing image dimensions — avoid when depth is needed. |

### Key Topics

| Direction | Topic |
|-----------|-------|
| Sub | `/vision/captured_image_raw` |
| Pub | `/vision/captured_image_color` |

### Services

| Service | Description |
|---------|-------------|
| `~/reload_roi` | Re-reads config from disk (live update). |
| `~/clear_roi` | Disables ROI, passes image through unchanged. |

---

## Stage 3 — Seam Intersection Detection (Processing Mode Nodes)

**Nodes:** `color_mode`, `yolo_segment`, or `manual_line` (run **exactly one**)

### Role

This stage performs the **core perception task**: detecting where the two workpieces meet (the weld seam location). Exactly one of three interchangeable processing mode nodes is run at a time. All three produce identical output topics, making the downstream stages completely agnostic to which mode is active.

### Why it matters

Different workpieces, environments, and operational requirements call for different detection strategies. This three-mode architecture makes the pipeline robust across a wide range of real-world conditions without requiring code changes — simply launching a different node changes the detection strategy.

### Mode Comparison

| Mode | Strategy | Best For |
|------|----------|----------|
| `color_mode` | HSV thresholding on green & blue workpieces | Controlled lighting, distinctly colored parts |
| `yolo_segment` | YOLOv8 instance segmentation (ML model) | Arbitrary workpiece shapes, variable lighting |
| `manual_line` | Operator-drawn polyline strokes overlaid on frames | Fixed fixtures, deterministic or repeated jobs |

### Shared Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` | Frame with detected/drawn seam region |
| `/vision/processing_mode/debug_image` | `sensor_msgs/Image` | Same with extra debug annotations |
| `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` | Pixel-space centroid of the detected seam |

#### `color_mode` Detail

Performs a 7-step OpenCV pipeline: BGR→HSV conversion → dual-channel masking → morphological OPEN (noise removal) → dilation by `expand_px` pixels (creates overlap at the seam gap) → bitwise AND to find intersection → contour selection → centroid via image moments.

#### `yolo_segment` Detail

Runs YOLOv8 instance segmentation inference, extracts binary masks for the two largest detected objects, applies the same morphological + dilation + intersection pipeline as `color_mode`. More robust to lighting and surface finish, but requires a trained model file.

#### `manual_line` Detail

No image analysis is performed. The operator draws polyline strokes once via the GUI. These strokes are serialized to `~/.parol6/manual_line_config.json` and replayed (painted in red) on every subsequent frame. Auto-loaded on restart — ideal for repeat jobs at the same fixture.

---

## Stage 4 — Red Line Optimization (`path_optimizer`)

**File:** `parol6_vision/path_optimizer.py`  
**Node name:** `path_optimizer`

### Role

Detects the red weld marker line(s) painted on the workpiece surface (or generated by `manual_line`) within the annotated image from Stage 3. It applies a multi-step computer vision pipeline and outputs a structured `WeldLineArray` message with ordered pixel-space points representing the weld path.

### Why it matters

The annotated image from Stage 3 shows *where* the seam region is, but the exact weld path is encoded as a red marker drawn by the operator. Stage 4 extracts this path with sub-pixel precision using skeletonization and PCA-based point ordering, producing a clean, ordered sequence of 2D pixels that Stage 5 can back-project into 3D space.

### Detection Pipeline

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | HSV dual-range red masking | Red hue wraps around 0°/180° in OpenCV HSV; two `inRange` calls handle both ends |
| 2 | Morphological erode + dilate | Remove salt-pepper noise; fill gaps and reconnect fragments |
| 3 | Skeletonization (`skimage`) | Reduces the thick marker region to a 1-pixel-wide centerline |
| 4 | Contour detection + PCA ordering | Finds connected components; sorts each contour from start to end using its principal axis |
| 5 | Douglas-Peucker simplification | Reduces point count while preserving line geometry |
| 6 | Confidence scoring | Accepts only the contour with the highest point count as the single output line |

### Key Topics

| Direction | Topic | Type |
|-----------|-------|------|
| Sub | `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` |
| Pub | `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` |

---

## Stage 5 — 2D → 3D Reconstruction (`depth_matcher`)

**File:** `parol6_vision/depth_matcher.py`  
**Node name:** `depth_matcher`

### Role

The critical **bridge between the 2D vision world and 3D robot space**. It takes the ordered pixel-space weld line points from Stage 4 and back-projects each pixel into a 3D Cartesian coordinate using the depth camera's intrinsic matrix and the robot's TF2 transform tree.

### Why it matters

MoveIt2 requires 3D poses in the robot's `base_link` frame. This stage performs the mathematical lifting of 2D image data into robot-space 3D data using the aligned depth image captured all the way back in Stage 1. A cache-based architecture (rather than timestamp synchronization) is used because the operator may draw the weld line minutes after the depth was captured.

### Algorithm

1. **Cache-based depth**: Depth image and camera info are stored immediately on arrival. The `weld_lines_2d` callback triggers processing using these cached values — no timestamp matching required.
2. **Rate-limit gate**: A 0.5 s gate prevents the continuous stream of `weld_lines_2d` from flooding downstream nodes.
3. **Pinhole back-projection**: For each pixel `(u, v)` with depth `Z`: `X = (u − cx) × Z / fx`, `Y = (v − cy) × Z / fy`.
4. **TF2 transform**: Each 3D camera-frame point is transformed to `base_link` using TF2.
5. **Statistical outlier filtering**: Points deviating more than 2σ from the centroid distance are discarded.
6. **Quality gating**: Line accepted only if ≥ 10 valid points and depth quality ≥ 60%.

### Key Topics

| Direction | Topic | QoS |
|-----------|-------|-----|
| Sub | `/vision/weld_lines_2d` | VOLATILE |
| Sub | `/vision/captured_image_depth` | **TRANSIENT_LOCAL** |
| Sub | `/vision/captured_camera_info` | **TRANSIENT_LOCAL** |
| Pub | `/vision/weld_lines_3d` | VOLATILE |

---

## Stage 6 — Path Generation (`path_generator`)

**File:** `parol6_vision/path_generator.py`  
**Node name:** `path_generator`

### Role

Converts the raw 3D point cloud of the weld seam into a **smooth, kinematically feasible welding trajectory** (`nav_msgs/Path`). It applies PCA ordering, duplicate removal, cubic B-spline fitting, arc-length resampling, and 6-DOF orientation assignment to produce a trajectory ready for MoveIt2 execution.

### Why it matters

Raw 3D points from depth sensors are noisy and unevenly spaced. Sending these directly to the robot would produce jerky, unsafe motion and inconsistent heat input during welding. The B-spline smoothing and arc-length resampling ensure the robot moves at a constant speed along the weld seam — a critical requirement for weld quality.

### Algorithm

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | PCA ordering | Sorts the unordered 3D point cloud along the seam's principal axis |
| 2 | Duplicate removal | Removes points within 0.1 mm of each other to prevent spline fitting failures |
| 3 | Cubic B-spline fitting | `scipy.interpolate.splprep()` fits a smooth curve through the ordered points |
| 4 | Arc-length resampling | Re-samples waypoints at 5 mm intervals; caps at 80 waypoints for OMPL compatibility |
| 5 | Orientation assignment | Assigns a fixed downward quaternion to every waypoint for horizontal surface welding |
| 6 | Rate-limit gate | 0.5 s gate prevents flooding `moveit_controller` |

### Key Topics

| Direction | Topic | QoS |
|-----------|-------|-----|
| Sub | `/vision/weld_lines_3d` | VOLATILE |
| Pub | `/vision/welding_path` | **TRANSIENT_LOCAL** |

---

## Stage 7 — Motion Execution (`moveit_controller`)

**File:** `parol6_vision/moveit_controller.py`  
**Node name:** `moveit_controller`

### Role

The **final execution stage**. It receives the smooth welding trajectory from Stage 6 and commands the PAROL6 robot arm to physically execute the weld using **MoveIt2 Cartesian planning**. It implements a robust 3-tier fallback strategy to handle kinematic singularities, joint limits, and planning failures.

### Why it matters

Cartesian trajectory planning in constrained robotic environments is inherently brittle. A single point near a singularity can cause the entire trajectory plan to fail. The 3-tier fallback dynamically relaxes precision requirements to ensure the robot always executes the weld rather than silently failing.

### Execution Sequence

```
[Plan 1/3] Home trajectory       (joint-space, plan only)
[Plan 2/3] Approach trajectory   (15 cm above weld start, plan only)
[Plan 3/3] Cartesian weld traj   (full Cartesian, with fallback, plan only)
                 ↓ all plans ready
[Exec 1/3] Move to Home
[Exec 2/3] Move to Approach Point
[Exec 3/3] Execute Weld
```

### Cartesian Planning Fallback Strategy

| Attempt | Step Size | Success Threshold | Description |
|---------|-----------|-------------------|-------------|
| 1 | 2 mm | 95% | High-precision welding |
| 2 | 5 mm | 95% | Relaxed step to skip micro-singularities |
| 3 | 10 mm | 90% | Coarse "get the job done" mode |

If all Cartesian attempts fail and `enable_joint_waypoint_fallback` is `True`, the node falls back to joint-space moves to a coarse subset of 8 waypoints.

### Key Topics

| Direction | Topic | QoS |
|-----------|-------|-----|
| Sub | `/vision/welding_path` | **TRANSIENT_LOCAL** |

### Services

| Service | Description |
|---------|-------------|
| `~/execute_welding_path` | Manually trigger path execution |
| `~/is_execution_idle` | Query whether the controller is idle and has a path ready |

---

## Complete Data Flow Summary

```
Kinect v2 Camera
    │  (RGB + Depth + CameraInfo)
    ▼
Stage 1: capture_images      ── triggers on keyboard / timed / topic
    │  /vision/captured_image_raw       (VOLATILE)
    │  /vision/captured_image_depth     (TRANSIENT_LOCAL)
    │  /vision/captured_camera_info     (TRANSIENT_LOCAL)
    ▼
Stage 2: crop_image_node     ── polygon mask / crop ROI
    │  /vision/captured_image_color
    ▼
Stage 3: [color_mode | yolo_segment | manual_line]   ── pick ONE
    │  /vision/processing_mode/annotated_image
    ▼
Stage 4: path_optimizer      ── red line extraction + PCA ordering
    │  /vision/weld_lines_2d  (WeldLineArray)
    ▼
Stage 5: depth_matcher       ── pinhole back-projection + TF2
    │  /vision/weld_lines_3d  (WeldLine3DArray)
    ▼
Stage 6: path_generator      ── B-spline smoothing + arc-length resample
    │  /vision/welding_path   (nav_msgs/Path, TRANSIENT_LOCAL)
    ▼
Stage 7: moveit_controller   ── Cartesian execution via MoveIt2
```

---

*Documentation generated from `GRADUATION_DOCUMENT.md` — PAROL6 Vision Pipeline, March / April 2026.*
