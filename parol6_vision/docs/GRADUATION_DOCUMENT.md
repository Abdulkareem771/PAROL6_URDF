# PAROL6 Vision-Guided Welding System — Graduation Documentation

**Project:** PAROL6 Robot Arm — Vision-Guided Welding Path Detection and Execution  
**Package:** `parol6_vision` (ROS 2)  
**Document Type:** Graduation / Thesis Technical Documentation  
**Date:** March 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 1: Image Capture — `capture_images_node`](#3-stage-1-image-capture--capture_images_node)
4. [Stage 1b: Image Relay / Crop — `crop_image_node`](#4-stage-1b-image-relay--crop--crop_image_node)
5. [Stage 2 (Alternative Input): Disk-Based Replay — `read_image_node`](#5-stage-2-alternative-input-disk-based-replay--read_image_node)
6. [Stage 3: Seam Intersection Detection — Processing Mode Nodes](#6-stage-3-seam-intersection-detection--processing-mode-nodes)
   - 6.1 [Color Mode: HSV-Based Detection — `color_mode`](#61-color-mode-hsv-based-detection--color_mode)
   - 6.2 [YOLO Segmentation Mode — `yolo_segment`](#62-yolo-segmentation-mode--yolo_segment)
   - 6.3 [Manual Line Mode — `manual_line`](#63-manual-line-mode--manual_line)
7. [Stage 4: Red Weld Line Detection — `red_line_detector` and `path_optimizer`](#7-stage-4-red-weld-line-detection--red_line_detector-and-path_optimizer)
   - 7.1 [Red Line Detector](#71-red-line-detector)
   - 7.2 [Path Optimizer](#72-path-optimizer)
8. [Stage 5: 2D → 3D Reconstruction — `depth_matcher`](#8-stage-5-2d--3d-reconstruction--depth_matcher)
9. [Stage 6: Path Generation — `path_generator`](#9-stage-6-path-generation--path_generator)
10. [Stage 7: Path Holding — `path_holder` and `inject_path_node`](#10-stage-7-path-holding--path_holder-and-inject_path_node)
11. [Stage 8: Motion Execution — `moveit_controller`](#11-stage-8-motion-execution--moveit_controller)
12. [Complete Topic & Message Flow Diagram](#12-complete-topic--message-flow-diagram)
13. [Key Custom Messages (`parol6_msgs`)](#13-key-custom-messages-parol6_msgs)
14. [Launch Files Summary](#14-launch-files-summary)
15. [Conclusion](#15-conclusion)

---

## 1. System Overview

The PAROL6 vision pipeline is a fully automated, vision-guided welding path detection and execution system built on **ROS 2**. It uses a **Microsoft Kinect v2** RGB-D camera to detect weld seam intersections on physical workpieces, extracts the welding path as a 3D trajectory, and commands the **PAROL6 6-DOF robot arm** to execute the weld.

### Design Goals

| Goal | Approach |
|------|----------|
| Detect weld seam location without physical markers | HSV color thresholding, YOLO segmentation, or manual stroke overlay |
| Support fixed-fixture repeat jobs without re-detection | Manual line node with saved stroke persistence |
| Achieve sub-pixel precision in 2D line detection | Skeletonization + PCA ordering |
| Reconstruct 3D welding path from depth data | Pinhole back-projection + TF2 transform |
| Generate smooth, kinematically feasible trajectories | Cubic B-spline fitting + arc-length resampling |
| Execute with constant velocity and fallback safety | MoveIt2 Cartesian planning with 3-tier fallback |

---

## 2. Pipeline Architecture

The complete pipeline flows through **8 stages**. Each stage is an independent ROS 2 node that communicates exclusively through typed topics and services:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PAROL6 Vision Pipeline                          │
└─────────────────────────────────────────────────────────────────────────┘

  [Kinect v2 Camera]
       │  /kinect2/sd/image_color_rect
       │  /kinect2/sd/image_depth_rect
       │  /kinect2/sd/camera_info
       ▼
 ┌─────────────────┐
 │ capture_images  │  Stage 1 — Synchronized RGB+Depth capture (keyboard or timed trigger)
 └────────┬────────┘
          │  /vision/captured_image_raw         (VOLATILE)
          │  /vision/captured_image_depth       (TRANSIENT_LOCAL)
          │  /vision/captured_camera_info       (TRANSIENT_LOCAL)
          ▼
 ┌─────────────────┐
 │ crop_image_node │  Stage 1b — Optional polygon mask / crop (ROI selection)
 └────────┬────────┘
          │  /vision/captured_image_color
          ▼
 ┌────────────────────────────────────────────────────────────┐
 │         Processing Mode (choose ONE mode)                  │  Stage 3
 │  ┌──────────────┐  ┌────────────────┐  ┌───────────────┐  │
 │  │  color_mode  │  │ yolo_segment   │  │  manual_line  │  │
 │  └──────────────┘  └────────────────┘  └───────────────┘  │
 └────────────────────────────┬───────────────────────────────┘
                    │  /vision/processing_mode/annotated_image
                    ▼
 ┌───────────────────────────────────────────────┐
 │   Red Line Detection (choose ONE detector)    │  Stage 4
 │  ┌──────────────────┐  ┌───────────────────┐  │
 │  │ red_line_detector│  │  path_optimizer   │  │
 │  └──────────────────┘  └───────────────────┘  │
 └─────────────────┬─────────────────────────────┘
                   │  /vision/weld_lines_2d  (WeldLineArray)
                   ▼
 ┌─────────────────┐
 │  depth_matcher  │  Stage 5 — Cache-based 2D → 3D projection via depth + TF2
 └────────┬────────┘
          │  /vision/weld_lines_3d  (WeldLine3DArray)
          ▼
 ┌─────────────────┐
 │ path_generator  │  Stage 6 — B-Spline smoothing + orientation generation
 └────────┬────────┘
          │  /vision/welding_path/generated  (TRANSIENT_LOCAL)
          ▼
 ┌────────────────────────────────────────────────┐
 │  path_holder  │  Stage 7 — Authoritative path latch & mux
 │  (also accepts /vision/welding_path/injected   │
 │   from inject_path_node for GUI injection)     │
 └────────┬───────────────────────────────────────┘
          │  /vision/welding_path  (TRANSIENT_LOCAL)
          ▼
 ┌─────────────────────┐
 │ moveit_controller   │  Stage 8 — Cartesian motion execution via MoveIt2
 └─────────────────────┘
```

---

## 3. Stage 1: Image Capture — `capture_images_node`

**File:** `parol6_vision/capture_images_node.py`  
**Node name:** `capture_images`

### Purpose

This is the **entry point** of the vision pipeline. It subscribes to the Kinect v2 camera topics, synchronizes color and depth image streams using `message_filters.ApproximateTimeSynchronizer`, and publishes matched image pairs to vision-internal topics upon a trigger event.

### Design Rationale

The captured depth image and camera info are published with **`TRANSIENT_LOCAL` QoS** (depth = 1). This ensures that `depth_matcher`, which may start long after the user pressed *Capture*, still receives the last depth frame immediately on joining. The color image is published `VOLATILE` because it is only consumed during active processing.

### Capture Modes

| Mode | How It Works |
|------|-------------|
| `keyboard` (default) | A background daemon thread reads `stdin`. Press `'s' + Enter` to capture. |
| `timed` | A ROS timer fires every `frame_time` seconds (default: 10 s) and publishes automatically. |

Both modes also accept a ROS topic trigger (`/vision/capture_trigger`, `std_msgs/Empty`) which is used by the GUI.

### Topic Interface

| Direction | Topic | Message Type | QoS | Description |
|-----------|-------|--------------|-----|-------------|
| Subscribed | `/kinect2/sd/image_color_rect` | `sensor_msgs/Image` | default | Rectified color |
| Subscribed | `/kinect2/sd/image_depth_rect` | `sensor_msgs/Image` | default | Aligned depth |
| Subscribed | `/kinect2/sd/camera_info` | `sensor_msgs/CameraInfo` | default | Camera intrinsics |
| Subscribed | `/vision/capture_trigger` | `std_msgs/Empty` | default | GUI trigger |
| Published | `/vision/captured_image_raw` | `sensor_msgs/Image` | VOLATILE | Captured color frame |
| Published | `/vision/captured_image_depth` | `sensor_msgs/Image` | **TRANSIENT_LOCAL** | Captured depth frame |
| Published | `/vision/captured_camera_info` | `sensor_msgs/CameraInfo` | **TRANSIENT_LOCAL** | Relayed intrinsics |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capture_mode` | string | `keyboard` | `keyboard` or `timed` |
| `frame_time` | float | `10.0` | Seconds between auto-captures in timed mode |
| `output_topic` | string | `/vision/captured_image_raw` | Output color topic |

---

## 4. Stage 1b: Image Relay / Crop — `crop_image_node`

**File:** `parol6_vision/crop_image_node.py`  
**Node name:** `crop_image`

### Purpose

This node is a **always-active relay** sitting between `capture_images` and the rest of the pipeline. It receives raw captured frames and optionally applies a spatial mask or crop before forwarding them downstream.

Its key design insight is that **pixel coordinate preservation** matters for 3D reconstruction:
- **Mask mode** (recommended): zeros out pixels outside a polygon, but keeps the **same image resolution**. Depth map alignment is preserved.
- **Crop mode** (legacy): crops to a rectangular bounding box, changing image dimensions (pixel coordinates are shifted—use only when downstream nodes are not depth-dependent).

### Configuration

Configuration is persisted in `~/.parol6/crop_config.json` and reloaded without restarting the node.

**Mask mode config:**
```json
{
  "enabled": true,
  "mode": "mask",
  "polygon": [[x1,y1], [x2,y2], [x3,y3], ...],
  "mask_color": [0, 0, 0]
}
```

**Crop mode config (legacy):**
```json
{
  "enabled": true,
  "mode": "crop",
  "x": 120, "y": 80, "width": 640, "height": 400
}
```

### Services

| Service | Type | Description |
|---------|------|-------------|
| `~/reload_roi` | `std_srvs/Trigger` | Re-read config from disk (live update) |
| `~/clear_roi` | `std_srvs/Trigger` | Disable processing (pass-through) |

### Topic Interface

| Direction | Topic | Message Type |
|-----------|-------|--------------|
| Subscribed | `/vision/captured_image_raw` | `sensor_msgs/Image` |
| Published | `/vision/captured_image_color` | `sensor_msgs/Image` |

---

## 5. Stage 2 (Alternative Input): Disk-Based Replay — `read_image_node`

**File:** `parol6_vision/read_image_node.py`  
**Node name:** `read_image`

### Purpose

This node provides an **offline testing path** for the pipeline. Instead of requiring a live Kinect camera, it watches a directory on disk (`parol6_vision/data/images_captured/`) and publishes new PNG image pairs as they appear.

### Mechanism

1. On startup, it scans the watch directory and **records all pre-existing files** (to avoid republishing old captures).
2. A polling timer (default: 1 Hz) checks for new `color_<timestamp>.png` + `depth_<timestamp>.png` pairs.
3. When a complete pair is found, both images are loaded and published with a **fresh ROS timestamp** and `frame_id = kinect2_rgb_optical_frame`, making them indistinguishable from live camera data to downstream nodes.

### Topic Interface

| Direction | Topic | Message Type |
|-----------|-------|--------------|
| Subscribed | `/kinect2/qhd/camera_info` | `sensor_msgs/CameraInfo` |
| Published | `/vision/captured_image_color` | `sensor_msgs/Image` (bgr8) |
| Published | `/vision/captured_image_depth` | `sensor_msgs/Image` (16UC1) |
| Published | `/vision/captured_camera_info` | `sensor_msgs/CameraInfo` |

---

## 6. Stage 3: Seam Intersection Detection — Processing Mode Nodes

The pipeline supports **three interchangeable processing modes**. All three produce identical output topics so that downstream nodes are completely agnostic to which mode is running:

- `/vision/processing_mode/annotated_image` — image showing the detected/drawn seam region
- `/vision/processing_mode/debug_image` — full debug overlay with extra annotations
- `/vision/processing_mode/seam_centroid` — pixel-space centroid (`geometry_msgs/PointStamped`)

| Mode | Detection Strategy | Best Used When |
|------|--------------------|----------------|
| `color_mode` | HSV color thresholding | Workpieces are distinctly green & blue |
| `yolo_segment` | YOLO instance segmentation | Arbitrary workpiece shapes / colors |
| `manual_line` | Operator-drawn stroke overlay | Fixed fixture, deterministic path required |

---

### 6.1 Color Mode: HSV-Based Detection — `color_mode`

**File:** `parol6_vision/color_mode.py`  
**Node name:** `color_mode`

#### Purpose

Detects the intersection between two colored workpieces (one **green**, one **blue**) using pure HSV color thresholding. This is the classical computer-vision approach—no neural network required.

#### Algorithm

The full processing pipeline runs on every incoming frame:

```
┌────────────────────────────────────────┐
│  1. BGR → HSV Color Space Conversion   │
├────────────────────────────────────────┤
│  2. Create Color Masks                  │
│     Green: H∈[35,100], S∈[50,255]      │
│     Blue:  H∈[100,140], S∈[50,255]     │
├────────────────────────────────────────┤
│  3. Morphological OPEN (5×5 kernel)    │
│     (noise suppression)                │
├────────────────────────────────────────┤
│  4. Dilation by `expand_px` pixels     │
│     (seam expansion using ellipse SE)  │
├────────────────────────────────────────┤
│  5. Bitwise AND → Intersection mask    │
│     Find largest external contour      │
├────────────────────────────────────────┤
│  6. Compute centroid via image moments │
├────────────────────────────────────────┤
│  7. Publish annotated_image,           │
│     debug_image, seam_centroid         │
└────────────────────────────────────────┘
```

**Step 4 Detail:** The dilation step is critical. By expanding each mask outward by `expand_px` pixels, the node creates an overlap region that corresponds to the narrow physical gap between the two workpieces — exactly where the weld seam lies.

**Step 5 Detail:** `cv2.bitwise_and(G_expanded, B_expanded)` yields a binary mask of the intersection. `cv2.moments()` on the largest contour gives the centroid coordinates.

#### Topic Interface

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribed | `/vision/captured_image_color` | `sensor_msgs/Image` |
| Published | `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` (bgr8) |
| Published | `/vision/processing_mode/debug_image` | `sensor_msgs/Image` (bgr8) |
| Published | `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` |

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/vision/captured_image_color` | Input topic |
| `expand_px` | `2` | Mask dilation radius (pixels) |
| `publish_debug` | `True` | Enable debug image publishing |

---

### 6.2 YOLO Segmentation Mode — `yolo_segment`

**File:** `parol6_vision/yolo_segment.py`  
**Node name:** `Yolo_segment`

#### Purpose

Detects workpiece seam intersections using **YOLO instance segmentation** (Ultralytics YOLOv8). Unlike the color mode, this approach does not require specific colored workpieces — it learns to identify the workpiece shapes from training data.

#### Algorithm

```
┌──────────────────────────────────────────────┐
│  1. YOLO Inference                           │
│     model(img, conf=mask_conf)               │
│     Extract masks for the 2 largest objects  │
├──────────────────────────────────────────────┤
│  2. Mask Extraction & Binarization           │
│     Resize to original image dimensions      │
│     Threshold > 0.5 → binary (0 or 255)      │
├──────────────────────────────────────────────┤
│  3. Morphological OPEN (5×5 kernel)          │
│     Removes noise from YOLO mask edges       │
├──────────────────────────────────────────────┤
│  4. Dilation by `expand_px` pixels           │
│     (elliptical structuring element)         │
├──────────────────────────────────────────────┤
│  5. Bitwise AND → Intersection mask          │
│     Find largest external contour            │
├──────────────────────────────────────────────┤
│  6. Centroid via image moments               │
├──────────────────────────────────────────────┤
│  7. Publish annotated_image,                 │
│     debug_image, seam_centroid               │
└──────────────────────────────────────────────┘
```

#### Model Configuration

The default model path is set to the pre-trained weights file:
```
/workspace/venvs/vision_venvs/ultralytics_cpu_env/
yolo_segmentation_models_results/experiment_2/weights/best.pt
```

This path is configurable via the `model_path` ROS parameter.

#### Key Differences from Color Mode

| Aspect | Color Mode | YOLO Mode |
|--------|-----------|-----------|
| Workpiece constraint | Must be green & blue | None (learned from data) |
| Computation | Lightweight, real-time | Heavier, needs model file |
| Robustness to lighting | Moderate | High |
| Training data required | No | Yes |

#### Topic Interface (identical to color mode)

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribed | `/vision/captured_image_color` | `sensor_msgs/Image` |
| Published | `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` |
| Published | `/vision/processing_mode/debug_image` | `sensor_msgs/Image` |
| Published | `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` |

---

### 6.3 Manual Line Mode — `manual_line`

**File:** `parol6_vision/manual_line_node.py`  
**Node name:** `manual_line`

#### Purpose

Provides a **stroke-replay** processing mode in which the operator manually draws one or more polyline paths on a GUI panel. Those strokes are then **painted onto every subsequent incoming camera frame** in red and published through the standard `processing_mode` topics — making this node a fully interchangeable replacement for `color_mode` or `yolo_segment`.

The defining feature is **persistence**: all strokes are serialised to `~/.parol6/manual_line_config.json` on every update, so the exact same seam annotation is replayed **automatically on every subsequent restart** — no re-drawing required for repeat jobs at the same fixture position.

#### Workflow

```
1. Operator draws lines in the GUI's Manual Red Line panel.
2. GUI serialises stroke data as JSON and calls ~/set_strokes service.
3. Node paints the strokes on every new frame; saves config to disk.
4. On next startup the node auto-loads the saved config and starts
   painting immediately — no action required from the operator.
5. To start fresh: GUI calls ~/reset_strokes → saved config is cleared.
```

#### Algorithm

The node is a pure **overlay renderer** — no image-analysis algorithm runs between input and output:

```
┌──────────────────────────────────────────────────┐
│  1. Image Decode                                 │
│     CvBridge → BGR NumPy array                   │
├──────────────────────────────────────────────────┤
│  2. Stroke Painting                               │
│     cv2.polylines() per stroke (anti-aliased)     │
│     Simultaneously build binary stroke_mask       │
├──────────────────────────────────────────────────┤
│  3. Centroid Computation                          │
│     ys, xs = np.where(stroke_mask > 0)            │
│     cx = xs.mean();  cy = ys.mean()               │
├──────────────────────────────────────────────────┤
│  4. Publish annotated_image, debug_image,         │
│     seam_centroid                                 │
└──────────────────────────────────────────────────┘
```

**Step 2 Detail:** Each stored stroke is an ordered list of `[x, y]` pixel coordinates (a polyline). `cv2.polylines()` renders the stroke with `cv2.LINE_AA` anti-aliasing. A parallel binary `stroke_mask` (same thickness) is rendered separately so the centroid calculation is unaffected by the colour overlay.

**Step 3 Detail:** The geometric centre of mass of **all painted pixels combined** across all strokes. This provides a meaningful single centroid even when multiple disjoint strokes are present. When no strokes are loaded, the centroid topic is **not published** for that frame.

**No-stroke pass-through:** When the stroke list is empty, the annotated image is a pass-through (unchanged input frame) and a status badge is displayed on the debug image: `Manual Line — no strokes saved`.

#### Config Persistence

The full stroke state is saved to `~/.parol6/manual_line_config.json`:

```json
{
  "color":   [0, 0, 255],
  "width":   5,
  "strokes": [
    [[x1, y1], [x2, y2], ...],
    [[x1, y1], [x2, y2], ...]
  ]
}
```

On startup, `_load_config()` silently restores this state. If the file does not exist the node starts with an empty stroke list and logs: `No saved config found — starting fresh.`

#### Topic Interface

| Direction | Topic | Type | Description |
|-----------|-------|------|-------------|
| Subscribed | `/vision/captured_image_color` | `sensor_msgs/Image` | Raw color frame (configurable) |
| Published | `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` (bgr8) | Frame with strokes painted in red |
| Published | `/vision/processing_mode/debug_image` | `sensor_msgs/Image` (bgr8) | Same + centroid crosshair & status badge |
| Published | `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` | Pixel-space centroid of all strokes |

#### Services

| Service | Type | Description |
|---------|------|-------------|
| `~/set_strokes` | `std_srvs/Trigger` | Reload strokes from `strokes_json` parameter and save to disk |
| `~/reset_strokes` | `std_srvs/Trigger` | Clear all strokes from memory and delete saved config |

**Setting strokes from the command line:**
```bash
# 1. Write the JSON payload into the ROS parameter
ros2 param set /manual_line strokes_json '[[[10,50],[200,50]],[[30,100],[180,100]]]'
# 2. Trigger the service to load it
ros2 service call /manual_line/set_strokes std_srvs/srv/Trigger {}
```

The `strokes_json` parameter accepts either a **plain JSON array** (list of polylines) or a **JSON object** with optional `color`, `width`, and `strokes` keys.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/vision/captured_image_color` | Input image topic |
| `stroke_color` | int[] | `[0, 0, 255]` | BGR paint colour (default = red) |
| `stroke_width` | int | `5` | Stroke thickness in pixels |
| `strokes_json` | string | `""` | JSON-encoded stroke list; updated by `set_strokes` |
| `publish_debug` | bool | `True` | Enable the debug overlay image |

#### Comparison with Other Processing Modes

| Feature | `manual_line` | `color_mode` | `yolo_segment` |
|---------|---------------|-------------|----------------|
| Detection strategy | Operator-drawn strokes | HSV thresholding | YOLO ML model |
| Workpiece constraint | None | Must be green & blue | None (learned) |
| Persistence across restarts | ✅ Saved to disk | ❌ | ❌ |
| Path re-drawing needed | Only once per fixture | Every session | Every session |
| Computation | Very lightweight | Lightweight | Heavy (GPU optional) |
| Robustness to lighting changes | Fully immune | Sensitive | High |

Use `manual_line` when:
- The part is clamped in a **fixed fixture** and does not move between runs.
- Automatic detection is unreliable due to lighting, reflections, or surface finish.
- A **deterministic, reproducible** weld path defined by the operator is required.

---

## 7. Stage 4: Red Weld Line Detection — `red_line_detector` and `path_optimizer`

Both nodes in this stage perform the **same fundamental task**: detect red marker lines drawn on the workpiece surface, which represent the desired weld path. They receive the annotated intersection image from Stage 3 and output `WeldLineArray` messages.

---

### 7.1 Red Line Detector

**File:** `parol6_vision/red_line_detector.py`  
**Node name:** `red_line_detector`

#### Purpose

Detects **one or more** red marker lines using a 5-stage computer vision pipeline. Designed for multi-line scenarios where the operator draws several candidate paths.

#### Detection Pipeline

**Stage 1 — HSV Color Segmentation:**  
Red hue wraps around 0°/180° in OpenCV's HSV space (H ∈ [0, 180]). Two inRange calls handle both ends of the wraparound:
```
mask1 = inRange(hsv, [0,100,100],   [10,255,255])   ← Low-red
mask2 = inRange(hsv, [160,50,0],    [180,255,255])   ← High-red
mask  = bitwise_or(mask1, mask2)
```

**Stage 2 — Morphological Processing:**
```
mask = erode(mask, kernel, erosion_iters)    # Remove salt-pepper noise
mask = dilate(mask, kernel, dilation_iters)  # Fill gaps, connect fragments
```

**Stage 3 — Skeletonization:**  
Uses `skimage.morphology.skeletonize()` to reduce the thick red marker region to a **1-pixel-wide centerline**. This provides sub-pixel precision in line localization and is crucial for clean downstream point ordering.

**Stage 4 — Contour Detection & PCA Ordering:**  
`cv2.findContours()` on the skeleton finds connected components. Each contour is then ordered using **Principal Component Analysis (PCA)**:
1. Fit PCA to the Nx2 point cloud (finds the principal direction of the line)
2. Project all points onto the first principal component
3. Sort by projection value → ordered from start to end of the line

**Stage 5 — Douglas-Peucker Simplification:**  
`cv2.approxPolyDP()` reduces point count while preserving line geometry. Note: dense skeleton points are kept for the `pixels` field (used by depth_matcher), while simplified points are used only for confidence computation.

**Confidence Computation:**
```
confidence = retention × continuity_score

retention      = pixels_after_morphology / pixels_before_morphology
continuity     = exp(−angle_variance × 5.0)
angle_variance = var(|consecutive angle differences|)
```

A confidence ≥ `min_confidence` (default 0.5) is required for a line to be published.

#### Topic Interface

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribed | `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` |
| Published | `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` |
| Published | `/red_line_detector/debug_image` | `sensor_msgs/Image` |
| Published | `/red_line_detector/markers` | `visualization_msgs/MarkerArray` |

---

### 7.2 Path Optimizer

**File:** `parol6_vision/path_optimizer.py`  
**Node name:** `path_optimizer`

#### Purpose

A **streamlined variant** of the red line detector, designed for manual annotation workflows or simple single-line scenarios. Key differences:

| Feature | `red_line_detector` | `path_optimizer` |
|---------|---------------------|-----------------|
| Minimum line length filter | Yes (configurable) | **None** — all lengths accepted |
| Lines published per frame | Multiple (up to `max_lines`) | **Exactly one** (highest point count) |
| Debug topic prefix | `/red_line_detector/` | `/path_optimizer/` |
| Input | Annotated image | Annotated image |

The same 7-stage algorithm (segmentation → morphology → skeletonization → contour extraction → PCA ordering → D-P simplification → confidence scoring) is used. The key difference is in contour selection: the **contour with the most skeleton points** is selected as the winner and published as a single-entry `WeldLineArray`.

---

## 8. Stage 5: 2D → 3D Reconstruction — `depth_matcher`

**File:** `parol6_vision/depth_matcher.py`  
**Node name:** `depth_matcher`

### Purpose

This node bridges the 2D computer vision world and the 3D robot space. It takes the detected 2D pixel-space weld lines and **back-projects them into 3D Cartesian coordinates** using depth camera data and the robot's TF tree.

### Algorithm

**Step 1 — Cache-Based Acquisition (no timestamp sync):**  
Depth image and camera info are stored on arrival into `_cached_depth` / `_cached_info`. The `/vision/weld_lines_2d` subscriber triggers processing immediately using those caches. This replaces an older `ApproximateTimeSynchronizer` approach, which failed because the user draws the weld line *minutes after* the depth frame was captured — so timestamps never matched.

A **0.5 s rate-limit gate** in `_on_lines()` ensures the continuous stream of `weld_lines_2d` (fired at camera frame rate by `manual_line_node`) generates at most 2 depth-projection calls per second, preventing a downstream message storm.

**Step 2 — Camera Intrinsics Parsing:**  
From `CameraInfo.K` (the 3×3 intrinsic matrix):
```
K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
```

**Step 3 — Pinhole Back-Projection:**  
For each pixel `(u, v)` in a detected weld line:
```
Z = depth_image[v, u] / 1000.0     # Convert mm → meters
X = (u − cx) × Z / fx              # Camera-frame X
Y = (v − cy) × Z / fy              # Camera-frame Y
```
This produces a 3D point in the camera's optical frame.

**Step 4 — TF2 Coordinate Transform:**  
Each 3D point is transformed from the camera optical frame to `base_link` using `tf2_geometry_msgs.do_transform_point()`. A `DELETEALL` marker is published before each new marker set to prevent stale RViz accumulation.

**Step 5 — Statistical Outlier Filtering:**  
Points that deviate more than `outlier_std_threshold` (default: 2σ) standard deviations from the centroid distance are discarded. This removes depth sensor noise (common with time-of-flight cameras like Kinect).

**Step 6 — Quality Gating:**  
A line is accepted only if:
- `len(filtered_points) >= min_valid_points` (default: 10)
- `depth_quality = valid_count / total_count >= min_depth_quality` (default: 0.6)

### Topic Interface

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Subscribed | `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | VOLATILE |
| Subscribed | `/vision/captured_image_depth` | `sensor_msgs/Image` | **TRANSIENT_LOCAL** |
| Subscribed | `/vision/captured_camera_info` | `sensor_msgs/CameraInfo` | **TRANSIENT_LOCAL** |
| Published | `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | VOLATILE |
| Published | `/depth_matcher/markers` | `visualization_msgs/MarkerArray` | default |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_frame` | `base_link` | Target TF frame for 3D points |
| `depth_scale` | `1.0` | Depth scaling factor |
| `min_depth` | `300.0` mm | Minimum valid depth |
| `max_depth` | `2000.0` mm | Maximum valid depth |
| `outlier_std_threshold` | `2.0` | σ-multiplier for outlier rejection |
| `min_valid_points` | `10` | Minimum points per accepted line |
| `min_depth_quality` | `0.6` | Minimum ratio of valid-depth pixels |
| `depth_topic` | `/kinect2/sd/image_depth_rect` | Override depth source |
| `camera_info_topic` | `/kinect2/sd/camera_info` | Override info source |

---

## 9. Stage 6: Path Generation — `path_generator`

**File:** `parol6_vision/path_generator.py`  
**Node name:** `path_generator`

### Purpose

Converts the raw 3D point cloud of the weld seam into a smooth, kinematically feasible **welding trajectory** (`nav_msgs/Path`) with full 6-DOF pose at each waypoint, then publishes it to `path_holder` which acts as the sole authoritative latch for downstream consumers.

### Algorithm

**Step 1 — PCA Point Ordering:**  
The incoming 3D points may be unordered or noisy. PCA is used on the 3D point cloud to find the principal axis of the weld seam, then points are sorted along that axis.

**Step 2 — Duplicate Removal:**  
Points within 0.1 mm of each other are deduplicated to prevent spline fitting failures.

**Step 3 — Cubic B-Spline Fitting:**  
`scipy.interpolate.splprep()` fits a **cubic B-spline** (degree=3) through the ordered 3D points. The smoothing parameter `s` (default: 0.005, i.e., 5 mm variance) controls the trade-off between fitting accuracy and curve smoothness.

> *"To mitigate sensor noise and ensure kinematic smoothness, raw 3D points are fitted with a cubic B-spline. The curve is then re-parameterized by arc length to generate equidistant waypoints, critical for maintaining constant heat input during welding."*

**Step 4 — Arc-Length Resampling with Waypoint Cap:**  
The spline is evaluated at 10× the number of original points. Arc length is computed numerically. Waypoints are re-sampled at **fixed Euclidean distance intervals** (default: 5 mm). If the resulting count exceeds `max_waypoints` (default: 80), the path is **dynamically downsampled** to avoid OMPL planning failures from extreme path complexity.

**Step 5 — Orientation Generation:**  
Every waypoint is assigned a **fixed downward-facing quaternion** `(x=0.7071068, z=−0.7071068)`, which corresponds to the end-effector pointing straight down (−Z in `base_link`). This was confirmed by forward kinematics at the home position and is the correct orientation for welding on horizontal surfaces. The `approach_angle_deg` parameter is declared but does not alter this fixed quaternion in the current implementation.

**Rate Limiting:**  
A 0.5 s gate in `callback()` prevents the continuous stream of `weld_lines_3d` (fired at camera rate when the manual GUI is active) from flooding `path_holder` and `moveit_controller`.

### Topic Interface

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Subscribed | `/vision/weld_lines_3d` | `parol6_msgs/WeldLine3DArray` | VOLATILE |
| Published | `/vision/welding_path/generated` | `nav_msgs/Path` | **TRANSIENT_LOCAL** |
| Published | `/path_generator/markers` | `visualization_msgs/MarkerArray` | default |

> **Note:** `path_generator` no longer publishes directly to `/vision/welding_path`. It publishes to `/vision/welding_path/generated` and `path_holder` is the sole publisher of `/vision/welding_path`.

### Service

| Service | Type | Description |
|---------|------|-------------|
| `~/trigger_path_generation` | `std_srvs/Trigger` | Manually trigger re-generation from latest buffered data |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spline_degree` | `3` | B-spline degree (cubic) |
| `spline_smoothing` | `0.005` | Smoothing factor `s` |
| `waypoint_spacing` | `0.005` m | Distance between consecutive waypoints |
| `max_waypoints` | `80` | Hard cap; oversized paths are downsampled |
| `approach_angle_deg` | `45.0` | Declared parameter (orientation currently fixed) |
| `min_points_for_path` | `5` | Minimum points required for spline |

---

## 10. Stage 7: Path Holding — `path_holder` and `inject_path_node`

**Files:** `parol6_vision/path_holder.py`, `parol6_vision/inject_path_node.py`  
**Node names:** `path_holder`, `inject_path_node`

### Purpose

`path_holder` is the **sole authoritative publisher** of `/vision/welding_path`. It acts as a multiplexer between two staging topics, re-latches the active path with `TRANSIENT_LOCAL` durability, and exposes a service so the GUI can force-republish the cached path to any late-joining `moveit_controller`.

`inject_path_node` allows the GUI to bypass the camera pipeline by publishing a path directly to `/vision/welding_path/injected` (TRANSIENT_LOCAL), which `path_holder` then re-latches.

### Topic Interface

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Subscribed | `/vision/welding_path/generated` | `nav_msgs/Path` | TRANSIENT_LOCAL |
| Subscribed | `/vision/welding_path/injected` | `nav_msgs/Path` | TRANSIENT_LOCAL |
| Published | `/vision/welding_path` | `nav_msgs/Path` | **TRANSIENT_LOCAL** |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/path_holder/set_source` | `std_srvs/Trigger` | Force-republish the currently cached path |
| `/path_holder/get_status` | `std_srvs/Trigger` | Query current source and path availability |

---

## 11. Stage 8: Motion Execution — `moveit_controller`

**File:** `parol6_vision/moveit_controller.py`  
**Node name:** `moveit_controller`

### Purpose

Executes the generated `nav_msgs/Path` welding trajectory using **MoveIt2**. Implements a robust 3-tier fallback strategy to handle the inherent brittleness of Cartesian path planning in constrained robotic environments. Subscribes to `/vision/welding_path` with `TRANSIENT_LOCAL` QoS to receive the latched path from `path_holder` even when starting after the path was published.

### Execution Sequence

All three phases are **planned first, then executed back-to-back** to eliminate gaps between approach and weld motion:

```
[Plan 1/3] Home trajectory       (joint-space, plan only)
[Plan 2/3] Approach trajectory   (pose goal, 15 cm above weld start, plan only)
[Plan 3/3] Cartesian weld traj   (Cartesian, with 3-tier fallback, plan only)
                 ↓ all plans ready
[Exec 1/3] Move to Home
[Exec 2/3] Move to Approach Point
[Exec 3/3] Execute Weld
```

Path offset parameters (`path_offset_x/y/z`) are applied in `path_callback` to every incoming waypoint, allowing ±50 mm weld position correction without re-scanning.

### Cartesian Planning Fallback Strategy

Cartesian path planning can fail due to kinematic singularities, joint limits, or collision avoidance constraints. The node implements a **3-tier fallback**:

| Attempt | Step Size | Success Threshold | Description |
|---------|-----------|-------------------|-------------|
| 1 | 2 mm | 95% | High precision welding |
| 2 | 5 mm | 95% | Relaxed step to skip micro-singularities |
| 3 | 10 mm | 90% | Coarse "get the job done" mode |

If all Cartesian attempts fail and `enable_joint_waypoint_fallback` is `True`, the node defaults to continuous **joint-space execution to a coarse subset of waypoints** (`joint_waypoint_fallback_count`, default 8). The controller pre-plans collision-free movements sequentially across the sampled waypoints and automatically stitches them into a single uninterrupted `RobotTrajectory`. Boundary velocities and accelerations are dynamically stripped while timestamps are scaled precisely to ensure a smooth continuous quintic/cubic spline motion matching the intended `weld_velocity`.

> *"To overcome the inherent brittleness of Cartesian trajectory generation in constrained environments, a hierarchical fallback strategy was implemented. This dynamically adjusts the discretization resolution (2mm to 10mm), prioritizing geometric fidelity while ensuring system reliability."*

### Topic Interface

| Direction | Topic | Type | QoS |
|-----------|-------|------|-----|
| Subscribed | `/vision/welding_path` | `nav_msgs/Path` | **TRANSIENT_LOCAL** |

### Services

| Service | Description |
|---------|-------------|
| `~/execute_welding_path` | Manually trigger path execution |
| `~/is_execution_idle` | Query execution state (idle + path available?) |

### MoveIt2 Interfaces

- **`compute_cartesian_path`** service (`moveit_msgs/GetCartesianPath`) — Cartesian planning
- **`execute_trajectory`** action (`moveit_msgs/ExecuteTrajectory`) — Trajectory execution
- **`move_action`** action (`moveit_msgs/MoveGroup`) — Joint-space approach moves

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `planning_group` | `parol6_arm` | MoveIt planning group |
| `base_frame` | `base_link` | Robot base frame |
| `end_effector_link` | `tcp_link` | End-effector link name |
| `cartesian_step_sizes` | `[0.002, 0.005, 0.010]` | Fallback step sizes |
| `min_success_rates` | `[0.95, 0.95, 0.90]` | Fallback success thresholds |
| `approach_distance` | `0.15` m | Pre-weld approach offset (15 cm keeps approach above z=0.10 m workspace floor) |
| `weld_velocity` | `0.01` m/s | Target welding speed |
| `path_offset_x/y/z` | `0.0` m | Live path offset correction (±50 mm range) |
| `enable_joint_waypoint_fallback` | `True` | Enable coarse joint-space fallback |
| `auto_execute` | `False` | Auto-execute on path receipt |

---

## 12. Complete Topic & Message Flow Diagram

```
   /kinect2/sd/image_color_rect ──┐
   /kinect2/sd/image_depth_rect ──┤──► capture_images_node ──► /vision/captured_image_raw      [VOLATILE]
   /kinect2/sd/camera_info      ──┘                       ├──► /vision/captured_image_depth     [TRANSIENT_LOCAL]
                                                           └──► /vision/captured_camera_info     [TRANSIENT_LOCAL]

   /vision/captured_image_raw ──► crop_image_node ──► /vision/captured_image_color

   /vision/captured_image_color ──► [color_mode OR yolo_segment OR manual_line] ──► /vision/processing_mode/annotated_image
                                                                                  ├──► /vision/processing_mode/debug_image
                                                                                  └──► /vision/processing_mode/seam_centroid

   /vision/processing_mode/annotated_image ──► [red_line_detector OR path_optimizer] ──► /vision/weld_lines_2d

   /vision/weld_lines_2d        ──► depth_matcher [0.5s gate] ──► /vision/weld_lines_3d
   /vision/captured_image_depth ──► (cached → TRANSIENT_LOCAL)  └──► /depth_matcher/markers
   /vision/captured_camera_info ──► (cached → TRANSIENT_LOCAL)

   /vision/weld_lines_3d ──► path_generator [0.5s gate] ──► /vision/welding_path/generated  [TRANSIENT_LOCAL]
                                                          └──► /path_generator/markers

   /vision/welding_path/generated ──┐
   /vision/welding_path/injected  ──┤──► path_holder ──► /vision/welding_path  [TRANSIENT_LOCAL]
   (from inject_path_node)        ──┘

   /vision/welding_path ──► moveit_controller ──► [Robot Execution via MoveIt2]
                           [TRANSIENT_LOCAL sub]
```

---

## 13. Key Custom Messages (`parol6_msgs`)

The pipeline uses two custom message types defined in the `parol6_msgs` package:

### `parol6_msgs/WeldLine`
Represents a single detected 2D weld line:
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `red_line_0`) |
| `confidence` | float32 | Detection quality score ∈ [0, 1] |
| `pixels` | geometry_msgs/Point32[] | Dense ordered pixel coordinates (u, v) |
| `bbox_min` | geometry_msgs/Point | Bounding box minimum corner |
| `bbox_max` | geometry_msgs/Point | Bounding box maximum corner |

### `parol6_msgs/WeldLineArray`
Array wrapper for multiple weld lines, with a `std_msgs/Header`.

### `parol6_msgs/WeldLine3D`
Represents a single 3D-reconstructed weld line:
| Field | Type | Description |
|-------|------|-------------|
| `id` | int32 | Source line ID |
| `confidence` | float32 | Inherited from 2D detection |
| `points` | geometry_msgs/Point[] | 3D points in `base_link` frame |
| `depth_quality` | float32 | Ratio of valid depth samples |
| `num_points` | int32 | Number of 3D points |
| `line_width` | float32 | Estimated line width (default: 3 mm) |
| `header` | std_msgs/Header | Frame and timestamp |

### `parol6_msgs/WeldLine3DArray`
Array wrapper for multiple 3D weld lines, with a `std_msgs/Header`.

---

## 14. Launch Files Summary

| File | Description |
|------|-------------|
| `live_pipeline.launch.py` | Full live pipeline with Kinect camera and all stages |
| `vision_moveit.launch.py` | Full pipeline including MoveIt2 execution |
| `capture_and_replay.launch.py` | Offline replay using `read_image_node` from disk |
| `vision_pipeline.launch.py` | Vision-only pipeline (no robot execution) |
| `camera_setup.launch.py` | Camera bringup only |
| `test_depth_matcher_bag.launch.py` | Test depth matching from rosbag |
| `test_path_generator_bag.launch.py` | Test path generation from rosbag |
| `red_detector_only.launch.py` | Run only the red line detector |
| `test_integration.launch.py` | Integration testing launch |

---

## 15. Conclusion

The PAROL6 vision pipeline achieves a fully automated, end-to-end solution for vision-guided welding:

1. **Synchronized acquisition** captures a temporally aligned RGB+Depth pair on demand; depth and camera info are latched with `TRANSIENT_LOCAL` QoS so late-starting nodes receive the last captured frame immediately.
2. **ROI masking** focuses processing on the relevant workspace region.
3. **Three-mode seam detection** (HSV color, YOLO segmentation, or manual stroke overlay) provides flexibility for different workpiece types, lighting conditions, and operational requirements. The manual mode additionally supports persistent stroke replay for fixed-fixture repeat jobs.
4. **Red line detection** extracts the operator-defined weld path using HSV segmentation, skeletonization, PCA ordering, and Douglas-Peucker simplification.
5. **Cache-based 3D back-projection** with outlier filtering reconstructs the weld path in robot space without requiring timestamp-synchronization between capture and annotation events.
6. **B-spline smoothing** with a dynamic waypoint cap eliminates sensor noise and generates kinematically smooth, OMPL-compatible trajectories with constant waypoint spacing.
7. **Path holding** decouples generation from execution: `path_holder` re-latches the path so `moveit_controller` always receives a valid path regardless of startup order.
8. **Hierarchical Cartesian planning** with a joint-space fallback ensures reliable execution even in constrained workspaces.

The modular ROS 2 architecture allows any stage to be independently tested, replaced, or extended without disrupting the rest of the pipeline. Topic-based communication and standardized message types ensure loose coupling between nodes.

---

*Document generated from source code analysis of `parol6_vision` package.*
