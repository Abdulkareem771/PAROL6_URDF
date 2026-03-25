# Yolo_segment Node Documentation

## Overview

`yolo_segment.py` implements the **`Yolo_segment`** ROS 2 node, which performs **YOLO-based instance segmentation** to detect weld seams between two workpieces. It is a ROS 2 port of `phase_2_first_mode.py`.

The core idea is: given a camera frame, run YOLO segmentation to find two workpiece objects, expand their masks outward, compute the **overlap (intersection) region** — which represents the **weld seam** — and publish the result as annotated images and a centroid point.

---

## ROS 2 Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/captured_image_color` | `sensor_msgs/Image` | Input colour image from camera (configurable via `image_topic` parameter) |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` (bgr8) | Clean frame with only the filled intersection contour drawn in red |
| `/vision/processing_mode/debug_image` | `sensor_msgs/Image` (bgr8) | Full debug overlay showing all intermediate contours |
| `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` | Pixel-space centroid of the detected weld seam intersection |

> **Note:** `debug_image` is only published when the `publish_debug` parameter is `True`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `string` | `/workspace/venvs/vision_venvs/ultralytics_cpu_env/yolo_segmentation_models_results/experiment_2/weights/best.pt` | Absolute path to YOLO weights file (`best.pt`) |
| `image_topic` | `string` | `/vision/captured_image_color` | Input camera topic to subscribe to |
| `expand_px` | `int` | `2` | Number of pixels to dilate each object mask outward before computing intersection |
| `publish_debug` | `bool` | `True` | Whether to publish the full debug overlay image |
| `mask_conf` | `float` | `0.85` | Minimum YOLO detection confidence threshold |
| `print_detections` | `bool` | `True` | Print detected object count per frame to the logger |

---

## Algorithm Pipeline

The full processing sequence runs inside `_run_pipeline()` on every incoming frame.

```
Input Frame
    │
    ▼
1. YOLO Inference
    │  Run model on frame with confidence threshold (mask_conf)
    │  Take the first two detected masks
    ▼
2. Mask Extraction & Binarisation
    │  Resize each mask to the original image resolution
    │  Threshold: pixel > mask_conf → 255, else → 0
    ▼
3. Morphological OPEN (5×5 kernel)
    │  Removes small noise blobs from each binary mask
    ▼
4. Dilation / Seam Expansion
    │  Dilate each mask outward by expand_px pixels
    │  Uses an elliptical structuring element
    ▼
5. Intersection (Bitwise AND)
    │  AND the two dilated masks → overlap region = weld seam
    │  Find largest contour in intersection
    ▼
6. Centroid Computation (image moments)
    │  cx = m10/m00,  cy = m01/m00
    ▼
7. Publish
       ├─ annotated_image : raw frame + filled red intersection
       ├─ debug_image     : raw frame + blue originals + green expanded + red intersection
       └─ seam_centroid   : PointStamped (x=cx, y=cy, z=0)
```

### Step-by-step Explanation

#### Step 1 — YOLO Inference
`ultralytics.YOLO` runs inference on the incoming BGR frame. The `conf=mask_conf` argument filters out any detections below the confidence threshold. The result contains segmentation masks and bounding boxes.

#### Step 2 — Mask Extraction & Binarisation
Only the **first two** detected objects are used (the two workpieces). Each raw mask (a float tensor from YOLO) is:
1. Transferred to CPU as a NumPy array.
2. Resized to match the original frame dimensions using `cv2.resize`.
3. Binarised: pixels with value > `mask_conf` become `255`; all others become `0`.

If fewer than two masks are detected, the pipeline returns unmodified frames and no centroid.

#### Step 3 — Morphological Opening
A 5×5 square kernel is used with `cv2.MORPH_OPEN` (erode then dilate). This removes small isolated noise pixels from each binary mask without significantly shrinking the main object regions.

#### Step 4 — Mask Dilation (Seam Expansion)
Each cleaned mask is dilated outward by `expand_px` pixels using an **elliptical** structuring element of size `(2*expand_px+1) × (2*expand_px+1)`. This widens each object's footprint so that the gap between adjacent workpieces — the seam — is captured in the intersection.

Pre-built kernels (`_morph_kernel`, `_dil_kernel`) are constructed once at initialisation and reused every frame for efficiency.

#### Step 5 — Intersection
```python
intersection_mask = cv2.bitwise_and(obj_1_exp, obj_2_exp)
```
Pixels that are `255` in **both** dilated masks form the intersection region. The largest external contour of this region is extracted with `_find_largest_contour()`.

#### Step 6 — Centroid Computation
Image moments of the intersection contour are calculated:
```
cx = M['m10'] / M['m00']
cy = M['m01'] / M['m00']
```
This gives the **area-weighted centroid** of the intersection polygon in pixel coordinates.

#### Step 7 — Publishing
| Output | Content |
|--------|---------|
| `annotated_image` | Copy of raw frame with the intersection contour **filled solid red** |
| `debug_image` | Copy of raw frame with: original contours in **blue** (2px), expanded contours in **green** (2px), intersection **filled red** |
| `seam_centroid` | `PointStamped` — `point.x = cx`, `point.y = cy`, `point.z = 0.0` |

All published messages carry the same `header` (timestamp + frame ID) as the incoming image message.

---

## Class & Method Reference

### `YoloSegmentNode` (inherits `rclpy.node.Node`)

| Method | Description |
|--------|-------------|
| `__init__()` | Declares and reads parameters, loads YOLO model, pre-builds kernels, creates publishers and subscriber |
| `_image_callback(msg)` | ROS subscriber callback — converts image, calls `_run_pipeline`, publishes results |
| `_run_pipeline(img)` | Core algorithm; returns `(annotated_img, debug_img, centroid_px, num_detections)` |
| `_find_largest_contour(mask)` | Static helper — returns the largest external contour from a binary mask using `cv2.findContours` |
| `_draw_crosshair(img, cx, cy, color, size)` | Static helper — draws a crosshair at pixel `(cx, cy)` (currently unused/commented out) |

### `main()`
Standard ROS 2 entry point. Initialises `rclpy`, spins the node, and on shutdown logs a summary:
```
Shutting down Yolo_segment.
Processed N frames, found intersections in M frames (X%).
```

---

## Internal Counters

| Counter | Purpose |
|---------|---------|
| `_frame_count` | Incremented on every received frame |
| `_detection_count` | Incremented only when a valid seam intersection is found and published |

These are used to compute the **intersection success rate** logged at shutdown.

---

## Dependencies

| Library | Usage |
|---------|-------|
| `rclpy` | ROS 2 Python client library |
| `sensor_msgs/Image` | ROS image message type |
| `geometry_msgs/PointStamped` | ROS point message for centroid |
| `cv_bridge` | Converts between ROS `Image` messages and OpenCV arrays |
| `opencv-python` (`cv2`) | Image processing (resize, morphology, dilation, contours, moments, drawing) |
| `numpy` | Array operations and mask manipulation |
| `ultralytics` (YOLO) | Instance segmentation model inference |

---

## Notes & Caveats

- **Two-object assumption**: The pipeline always takes exactly the **first two** YOLO detections. If fewer than 2 objects are found above `mask_conf`, no centroid is published and a `WARN` log is emitted.
- **Confidence threshold dual use**: `mask_conf` is used both as the YOLO inference confidence filter (`model(..., conf=mask_conf)`) and as the per-pixel binarisation threshold for the mask tensors. Increasing it makes detections stricter and masks tighter.
- **`expand_px` tuning**: A small value (2–5 px) is usually sufficient. Too large a value can cause false intersections where objects are not actually touching.
- **Commented-out code**: Several blocks are left commented (centroid crosshair/circle drawing, BGR→RGB conversion, real-time detection printing). These are preserved as optional features for easy re-enabling.
- **Docker path**: The default `model_path` is hardcoded to a path inside the project's Docker container. Override this parameter at launch if running outside the container.
