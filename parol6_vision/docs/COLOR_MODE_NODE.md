# color_mode Node Documentation

## Overview

`color_mode.py` implements the **`color_mode`** ROS 2 node, which performs **HSV colour-based seam intersection detection** between two coloured workpieces (green and blue). It is a ROS 2 port of `color_mode.py`.

The core idea is: given a camera frame, apply HSV colour thresholding to isolate the green and blue workpieces, expand each colour mask outward, compute the **overlap (intersection) region** — which represents the **weld seam** — and publish the result as annotated images and a centroid point.

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
| `image_topic` | `string` | `/vision/captured_image_color` | Input camera topic to subscribe to |
| `expand_px` | `int` | `2` | Number of pixels to dilate each colour mask outward before computing intersection |
| `publish_debug` | `bool` | `True` | Whether to publish the full debug overlay image |

---

## HSV Colour Ranges

The node uses the following fixed HSV thresholds (OpenCV convention: H 0–180, S 0–255, V 0–255):

| Colour | Lower Bound (H, S, V) | Upper Bound (H, S, V) |
|--------|-----------------------|-----------------------|
| **Green** | `[35, 50, 50]` | `[100, 255, 255]` |
| **Blue**  | `[100, 50, 50]` | `[140, 255, 255]` |

---

## Algorithm Pipeline

The full processing sequence runs inside `_run_pipeline()` on every incoming frame.

```
Input Frame
    │
    ▼
1. HSV Conversion
    │  Convert BGR frame → HSV colour space
    ▼
2. Mask Creation
    │  Green mask (G): pixels within [35,50,50]–[100,255,255]
    │  Blue  mask (B): pixels within [100,50,50]–[140,255,255]
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

#### Step 1 — HSV Conversion
The incoming BGR frame is converted to the HSV colour space using `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`. HSV separation of hue from brightness makes colour thresholding more robust to lighting changes than working in BGR directly.

#### Step 2 — Mask Creation
Two binary masks are created using `cv2.inRange()`:
- **Green mask (G)**: retains pixels whose HSV values fall within `[35, 50, 50]` → `[100, 255, 255]`.
- **Blue mask (B)**: retains pixels whose HSV values fall within `[100, 50, 50]` → `[140, 255, 255]`.

#### Step 3 — Morphological Opening
A 5×5 square kernel is used with `cv2.MORPH_OPEN` (erode then dilate). This removes small isolated noise pixels from each binary mask without significantly shrinking the main object regions.

#### Step 4 — Mask Dilation (Seam Expansion)
Each cleaned mask is dilated outward by `expand_px` pixels using an **elliptical** structuring element of size `(2*expand_px+1) × (2*expand_px+1)`. This widens each object's footprint so that the narrow gap between adjacent workpieces — the seam — is captured in the intersection.

Pre-built kernels (`_morph_kernel`, `_dil_kernel`) are constructed once at initialisation and reused every frame for efficiency.

#### Step 5 — Intersection
```python
intersection_mask = cv2.bitwise_and(G_exp, B_exp)
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

### `ColorModeNode` (inherits `rclpy.node.Node`)

| Method | Description |
|--------|-------------|
| `__init__()` | Declares and reads parameters, pre-builds morphological kernels, creates publishers and subscriber |
| `_image_callback(msg)` | ROS subscriber callback — converts image, calls `_run_pipeline`, publishes results |
| `_run_pipeline(img)` | Core algorithm; returns `(annotated_img, debug_img, centroid_px)` |
| `_find_largest_contour(mask)` | Static helper — returns the largest external contour from a binary mask using `cv2.findContours` |
| `_draw_crosshair(img, cx, cy, color, size)` | Static helper — draws a crosshair at pixel `(cx, cy)` (currently unused/commented out) |

### `main()`
Standard ROS 2 entry point. Initialises `rclpy`, spins the node, and on shutdown logs a summary:
```
Shutting down color_mode.
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
| `opencv-python` (`cv2`) | Image processing (HSV conversion, morphology, dilation, contours, moments, drawing) |
| `numpy` | Array operations and HSV threshold arrays |

---

## Notes & Caveats

- **Fixed colour ranges**: The HSV thresholds for green and blue are hardcoded as class-level constants (`_LOWER_GREEN`, `_UPPER_GREEN`, `_LOWER_BLUE`, `_UPPER_BLUE`). They are not exposed as ROS parameters — adjust them directly in the source if the workpiece colours change.
- **Single-intersection assumption**: The pipeline finds only the **largest** contour of the intersection mask. If the two workpieces produce multiple disjoint intersection regions, only the biggest one is used.
- **`expand_px` tuning**: A small value (2–5 px) is usually sufficient. Too large a value can cause false intersections where objects are not actually touching.
- **No minimum area filter**: Any non-empty intersection contour is accepted as a valid seam detection, regardless of its size. Very small intersections may generate noisy centroids.
- **Commented-out code**: Centroid crosshair drawing on both `annotated_img` and `debug_img` is preserved as commented-out code for easy re-enabling.
- **Lighting sensitivity**: While HSV is more robust than BGR for colour thresholding, significant ambient light changes may cause the fixed thresholds to miss or over-segment the workpieces. Consider adding dynamic parameter reconfiguration or exposure control if this is an issue.
