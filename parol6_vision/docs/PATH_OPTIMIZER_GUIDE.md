# Path Optimizer Node - Developer Guide

## 1. Overview

The `path_optimizer` node is a vision-based ROS 2 node in the PAROL6 robot system. It detects red weld-line markers in camera images and extracts their 2D geometry as a centerline skeleton. It is designed specifically to handle weld lines of **variant lengths** — there is no minimum-length filter, so both short and long lines are detected equally.

The node publishes exactly **one line per frame** — the contour with the highest confidence score — making its output unambiguous for downstream consumers.

**Node Name:** `path_optimizer`  
**Package:** `parol6_vision`  
**Source:** `parol6_vision/path_optimizer.py`

---

## 2. Architecture & Pipeline

The detection pipeline executes sequentially on every incoming image frame.

```
/vision/processing_mode/annotated_image  (sensor_msgs/Image)
            │
            ▼
  ┌─────────────────────┐
  │  1. HSV Segmentation │  BGR → HSV, dual red-range masking
  └──────────┬──────────┘
             │  binary red mask
             ▼
  ┌──────────────────────────┐
  │  2. Morphological Cleanup │  Erosion → salt noise removal
  │                           │  Dilation → fragment connection
  └──────────┬───────────────┘
             │  clean binary mask
             ▼
  ┌────────────────────┐
  │  3. Skeletonization │  thick mask → 1-pixel-wide centerline
  └──────────┬─────────┘
             │  skeleton image
             ▼
  ┌─────────────────────────────┐
  │  4. Contour Extraction       │  all contours accepted (no min-length)
  └──────────┬──────────────────┘
             │  list of (N,2) point arrays
             ▼
  ┌───────────────────────────────────────────────────────────┐
  │  5. Per-Contour Processing                                 │
  │     a) PCA-based Point Ordering  (start → end)            │
  │     b) Douglas-Peucker Simplification                     │
  │     c) Confidence Scoring  (retention × continuity)       │
  └──────────┬────────────────────────────────────────────────┘
             │  scored WeldLine candidates
             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  6. Best-Line Selection                                   │
  │     Pick the single highest-confidence line ≥ threshold  │
  └──────────┬───────────────────────────────────────────────┘
             │
       ┌─────┴──────────────────────────┐
       ▼                                ▼
/vision/weld_lines_2d        /path_optimizer/debug_image
(WeldLineArray, 0 or 1 line) /path_optimizer/markers
```

---

## 3. Detailed Stage Explanations

### Stage 1 — HSV Color Segmentation

**Method:** `segment_red_color(image)`

Red is unique in the OpenCV HSV color space (H ∈ [0, 180]): it wraps around at 0° and 180°, meaning two separate `cv2.inRange()` calls are needed to capture the full red spectrum.

| Range | Hue | Captures |
|---|---|---|
| Range 1 (`hsv_lower_1 / hsv_upper_1`) | 0–10° | Pure red, orange-red |
| Range 2 (`hsv_lower_2 / hsv_upper_2`) | 160–180° | Purple-red, crimson |

Both binary masks are combined with `cv2.bitwise_or()`, producing a single mask where white pixels (255) indicate red regions and black (0) indicates everything else.

```python
hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, hsv_lower_1, hsv_upper_1)   # low red
mask2 = cv2.inRange(hsv, hsv_lower_2, hsv_upper_2)   # high red
mask  = cv2.bitwise_or(mask1, mask2)
```

---

### Stage 2 — Morphological Processing

**Method:** `apply_morphology(mask)`

The raw color mask after segmentation typically contains:
- **Salt noise** — isolated white pixels from reflections or color noise
- **Fragmented segments** — small gaps breaking the line into disconnected pieces

Two morphological operations are applied in sequence using a rectangular structuring element of size `morphology_kernel_size × morphology_kernel_size`:

#### 2a. Erosion
```python
cv2.erode(mask, kernel, iterations=erosion_iterations)
```
- Shrinks white regions by removing pixels at their boundaries
- Eliminates small isolated noise blobs that are smaller than the kernel
- May thin or disconnect narrow line features (corrected by next step)

#### 2b. Dilation
```python
cv2.dilate(mask, kernel, iterations=dilation_iterations)
```
- Expands white regions outward
- Fills small holes and closes narrow gaps within the line
- Reconnects nearby fragments that were disconnected during erosion

**Net result:** An `open` morphology that removes noise while preserving and strengthening the main line structure.

---

### Stage 3 — Skeletonization

**Method:** `skeletonize(mask)`

The morphological mask is still a thick blob (several pixels wide). For precise line localization, the mask is reduced to a **1-pixel-wide centerline** using `skimage.morphology.skeletonize`.

```python
skeleton_bool = skimage.morphology.skeletonize(mask > 0)
skeleton      = (skeleton_bool * 255).astype(np.uint8)
```

**Properties of the skeleton:**
- **1-pixel wide** at every point — no ambiguity about the line's position
- **Topologically equivalent** to the original mask — connectivity and branching are preserved
- **Centerline accurate** — the skeleton follows the medial axis of the thick marker

This is more reliable than edge detection or Hough transforms because it handles curves, variable widths, and partial occlusions gracefully.

---

### Stage 4 — Contour Extraction

**Method:** `extract_contours(skeleton)`

`cv2.findContours` is used on the skeleton image to retrieve connected components, each representing a candidate line segment.

```python
raw_contours, _ = cv2.findContours(
    skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
)
```

> **Key design decision:** Unlike `red_line_detector`, **no minimum length filter is applied here.** Every contour — regardless of how few points it contains — is passed to the scoring stage. This allows the node to handle short weld-line segments that would otherwise be discarded.

Each contour is reshaped from OpenCV's `(N, 1, 2)` format to a simpler `(N, 2)` array of `(x, y)` pixel coordinates.

---

### Stage 5a — PCA-based Point Ordering

**Method:** `order_points_along_line(points)`

Contour points extracted by `cv2.findContours` are not guaranteed to be in spatial order along the line. For weld-path planning, points must progress continuously from one endpoint to the other.

**Principal Component Analysis (PCA)** is used to find the primary direction of the point cloud:

1. Fit a 1-component PCA model to the `(N, 2)` point array
2. Project each point onto the first principal component (the dominant direction)
3. Sort points by their projection value — smallest projection = one end, largest = other end

```python
pca        = PCA(n_components=1)
projections = pca.fit_transform(points).flatten()
ordered    = points[np.argsort(projections)]
```

This works correctly for straight lines, curved lines, and diagonal lines regardless of how the skeleton pixels were visited by `findContours`.

---

### Stage 5b — Douglas-Peucker Simplification

**Method:** `simplify_polyline(points)`

The ordered skeleton can have hundreds of closely-spaced points. The **Douglas-Peucker algorithm** reduces this to a minimal set of points that still accurately represents the line's shape, controlled by `douglas_peucker_epsilon` (tolerance in pixels).

```python
simplified = cv2.approxPolyDP(points_cv, epsilon=dp_epsilon, closed=False)
```

- **Low epsilon** → more points retained → more shape detail
- **High epsilon** → fewer points → smoother approximation

The simplified points are used only for **confidence scoring**. The full dense ordered skeleton points are stored in `WeldLine.pixels` for downstream 3-D reconstruction.

---

### Stage 5c — Confidence Scoring

**Method:** `compute_continuity(simplified_points)`

Each candidate line is assigned a **confidence score** ∈ [0, 1] composed of two factors:

#### Retention Ratio
```
retention = pixels_after_morphology / pixels_before_morphology
```
Measures how much of the original red signal survived the morphological cleanup:
- High retention → dense, solid line → reliable detection
- Low retention → line was mostly noise → less reliable

#### Continuity Score
Measures the geometric smoothness of the simplified line by computing angle variance between consecutive segments:

```python
vectors     = np.diff(simplified_points, axis=0)
angles      = np.arctan2(vectors[:, 1], vectors[:, 0])
angle_diffs = np.abs(np.diff(angles))
variance    = np.var(angle_diffs)
continuity  = np.exp(-variance * 5.0)   # ∈ [0, 1]
```

- Low angle variance → smooth, continuous line → continuity near 1.0
- High angle variance → jagged, fragmented line → continuity near 0.0

#### Final Confidence
```
confidence = retention × continuity
```

---

### Stage 6 — Best-Line Selection

After scoring all contours, the node selects the **single contour with the highest confidence** that meets or exceeds `min_confidence`:

```python
if confidence >= min_confidence and confidence > best_confidence:
    best_confidence = confidence
    best_line = <WeldLine built from this contour>
```

If no contour meets the threshold, an empty `WeldLineArray` (zero lines) is published that frame.

---

## 4. Output Message Format

### `/vision/weld_lines_2d` — `parol6_msgs/WeldLineArray`

The array will contain **0 or 1** `WeldLine` message per frame.

Each `WeldLine` contains:

| Field | Content |
|---|---|
| `id` | `"path_optimizer_line"` |
| `confidence` | Score ∈ [0, 1] |
| `pixels` | Ordered `Point32[]` — dense skeleton points, x=col, y=row, z=0 |
| `bbox_min` | Top-left bounding box corner (image coordinates) |
| `bbox_max` | Bottom-right bounding box corner (image coordinates) |

---

## 5. ROS API

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` | Annotated image from the active processing mode node |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/weld_lines_2d` | `parol6_msgs/WeldLineArray` | Detected weld line (0 or 1 per frame) |
| `/path_optimizer/debug_image` | `sensor_msgs/Image` | Original image with detected line overlaid |
| `/path_optimizer/markers` | `visualization_msgs/MarkerArray` | `LINE_STRIP` marker for RViz |

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hsv_lower_1` | int[] | `[0, 100, 100]` | HSV lower bound — low-red range |
| `hsv_upper_1` | int[] | `[10, 255, 255]` | HSV upper bound — low-red range |
| `hsv_lower_2` | int[] | `[160, 50, 0]` | HSV lower bound — high-red range |
| `hsv_upper_2` | int[] | `[180, 255, 255]` | HSV upper bound — high-red range |
| `morphology_kernel_size` | int | `5` | Structuring element size (px) for erosion/dilation |
| `erosion_iterations` | int | `1` | Number of erosion passes |
| `dilation_iterations` | int | `2` | Number of dilation passes |
| `douglas_peucker_epsilon` | float | `2.0` | Simplification tolerance (px) |
| `min_confidence` | float | `0.5` | Minimum confidence to publish a line |
| `publish_debug_images` | bool | `True` | Enable `/path_optimizer/debug_image` and `/path_optimizer/markers` |

---

## 6. Debug Visualization

When `publish_debug_images: True`, the node overlays the detected line on the original image using a **confidence-based color scheme**:

| Confidence | Color | Meaning |
|---|---|---|
| ≥ 0.9 | 🟢 Green | Excellent detection |
| 0.7 – 0.9 | 🟡 Yellow | Good detection |
| < 0.7 | 🟠 Orange | Acceptable detection |

The debug image also shows:
- A polyline through all skeleton points
- A bounding box rectangle
- A text label: `path_optimizer_line: <confidence value>`

---

## 7. Differences vs. `red_line_detector`

| Feature | `red_line_detector` | `path_optimizer` |
|---|---|---|
| Node name | `red_line_detector` | `path_optimizer` |
| Debug topic prefix | `/red_line_detector/` | `/path_optimizer/` |
| Min line-length filter | Yes — `min_line_length`, `min_contour_area` | **None** |
| Lines per frame | Up to `max_lines_per_frame` | **Exactly 1** (best confidence) |
| Intended use | General multi-line detection | Variant-length single-line detection |

---

## 8. Build & Run

```bash
# Build (inside Docker or with ROS 2 sourced)
cd /path/to/PAROL6_URDF
colcon build --packages-select parol6_vision --symlink-install
source install/setup.bash

# Run
ros2 run parol6_vision path_optimizer

# Run with custom HSV ranges
ros2 run parol6_vision path_optimizer \
  --ros-args \
  -p hsv_lower_1:="[0, 120, 80]" \
  -p hsv_upper_1:="[8, 255, 255]" \
  -p min_confidence:=0.4
```

---

## 9. Troubleshooting

### No line detected despite visible red in image

1. Check HSV ranges — use the `hsv_inspector` tool:
   ```bash
   ros2 run parol6_vision hsv_inspector
   ```
2. Lower `min_confidence` (e.g., `0.3`) to see if a low-confidence line exists
3. Monitor `/path_optimizer/debug_image` in RViz — if the mask is empty, it is an HSV tuning issue; if the mask has pixels but no line, it is a confidence issue

### Detected line is fragmented / wrong

- Increase `dilation_iterations` (e.g., `3`) to connect gap-prone lines
- Decrease `douglas_peucker_epsilon` (e.g., `1.0`) for more precise polyline tracing
- Reduce `min_confidence` if continuity scoring is penalizing a genuinely curved line

### CV Bridge Error (`bgr8` conversion failed)

- Ensure the upstream node publishes with `encoding: bgr8` or `rgb8`
- The node always requests `bgr8`; if the source encoding differs, `cv_bridge` will convert automatically when possible

### Debug images not appearing

- Confirm `publish_debug_images` is `True` (default)
- Check `ros2 topic list` for `/path_optimizer/debug_image`
- Add the topic to RViz as an `Image` display
