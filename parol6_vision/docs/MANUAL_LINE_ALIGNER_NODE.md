# Manual Line Aligner Node — Developer Guide

## 1. Overview

The `manual_line_aligner` node is a **hybrid** processing-mode node that combines manually drawn weld-seam annotations with real-time visual tracking. When the operator teaches a reference Region of Interest (ROI), the node continuously re-aligns the strokes to the current camera frame using **ORB feature matching** and an **Affine transformation** — meaning the painted path follows the part even if it shifts or rotates slightly between runs.

It provides two operating modes selectable at runtime without a restart:

| Mode | Trigger | Behaviour |
|---|---|---|
| **Fixed** | `set_strokes` (no ROI taught) | Strokes are painted at fixed pixel positions, identical to `manual_line` |
| **Adaptive** | `teach_reference` (ROI taught) | Strokes are warped every frame to match the part's current position |

**Node Name:** `manual_line_aligner`  
**Package:** `parol6_vision`  
**Source:** `parol6_vision/manual_line_aligner_node.py`

---

## 2. Architecture & Pipeline

### Fixed Mode

```
/vision/captured_image_color
            │
            ▼
  ┌──────────────────────────────┐
  │  Decode Image (CvBridge)     │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │  Paint strokes at fixed px   │
  │  cv2.polylines() per stroke  │
  └──────┬───────────────────────┘
         │
         ▼
/vision/processing_mode/annotated_image
/vision/processing_mode/debug_image      (FIXED MODE badge)
/vision/processing_mode/seam_centroid
```

### Adaptive Mode

```
/vision/captured_image_color
            │
            ▼
  ┌───────────────────────────────────────────┐
  │  1. ORB Feature Extraction (full frame)    │
  └──────────────────┬────────────────────────┘
                     │  current keypoints + descriptors
                     ▼
  ┌───────────────────────────────────────────┐
  │  2. BFMatcher knnMatch (k=2)              │
  │     Lowe's Ratio Test  (threshold 0.75)   │
  └──────────────────┬────────────────────────┘
                     │  good_matches  (≥10 required)
                     ▼
  ┌───────────────────────────────────────────┐
  │  3. Spatial Filtering                      │
  │     Top-50 by Hamming distance            │
  │     Spatial spread check (std > 10 px)    │
  └──────────────────┬────────────────────────┘
                     │  filtered src / dst point pairs
                     ▼
  ┌───────────────────────────────────────────┐
  │  4. estimateAffinePartial2D (RANSAC)       │
  │     Inlier ratio ≥ 0.40                   │
  │     Determinant ≥ 0.1, Scale ∈ [0.5, 2.0]│
  └──────────────────┬────────────────────────┘
                     │  2×3 affine matrix
                     ▼
  ┌───────────────────────────────────────────┐
  │  5. Temporal Smoothing                     │
  │     Exponential smoothing on translation  │
  │     (α = 0.5, rotation/scale not smoothed)│
  └──────────────────┬────────────────────────┘
                     │  smoothed affine matrix
                     ▼
  ┌───────────────────────────────────────────┐
  │  6. Stroke Transformation                  │
  │     cv2.transform() per stroke polyline   │
  └──────┬────────────────────────────────────┘
         │  transformed_strokes (pixel coords in current frame)
         ▼
/vision/processing_mode/annotated_image   (warped strokes)
/vision/processing_mode/debug_image       (inlier matches, ROI warp, FPS)
/vision/processing_mode/seam_centroid
```

---

## 3. Detailed Stage Explanations

### Stage 1 — ORB Feature Extraction

**ORB** (Oriented FAST and Rotated BRIEF) is a binary feature detector/descriptor that runs efficiently on CPU without GPU acceleration. The node creates a single shared ORB instance with a keypoint budget of **1000** per frame:

```python
self._orb = cv2.ORB_create(1000)
```

During the **teach phase** (`teach_reference`), ORB is run only inside the operator-defined ROI polygon (masked). During the **run phase**, ORB runs on the full frame (no mask) so that it can find the ROI region wherever it has moved.

---

### Stage 2 — KNN Matching with Lowe's Ratio Test

A `BFMatcher` (Brute-Force Matcher) with Hamming distance is used to match the reference descriptors against the current frame. `crossCheck=False` enables `knnMatch` with `k=2` neighbours.

**Lowe's ratio test** filters ambiguous matches where the best match is not significantly better than the second-best:

```python
matches = self._bf.knnMatch(self._ref_desc, curr_desc, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

If fewer than **10 good matches** are found, the node falls back to publishing without alignment and displays a `LOW MATCHES` banner in the debug image.

---

### Stage 3 — Spatial Filtering

Before computing the transform, the top-50 matches by Hamming distance are kept. A **spatial spread check** then validates that the selected reference points are geometrically distributed (not all clustered in one corner):

```python
std = np.std(src_pts.reshape(-1, 2), axis=0)
if std[0] < 10 or std[1] < 10:
    # Reject: POOR SPATIAL SPREAD
```

This prevents degenerate transforms when all good matches happen to lie on a single line or in a tiny patch.

---

### Stage 4 — Affine Estimation with RANSAC

`cv2.estimateAffinePartial2D` computes a **2×3 partial affine matrix** (rotation + isotropic scale + translation — no shear) using RANSAC to reject outlier matches:

```python
matrix, inliers_mask = cv2.estimateAffinePartial2D(
    src_pts, dst_pts, method=cv2.RANSAC
)
```

Three sequential sanity checks guard against degenerate results:

| Check | Threshold | Failure Banner |
|---|---|---|
| Inlier ratio | ≥ 0.40 | `LOW INLIER RATIO: X.XX` |
| Determinant of 2×2 rotation sub-matrix | ≥ 0.10 | `TRANSFORM DEGENERATE` |
| Scale factor (`‖matrix[:,0]‖`) | ∈ [0.5, 2.0] | `TRANSFORM DEGENERATE` |

Any failure returns the annotated image without alignment.

---

### Stage 5 — Temporal Smoothing

To prevent jitter from frame-to-frame RANSAC variance, the **translation component** of the matrix is exponentially smoothed with `α = 0.5`:

```python
matrix[:, 2] = alpha * matrix[:, 2] + (1.0 - alpha) * self._last_matrix[:, 2]
```

> **Important:** Only the translation column (`matrix[:, 2]`) is smoothed. The rotation and scale columns (`matrix[:, :2]`) are used as-is from RANSAC to prevent matrix skew or distortion from accumulating over time.

A lower `α` value produces smoother but more lagged tracking; a higher `α` responds faster but jitters more.

---

### Stage 6 — Stroke Transformation

Each stored stroke is transformed into the current frame's coordinate system:

```python
pts = np.float32(stroke).reshape(-1, 1, 2)
t_pts = cv2.transform(pts, matrix)
transformed_strokes.append(t_pts.reshape(-1, 2).astype(np.int32).tolist())
```

The centroid of all transformed strokes is then computed from a binary mask (same method as `manual_line`) and published as `seam_centroid`.

---

### Config Persistence

The aligner's config is saved to `~/.parol6/manual_aligner_config.json`. The schema extends the base `manual_line` format with a `reference` block:

```json
{
  "color":   [0, 0, 255],
  "width":   5,
  "strokes": [[[x1,y1],[x2,y2],...], ...],
  "reference": {
    "polygon":     [[x1,y1],[x2,y2],[x3,y3],...],
    "keypoints":   [[x1,y1],[x2,y2],...],
    "descriptors": {
      "data":  "<base64-encoded uint8 bytes>",
      "shape": [N, 32]
    },
    "image_size":  [width, height]
  }
}
```

ORB descriptors are serialised as base64-encoded raw bytes to avoid a NumPy JSON dependency.

**Fallback:** if the aligner config is missing but `~/.parol6/manual_line_config.json` exists, the node loads it as a Legacy Fixed-Mode fallback and emits a WARN log.

---

## 4. ROS API

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/captured_image_color` | `sensor_msgs/Image` | Raw colour image from the camera (configurable) |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` | Frame with aligned strokes painted |
| `/vision/processing_mode/debug_image` | `sensor_msgs/Image` | Alignment diagnostics: inlier matches, ROI warp, FPS, status banner |
| `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` | Pixel-space centroid of transformed strokes |

### Services

| Service | Type | Description |
|---|---|---|
| `~/set_strokes` | `std_srvs/Trigger` | Load strokes from `strokes_json` parameter; clears any taught reference; activates Fixed Mode |
| `~/teach_reference` | `std_srvs/Trigger` | Extract ORB features from the current frame inside the ROI polygon; activates Adaptive Mode |
| `~/reset_strokes` | `std_srvs/Trigger` | Clear all strokes, ROI, and reference features; delete saved config |

**Workflow — Adaptive Mode Setup:**
```bash
# 1. Set strokes + ROI polygon in strokes_json (GUI typically does this)
ros2 param set /manual_line_aligner strokes_json \
  '{"strokes": [[[10,50],[200,50]]], "roi_polygon": [[0,0],[320,0],[320,240],[0,240]]}'

# 2. Teach the current frame as the reference
ros2 service call /manual_line_aligner/teach_reference std_srvs/srv/Trigger {}

# 3. Node is now in Adaptive Mode — strokes follow the part
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_topic` | string | `/vision/captured_image_color` | Input image topic |
| `stroke_color` | int[] | `[0, 0, 255]` | BGR paint colour (default = red) |
| `stroke_width` | int | `5` | Stroke thickness in pixels |
| `strokes_json` | string | `""` | JSON payload read by `set_strokes` and `teach_reference` |
| `publish_debug` | bool | `True` | Enable debug image with alignment diagnostics |

---

## 5. Debug Visualisation

| Element | Mode | Description |
|---|---|---|
| Gray polylines | Adaptive | Original (un-warped) strokes — shows where they started |
| Green polylines | Adaptive | Transformed (aligned) strokes in the current frame |
| Blue polygon | Adaptive | Transformed ROI boundary |
| Green circles + lines | Adaptive | Up to 20 inlier feature matches (reference → current) |
| Coloured polylines | Fixed | Strokes at fixed pixel positions |
| Status banner | Both | `ALIGNMENT OK` (green) / `FIXED MODE (NO ROI)` (orange) / error banners (red) |
| FPS counter | Both | Top-right corner: real-time throughput in frames/second |

---

## 6. Differences vs. `manual_line`

| Feature | `manual_line` | `manual_line_aligner` |
|---|---|---|
| Alignment | Fixed pixel positions | Affine-warped to follow part movement |
| Teach step required | No | Yes (for Adaptive Mode) |
| Feature matching | None | ORB + BFMatcher + RANSAC |
| Config path | `manual_line_config.json` | `manual_aligner_config.json` |
| Temporal smoothing | None | Exponential smoothing on translation (α = 0.5) |
| Debug information | Centroid crosshair | Inlier matches, ROI warp, FPS, alignment status |

---

## 7. Build & Run

```bash
# Build (inside Docker or with ROS 2 sourced)
cd /workspace
colcon build --packages-select parol6_vision --symlink-install
source install/setup.bash

# Run with defaults
ros2 run parol6_vision manual_line_aligner

# Override smoothing factor (lower = smoother/laggier)
ros2 run parol6_vision manual_line_aligner \
  --ros-args \
  -p stroke_width:=6
```

---

## 8. Troubleshooting

### `LOW FEATURES IN FRAME` banner

The current frame has fewer than 10 ORB keypoints. Common causes:
- Camera image is blurry or overexposed — check `/vision/captured_image_color` in RViz
- Part surface has very few textures — try placing a textured reference marker near the ROI

### `LOW MATCHES` banner

ORB finds enough features in the frame but fewer than 10 survive Lowe's ratio test. Causes:
- The part has changed position too drastically since teaching (>≈30% scale change or large rotation)
- Lighting conditions changed significantly — re-teach with `teach_reference`

### `POOR SPATIAL SPREAD` banner

All matched keypoints are in the same small region. The affine estimation cannot be reliably solved. Ensure the ROI polygon covers a textured area with features spread across its full extent.

### `TRANSFORM DEGENERATE` banner

RANSAC returned a matrix that implies an implausible scale (< 0.5× or > 2×). Usually means too few inliers or a symmetric surface causing false matches. Re-teach or increase the ROI area to include more unique features.

### `RANSAC FAIL` banner

`estimateAffinePartial2D` returned `None`. This can happen when the number of inliers drops below the RANSAC minimum. Try re-teaching the reference frame.

### Strokes drifting over many frames

The temporal smoothing converges on a bias if RANSAC consistently returns a slightly incorrect transform. To reset: call `~/reset_strokes` and re-teach the reference.

### CV Bridge Error (`bgr8` conversion failed)

The node always requests `bgr8`. Ensure the upstream topic's encoding is `bgr8` or `rgb8`.
