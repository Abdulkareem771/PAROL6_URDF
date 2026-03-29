# Manual Line Node — Developer Guide

## 1. Overview

The `manual_line` node is a **stroke-replay** processing-mode node in the PAROL6 vision pipeline. Instead of detecting weld seams automatically, it lets the operator draw one or more polyline strokes on a GUI panel. Those strokes are then **painted on every subsequent camera frame**, making the node compatible with the same downstream consumer (`path_optimizer`) as `color_mode` or `yolo_segment`.

A key feature is **persistence**: strokes are serialised to `~/.parol6/manual_line_config.json` on disk so that the exact same seam annotation is replayed automatically every time the node restarts — no re-drawing required for repeat jobs at the same fixture position.

**Node Name:** `manual_line`  
**Package:** `parol6_vision`  
**Source:** `parol6_vision/manual_line_node.py`

---

## 2. Architecture & Pipeline

The node operates as a pure **overlay renderer** — it does not run any image-processing algorithm between input and output.

```
/vision/captured_image_color  (sensor_msgs/Image)
            │
            ▼
  ┌───────────────────────────────┐
  │  1. Decode Image (CvBridge)   │
  └──────────────┬────────────────┘
                 │  BGR ndarray
                 ▼
  ┌───────────────────────────────────┐
  │  2. Stroke Painting               │
  │     cv2.polylines() per stroke    │
  │     + stroke mask for centroid    │
  └──────────┬────────────────────────┘
             │  annotated frame
             ▼
  ┌───────────────────────────────┐
  │  3. Centroid Computation       │
  │     mean (x, y) of mask px    │
  └──────┬─────────────┬──────────┘
         │             │
         ▼             ▼
/vision/processing_mode/annotated_image   (to path_optimizer)
/vision/processing_mode/debug_image       (centroid crosshair overlay)
/vision/processing_mode/seam_centroid     (geometry_msgs/PointStamped)
```

> **No minimum-stroke filter.** A frame is always published even when the stroke list is empty — the annotated image is a pass-through and no centroid is emitted.

---

## 3. Detailed Stage Explanations

### Stage 1 — Image Decode

Incoming `sensor_msgs/Image` messages are converted to BGR NumPy arrays via `cv_bridge`. The node subscribes only to a single configurable topic (default `/vision/captured_image_color`). If the conversion fails, an `ERROR`-level log is emitted and the frame is silently dropped.

---

### Stage 2 — Stroke Painting

**Method:** `_apply_strokes(img)`

Each stored stroke is a polyline represented as an ordered list of `[x, y]` pixel coordinates. The node renders all strokes with `cv2.polylines()` using anti-aliasing (`cv2.LINE_AA`).

```python
pts = np.array(stroke_pts, dtype=np.int32)
cv2.polylines(annotated, [pts], isClosed=False,
              color=bgr_color, thickness=stroke_width,
              lineType=cv2.LINE_AA)
```

Simultaneously, a binary `stroke_mask` is rendered at the same thickness — used in Stage 3 to compute the centroid without interfering with the colour overlay.

---

### Stage 3 — Centroid Computation

After all strokes are painted onto the mask, the mean position of all non-zero pixels is computed:

```python
ys, xs = np.where(stroke_mask > 0)
cx = float(xs.mean())
cy = float(ys.mean())
```

This gives the **geometric centre of mass** of the combined stroke annotation. The centroid is published as a `geometry_msgs/PointStamped` with `z = 0.0` (image-plane coordinates).

When no strokes are loaded, the centroid topic is **not published** for that frame.

---

### Config Persistence

The full stroke state (colour, width, list of polylines) is saved to `~/.parol6/manual_line_config.json` using `_save_config()` every time `set_strokes` is called.

**Schema:**
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

On startup, `_load_config()` silently restores this state. If the file does not exist, the node starts with an empty stroke list and logs an `INFO` message: `No saved config found — starting fresh.`

---

## 4. ROS API

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/captured_image_color` | `sensor_msgs/Image` | Raw colour image from the camera (configurable) |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` | Camera frame with strokes painted in red |
| `/vision/processing_mode/debug_image` | `sensor_msgs/Image` | Same frame + centroid crosshair and status badge |
| `/vision/processing_mode/seam_centroid` | `geometry_msgs/PointStamped` | Pixel-space centroid of all painted strokes |

### Services

| Service | Type | Description |
|---|---|---|
| `~/set_strokes` | `std_srvs/Trigger` | Reload strokes from `strokes_json` parameter; save to disk |
| `~/reset_strokes` | `std_srvs/Trigger` | Clear all strokes from memory and delete saved config |

**Calling `set_strokes` from the command line:**
```bash
# 1. Write the JSON payload into the parameter
ros2 param set /manual_line strokes_json '[[[10,50],[200,50]],[[30,100],[180,100]]]'
# 2. Trigger the service to load it
ros2 service call /manual_line/set_strokes std_srvs/srv/Trigger {}
```

The service accepts the `strokes_json` parameter either as:
- A **plain JSON array** — list of polyline point arrays
- A **JSON object** with optional `color`, `width`, and `strokes` keys

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_topic` | string | `/vision/captured_image_color` | Input image topic |
| `stroke_color` | int[] | `[0, 0, 255]` | BGR paint colour (default = red) |
| `stroke_width` | int | `5` | Stroke thickness in pixels |
| `strokes_json` | string | `""` | JSON-encoded stroke list; updated by `set_strokes` service |
| `publish_debug` | bool | `True` | Enable `/vision/processing_mode/debug_image` |

---

## 5. Debug Visualisation

When `publish_debug: True`, the debug image shows:

| Element | Description |
|---|---|
| Coloured polylines | All stored strokes rendered at configured colour and width |
| White crosshair (`+`) | Centroid position of all painted pixels combined |
| Status badge | Dark banner with stroke count and centroid coordinates |

When no strokes are loaded, the badge reads:
```
Manual Line — no strokes saved
```

---

## 6. Build & Run

```bash
# Build (inside Docker or with ROS 2 sourced)
cd /workspace
colcon build --packages-select parol6_vision --symlink-install
source install/setup.bash

# Run with defaults
ros2 run parol6_vision manual_line

# Run with a thicker blue stroke override
ros2 run parol6_vision manual_line \
  --ros-args \
  -p stroke_color:="[255, 0, 0]" \
  -p stroke_width:=8
```

---

## 7. Relationship to Other Nodes

| Feature | `manual_line` | `color_mode` / `yolo_segment` |
|---|---|---|
| Detection strategy | Operator-drawn strokes | Automatic CV / ML detection |
| Output topics | Same `processing_mode/*` topics | Same `processing_mode/*` topics |
| Centroid source | Mean of painted pixels | Detected object centroid |
| Persistence | Saved to `~/.parol6/manual_line_config.json` | None |
| Part movement handling | Fixed position — no re-alignment | Depends on node |

Use `manual_line` when:
- The part is clamped in a fixed fixture and does not move between runs.
- Automatic red-line detection is unreliable due to lighting or surface finish.
- You need a deterministic, reproducible weld path defined by the operator.

---

## 8. Troubleshooting

### Strokes not appearing on the image

1. Check that `set_strokes` was called after writing `strokes_json`.
2. Verify the JSON is valid: `echo '<json>' | python3 -m json.tool`
3. Confirm the node subscribed to the correct camera topic:
   ```bash
   ros2 topic echo /vision/captured_image_color --no-arr
   ```

### Centroid not published

- This is expected when the stroke list is empty.
- If strokes are loaded but centroid is missing, check that `stroke_width` is large enough to produce visible mask pixels.

### Old strokes reappear after `reset_strokes`

- The saved config file is deleted by `reset_strokes`, but verify by checking:
  ```bash
  ls ~/.parol6/manual_line_config.json
  ```
  If the file still exists, a permissions error may have prevented deletion (the node logs a WARN in this case).

### CV Bridge Error (`bgr8` conversion failed)

- Ensure the upstream camera publishes with `encoding: bgr8` or `rgb8`.
- The node always requests `bgr8`; cv_bridge will auto-convert when possible.
