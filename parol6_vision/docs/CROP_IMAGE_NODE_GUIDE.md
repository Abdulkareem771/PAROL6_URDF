# `crop_image_node` — PAROL6 Vision Pipeline (Stage 1b)

## Overview

`crop_image_node` is an always-active relay node that sits between the image capture stage and the rest of the PAROL6 vision pipeline. It receives raw camera frames, optionally applies a Region of Interest (ROI) operation to them, and republishes the result for downstream nodes to consume.

```
/vision/captured_image_raw  ──►  [crop_image_node]  ──►  /vision/captured_image_color
```

The node loads its configuration automatically on startup from `~/.parol6/crop_config.json`. If the file is missing or cropping is disabled, the node operates in **pass-through mode**, forwarding every incoming frame unchanged.

---

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/captured_image_raw` | `sensor_msgs/Image` | Raw camera frames from the capture node |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/vision/captured_image_color` | `sensor_msgs/Image` | Cropped/masked output (or raw pass-through) |

### Services

| Service | Type | Description |
|---|---|---|
| `~/reload_roi` | `std_srvs/Trigger` | Re-read config file from disk and apply changes immediately without restarting the node |
| `~/clear_roi` | `std_srvs/Trigger` | Disable processing, save `enabled: false` to config, and switch to pass-through mode |

### ROS Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_topic` | `string` | `/vision/captured_image_raw` | Topic to subscribe to |
| `output_topic` | `string` | `/vision/captured_image_color` | Topic to publish to |
| `config_path` | `string` | `~/.parol6/crop_config.json` | Path to the JSON configuration file |
| `roi` | `int[4]` | *(optional)* | `[x, y, width, height]` — sets a rectangular crop directly via parameter |

> **Note:** Setting the `roi` parameter at runtime (e.g., via `ros2 param set`) immediately activates **crop mode** and saves the new config to disk. For **mask mode**, write the config file manually and call the `~/reload_roi` service instead.

---

## Operating Modes

The node supports two distinct processing modes, controlled by the `mode` field in the config file.

### 1. Mask Mode *(Recommended — Default)*

Zeroes out all pixels **outside** a user-defined polygon. The output image has the **same resolution** as the input, meaning pixel coordinates are fully preserved. This makes it safe for use with depth maps and any downstream nodes that rely on absolute pixel positions.

The masked-out region is filled with a configurable color (default: black `[0, 0, 0]`).

**When to use:** Any time downstream nodes need pixel positions to remain valid (e.g., depth matching, seam detection).

### 2. Crop Mode *(Legacy)*

Performs a rectangular crop to a defined bounding box `(x, y, width, height)`. The output is a **smaller image** with different dimensions. Pixel coordinates are no longer aligned with the original frame.

**When to use:** Only if downstream nodes are resolution-agnostic and do not depend on absolute pixel positions.

---

## Configuration File

The node reads from/writes to `~/.parol6/crop_config.json`.

### Mask Mode (Recommended)

```json
{
  "enabled": true,
  "mode": "mask",
  "polygon": [[x1, y1], [x2, y2], [x3, y3], ...],
  "mask_color": [0, 0, 0]
}
```

| Field | Type | Description |
|---|---|---|
| `enabled` | `bool` | Whether processing is active. `false` → pass-through |
| `mode` | `string` | `"mask"` |
| `polygon` | `list of [x, y]` | At least 3 points defining the keep region in image pixel coordinates |
| `mask_color` | `[R, G, B]` | Fill color for the masked-out region (default: black) |

### Crop Mode (Legacy)

```json
{
  "enabled": true,
  "mode": "crop",
  "x": 120,
  "y": 80,
  "width": 640,
  "height": 400
}
```

| Field | Type | Description |
|---|---|---|
| `enabled` | `bool` | Whether processing is active |
| `mode` | `string` | `"crop"` |
| `x`, `y` | `int` | Top-left corner of the bounding box (pixels) |
| `width`, `height` | `int` | Dimensions of the crop region (pixels) |

---

## Startup Behaviour

1. The node declares all ROS parameters and reads them.
2. `_load_config()` is called to parse and apply the JSON config file:
   - **File not found** → pass-through mode (no error).
   - **`enabled: false`** → pass-through mode.
   - **Mask mode with a valid polygon (≥ 3 points)** → polygon mask is active.
   - **Mask mode with no polygon but a bounding box present** → graceful fallback: the bounding box corners are converted to a 4-point polygon and used as a rectangular mask.
   - **Mask mode with no polygon and no bounding box** → processing disabled, warning logged.
   - **Crop mode** → reads `x`, `y`, `width`, `height` fields.
3. Publisher, subscriber, and services are created.
4. A startup log message reports the active configuration.

---

## Frame Processing Flow

For every incoming frame on `/vision/captured_image_raw`:

```
Receive frame
    │
    ▼
enabled? ──No──► Publish original (pass-through)
    │
   Yes
    │
    ├── mode = "mask" ──► _apply_polygon_mask() ──► Publish masked frame
    │
    └── mode = "crop" ──► _apply_crop()         ──► Publish cropped frame
                                                         │
                                               (on any error) ──► Publish original (fallback)
```

---

## Processing Internals

### `_apply_polygon_mask(cv_img, polygon)`

1. Clamps all polygon points to valid image bounds.
2. Creates a single-channel binary mask (`cv2.fillPoly`), white inside the polygon.
3. Constructs a fill image initialized to `mask_color` (converted from RGB to BGR for OpenCV).
4. Uses `np.where` to combine: keep original pixels where mask = 255, use fill color elsewhere.
5. Handles both color (3-channel) and grayscale images. For grayscale, the fill color is converted to luminance: `L = 0.299R + 0.587G + 0.114B`.
6. Output dtype is preserved to match the input.

### `_apply_crop(cv_img, roi)`

1. Clamps the bounding box to valid image bounds.
2. Returns the sliced NumPy array `cv_img[y:y2, x:x2]`.

---

## Runtime Updates

### Reloading via Service

```bash
ros2 service call /crop_image/reload_roi std_srvs/srv/Trigger {}
```

Re-reads `~/.parol6/crop_config.json` and applies the new settings immediately. The very next call to `_publish_current()` uses the updated configuration.

### Clearing via Service

```bash
ros2 service call /crop_image/clear_roi std_srvs/srv/Trigger {}
```

Sets `enabled = false`, clears polygon and ROI state, saves config, and switches to pass-through mode.

### Updating ROI via Parameter

```bash
ros2 param set /crop_image roi "[x, y, width, height]"
```

Directly pushes a rectangular crop region at runtime. This activates **crop mode**, saves the config, and immediately republishes the last frame using the new ROI. For mask-mode polygon updates, edit the config file and call `~/reload_roi` instead.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Config file missing | Pass-through mode (no error raised) |
| Config file malformed / unreadable | Error logged; pass-through mode |
| Mode is `"mask"` but polygon has < 3 points | Warning logged; processing disabled |
| Mode is `"mask"`, no polygon, but bounding box present | Warning logged; bbox converted to 4-corner rectangle mask |
| Exception during frame processing | Error logged; original frame published as fallback |
| Invalid `roi` parameter update (negative/zero dimensions) | Silently ignored |

---

## Dependencies

| Library | Purpose |
|---|---|
| `rclpy` | ROS 2 Python client library |
| `sensor_msgs` | `Image` message type |
| `std_srvs` | `Trigger` service type |
| `cv_bridge` | Converts between ROS `Image` messages and OpenCV arrays |
| `opencv-python` (`cv2`) | Polygon fill and image processing |
| `numpy` | Array operations for mask application |
| `json` / `pathlib` | Config file I/O |

---

## File Location

```
parol6_vision/
└── parol6_vision/
    └── crop_image_node.py
```

Config file (auto-created on first save):
```
~/.parol6/crop_config.json
```
