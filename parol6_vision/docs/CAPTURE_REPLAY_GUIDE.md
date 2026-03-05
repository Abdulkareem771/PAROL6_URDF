# Image Capture & Replay Pipeline

> **Package:** `parol6_vision`  
> **Added nodes:** `capture_images` · `read_image`  
> **Purpose:** Decouple the vision pipeline from a live Kinect2 camera by saving colour + depth images to disk, then replaying them as ROS 2 topics.

---

## Overview

The pipeline is split into two independent stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1  –  capture_images node                                    │
│                                                                     │
│  /kinect2/qhd/image_color_rect ──┐                                  │
│                                  ├──► color_<ts>.png               │
│  /kinect2/qhd/image_depth_rect ──┘    depth_<ts>.png               │
│                                       (parol6_vision/data/          │
│                                        images_captured/)            │
└─────────────────────────────────────────────────────────────────────┘
                        │  new files appear
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2  –  read_image node                                        │
│                                                                     │
│  color_<ts>.png ──► /vision/captured_image_color  (bgr8)           │
│  depth_<ts>.png ──► /vision/captured_image_depth  (16UC1)          │
└─────────────────────────────────────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
  red_line_detector            depth_matcher
  (subscribes to               (subscribes to
  /vision/captured_            /vision/captured_
   image_color)                 image_depth)
```

---

## Saved Files

Each capture saves a **matched pair** with the same timestamp token:

```
parol6_vision/data/images_captured/
    color_20260305_101530_123456.png   ← 8-bit BGR colour
    depth_20260305_101530_123456.png   ← 16-bit unsigned depth (millimetres)
```

> [!IMPORTANT]
> The depth PNG is 16-bit. Always load it with `cv2.imread(path, cv2.IMREAD_UNCHANGED)` to preserve depth values.

---

## Node Reference

### `capture_images`

**Source:** `parol6_vision/capture_images_node.py`

| | |
|---|---|
| **ROS name** | `capture_images` |
| **Subscribed topics** | `/kinect2/qhd/image_color_rect` · `/kinect2/qhd/image_depth_rect` |
| **Published topics** | *(none)* |

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_dir` | string | `parol6_vision/data/images_captured` | Output folder (auto-created) |
| `capture_mode` | string | `keyboard` | `keyboard` or `timed` |
| `frame_time` | float | `60.0` | Seconds between auto-saves *(timed mode only)* |
| `image_encoding` | string | `bgr8` | cv_bridge encoding for colour image |

#### Capture Modes

**`keyboard` (default)**  
A background thread reads stdin. Type **`s`** and press **Enter** in the terminal where the node is running to save one pair immediately.

**`timed`**  
Automatically saves one pair every `frame_time` seconds. No keyboard interaction needed — useful for unattended recording sessions.

#### How synchronisation works
The node uses `message_filters.ApproximateTimeSynchronizer` with a `slop` of 100 ms to match colour and depth frames by timestamp. Only properly matched pairs are saved.

---

### `read_image`

**Source:** `parol6_vision/read_image_node.py`

| | |
|---|---|
| **ROS name** | `read_image` |
| **Subscribed topics** | *(none — reads from disk)* |
| **Published topics** | `/vision/captured_image_color` (bgr8) · `/vision/captured_image_depth` (16UC1) |

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_dir` | string | `parol6_vision/data/images_captured` | Folder to watch |
| `poll_rate` | float | `1.0` | Hz — how often to check for new files |
| `frame_id` | string | `kinect2_rgb_optical_frame` | TF frame placed in image headers |

#### How the watcher works
1. On **startup** the node scans the folder and records every file already present → these are **skipped**.
2. The timer fires every `1 / poll_rate` seconds and lists the folder.
3. When a **new** timestamp token appears with **both** `color_<ts>.png` and `depth_<ts>.png`, the pair is loaded and published **once**.
4. The token is added to the seen-set so it is never published again.

This means the node reacts to captures in near-real-time without re-publishing old data after a restart.

---

## Topic Wiring Summary

| Publisher | Topic | Subscriber |
|---|---|---|
| Kinect2 driver | `/kinect2/qhd/image_color_rect` | `capture_images` |
| Kinect2 driver | `/kinect2/qhd/image_depth_rect` | `capture_images` |
| `read_image` | `/vision/captured_image_color` | `red_line_detector` |
| `read_image` | `/vision/captured_image_depth` | `depth_matcher` |
| `red_line_detector` | `/vision/weld_lines_2d` | `depth_matcher` |
| `depth_matcher` | `/vision/weld_lines_3d` | `path_generator` |

> [!NOTE]
> `red_line_detector` and `depth_matcher` subscribe directly to the `/vision/captured_image_*` topics in their source code. **No `remappings` are needed** in the launch files.

---

## Usage

### Option A — Full launch (recommended)

Starts both stages **plus** `red_line_detector` and `depth_matcher`:

```bash
# Keyboard mode (default)
ros2 launch parol6_vision capture_and_replay.launch.py

# Timed mode — save one pair every 30 s
ros2 launch parol6_vision capture_and_replay.launch.py \
    capture_mode:=timed frame_time:=30.0

# Custom folder
ros2 launch parol6_vision capture_and_replay.launch.py \
    save_dir:=/path/to/my/images
```

### Option B — Run nodes individually

**Terminal 1 — Stage 1 (capture)**
```bash
# Keyboard mode (default)
ros2 run parol6_vision capture_images \
    --ros-args -p capture_mode:=keyboard

# When ready, type 's' + Enter to save a pair

# Timed mode — save one pair every 90 s
ros2 run parol6_vision capture_images \
    --ros-args -p capture_mode:=timed -p frame_time:=90.0

```

**Terminal 2 — Stage 2 (replay)**
```bash
ros2 run parol6_vision read_image
```

**Terminal 3 — Verify topics are publishing**
```bash
ros2 topic hz /vision/captured_image_color
ros2 topic hz /vision/captured_image_depth
```

---

## End-to-End Workflow

```
1.  Start the Kinect2 driver (or kinect2_bridge)
2.  Launch the pipeline:
        ros2 launch parol6_vision capture_and_replay.launch.py
3.  Position the workpiece in the camera's field of view
4.  Press 's' + Enter  →  colour + depth PNGs are saved
5.  read_image detects the new files and publishes them
6.  red_line_detector processes the colour image
7.  depth_matcher back-projects the detected weld line to 3D
8.  path_generator produces a robot welding path
```

---

## File Locations

```
parol6_vision/
├── parol6_vision/
│   ├── capture_images_node.py      ← Stage 1 source
│   └── read_image_node.py          ← Stage 2 source
├── launch/
│   └── capture_and_replay.launch.py
├── data/
│   └── images_captured/            ← PNG pairs saved here
└── docs/
    └── CAPTURE_REPLAY_GUIDE.md     ← this file
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `capture_images` starts but never saves | Colour or depth topic not publishing | Check `ros2 topic hz /kinect2/qhd/image_color_rect` |
| `read_image` starts but no topics appear | No matched pairs in folder yet | Capture at least one pair with Stage 1 |
| `red_line_detector` logs nothing after pair is published | RGB image has no red content | Check image quality / lighting |
| `depth_matcher` logs "0 3D lines" | Depth image all zeros or out of range | Check driver encoding; default range is 300–2000 mm |
| Build fails after adding files | Stale build cache | `rm -rf build/parol6_vision install/parol6_vision && colcon build --packages-select parol6_vision --symlink-install` |
