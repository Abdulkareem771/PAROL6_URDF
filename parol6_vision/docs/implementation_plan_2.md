# Capture_images & Read_image ROS2 Nodes

Two new nodes that **decouple** image acquisition from live Kinect2 streaming so
the full vision pipeline can work on saved images.

---

## Proposed Changes

### Stage 1 — Capture_images Node

#### [NEW] [capture_images_node.py](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/capture_images_node.py)

A ROS2 node named `capture_images` that:

- Subscribes to **`/kinect2/qhd/image_color_rect`** (RGB) and **`/kinect2/qhd/image_depth_rect`** (depth) simultaneously
- Saves a matched pair of PNGs (`color_<timestamp>.png` + `depth_<timestamp>.png`) to `~/images_captured/`
- Supports **two capture modes** (selected via the `capture_mode` parameter):
  - **`keyboard`** – a background thread watches for the user pressing **`s`** + Enter; one pair is saved per keypress
  - **`timed`** – automatically saves one pair every `frame_time` seconds (default `60.0`)
- Logs the saved file paths after each capture

Key parameters:
| Parameter | Default | Description |
|---|---|---|
| `save_dir` | `parol6_vision/data/images_captured` | Output folder (created if absent) |
| `capture_mode` | `keyboard` | `keyboard` or `timed` |
| `frame_time` | `60.0` | Seconds between saves (timed mode only) |
| `image_encoding` | `bgr8` | cv_bridge encoding for color image |

---

### Stage 2 — Read_image Node

#### [NEW] [read_image_node.py](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/parol6_vision/read_image_node.py)

A ROS2 node named `read_image` that:

- Watches `~/images_captured/` for **new files only** (using a polling loop that tracks already-seen filenames)
- When a new matched pair (`color_<ts>.png` + `depth_<ts>.png`) appears, publishes:
  - color image → **`/vision/captured_image_color`** (`sensor_msgs/Image`)
  - depth image → **`/vision/captured_image_depth`** (`sensor_msgs/Image`, encoding `16UC1`)
- Sets `frame_id = kinect2_rgb_optical_frame` and a fresh ROS timestamp on each published pair
- Does **not** republish images that were already present when the node started

Key parameters:
| Parameter | Default | Description |
|---|---|---|
| `save_dir` | `parol6_vision/data/images_captured` | Folder to watch |
| `poll_rate` | `1.0` | Hz, how often to check for new files |
| `frame_id` | `kinect2_rgb_optical_frame` | TF frame in published image headers |

---

### Integration — Launch File & Entry Points

#### [NEW] [capture_and_replay.launch.py](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/launch/capture_and_replay.launch.py)

Launches both new nodes plus `red_line_detector` and `depth_matcher` remapped to consume from `read_image`:

```
capture_images  →  saves to parol6_vision/data/images_captured/
                     color_<ts>.png  +  depth_<ts>.png

read_image      →  publishes /vision/captured_image_color
                             /vision/captured_image_depth

red_line_detector  remapped: /kinect2/qhd/image_color_rect  → /vision/captured_image_color
depth_matcher      remapped: /kinect2/qhd/image_depth_rect  → /vision/captured_image_depth
```

#### [MODIFY] [setup.py](file:///home/osama/Desktop/PAROL6_URDF/parol6_vision/setup.py)

Add two new `console_scripts` entries:

```python
'capture_images = parol6_vision.capture_images_node:main',
'read_image     = parol6_vision.read_image_node:main',
```

---

## Verification Plan

### Automated (syntax/import)

```bash
# Inside Docker container:
cd /workspace
python3 -c "import parol6_vision.capture_images_node"
python3 -c "import parol6_vision.read_image_node"
```

### Build check

```bash
# Inside Docker container:
cd /workspace
colcon build --packages-select parol6_vision --symlink-install 2>&1 | tail -20
```

### Manual end-to-end flow

1. **Stage 1** – With Kinect2 driver running, in one terminal:
   ```bash
   ros2 run parol6_vision capture_images
   ```
   Then call the capture service:
   ```bash
   ros2 service call /capture_images/capture std_srvs/srv/Trigger {}
   ```
   Check that a PNG appears in `~/images_captured/`.

2. **Stage 2** – In a second terminal:
   ```bash
   ros2 run parol6_vision read_image
   ```
   Verify the topic is publishing:
   ```bash
   ros2 topic echo /vision/captured_image --field header
   ```

3. **Full pipeline** – Use the new launch file (remaps red_line_detector input to replay topic):
   ```bash
   ros2 launch parol6_vision capture_and_replay.launch.py
   ```
