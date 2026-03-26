# Vision Pipeline GUI — Full Reference Guide

`parol6_vision/scripts/vision_pipeline_gui.py`

A unified PySide6 launcher for the PAROL6 vision pipeline.  
Manage every pipeline node, preview live ROS topics, define the crop ROI, and annotate
manual red-lines — all from one window.

---

## Table of Contents

1. [Launch the GUI](#1-launch-the-gui)
2. [Interface Overview](#2-interface-overview)
3. [Pipeline Stages](#3-pipeline-stages)
   - [Stage 1 — Camera](#stage-1--camera)
   - [Stage 2 — Processing Mode](#stage-2--processing-mode)
   - [Stage 3 — Backend Pipeline](#stage-3--backend-pipeline)
   - [Stage 4 — Send to MoveIt](#stage-4--send-to-moveit)
4. [Right-Side Tabs](#4-right-side-tabs)
   - [Visual Outputs](#visual-outputs-tab)
   - [Manual Red Line](#manual-red-line-tab)
   - [ROS Launch](#ros-launch-tab)
   - [Crop Image](#crop-image-tab)
   - [Console Logs](#console-logs-tab)
5. [Full Workflow Walk-Through](#5-full-workflow-walk-through)
6. [ROS Topics & Services Reference](#6-ros-topics--services-reference)
7. [Config Files](#7-config-files)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Launch the GUI

### Recommended — from the launcher

```bash
cd ~/Desktop/PAROL6_URDF

# If ROS environment is already sourced:
python3 vision_work/launcher.py
# Then click 🔭 Vision Pipeline Launcher (ROS 2)
```

The launcher automatically sources ROS before spawning the GUI, so you will
always see **ROS 2 ✅** in the header.

### Direct launch (ensure ROS is sourced first)

```bash
source /opt/ros/humble/setup.bash
source /opt/kinect_ws/install/setup.bash  # optional, skip if not using Kinect
source ~/Desktop/PAROL6_URDF/install/setup.bash

python3 ~/Desktop/PAROL6_URDF/parol6_vision/scripts/vision_pipeline_gui.py
```

### ROS status badge

The top-right corner of the header shows the ROS 2 import status:

| Badge | Meaning |
|-------|---------|
| `ROS 2 ✅` | All ROS imports succeeded; live topic previews and service calls are active |
| `ROS 2 offline: <error>` | Import failed; GUI still opens but cannot talk to nodes |

---

## 2. Interface Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔭 PAROL6 Vision Pipeline                          ROS 2 ✅         │
├──────────────────────┬──────────────────────────────────────────────┤
│  SIDEBAR             │  TABS                                         │
│  ─────────────────── │  ─────────────────────────────────────────── │
│  Stage 1 — Camera    │  👁 Visual Outputs   (live ROS topic images)  │
│  Stage 2 — Mode      │  ✏️ Manual Red Line  (draw annotation canvas) │
│  Stage 3 — Backend   │  🚀 ROS Launch       (launch files / kill)   │
│  Stage 4 — MoveIt    │  ✂️ Crop Image       (ROI selector)          │
│  Legacy Tools        │  📋 Console Logs     (per-node log tabs)     │
└──────────────────────┴──────────────────────────────────────────────┘
```

Every node button follows the same pattern:

- **● (grey)** — stopped; **▶ Start** button visible
- **● (green)** — running; **■ Stop** button visible
- Output is streamed to `📋 Console Logs → <Node Name>` tab as well as the **All Nodes** tab

---

## 3. Pipeline Stages

### Stage 1 — Camera

| Control | Action |
|---------|--------|
| **Live Kinect Camera ▶ Start** | Runs `ros2 launch ~/Desktop/PAROL6_URDF/kinect2_bridge_gpu.yaml` |
| **Kill Stale Camera** | Sends SIGINT to any lingering `kinect2_bridge_node` process |
| **Capture mode** | `keyboard (press 's')` — manual single-shot; `timed (auto)` — continuous |
| **Capture Images Node ▶ Start** | Runs `ros2 run parol6_vision capture_images`; also auto-starts Crop Image Node |
| **📷 Trigger Capture** | If mode is `keyboard`: sends `s\n` to the node's stdin. Fallback: publishes a `std_msgs/Empty` to `/vision/capture_trigger` |

> **Note:** The camera does **not** auto-start. Click **▶ Start** on **Live Kinect Camera** first,
> then start **Capture Images Node**.

---

### Stage 2 — Processing Mode

Select the algorithm that will process each captured image:

| Radio | Node started | Output topic |
|-------|-------------|--------------|
| 🔮 YOLO Segment | `ros2 run parol6_vision yolo_segment` | `/vision/processing_mode/annotated_image` |
| 🎨 Color Mode | `ros2 run parol6_vision color_mode` | `/vision/processing_mode/annotated_image` |
| ✏️ Manual Red Line | *(no node)* — uses the Manual Red Line tab | `/vision/processing_mode/annotated_image` |

**Crop Image Node** (also in this section) is a relay node that sits between
raw capture and the rest of the pipeline:

```
/vision/captured_image_raw  →  [crop_image_node]  →  /vision/captured_image_color
```

It loads its config from `~/.parol6/crop_config.json` on startup.
If no config exists the node runs in **pass-through** mode (full frame).

Two operational modes:

| Mode | Behaviour | Effect on downstream |
|------|-----------|---------------------|
| **Mask** *(default)* | Pixels outside the polygon are filled with the mask colour | Resolution unchanged — depth coords valid ✅ |
| **Crop** *(legacy)* | Image is cut to the polygon's bounding box | Resolution changes — depth coords shift ⚠️ |

---

### Stage 3 — Backend Pipeline

These three nodes are chained after the processing mode node:

```
/vision/processing_mode/annotated_image
        ↓
  [path_optimizer]  →  publishes /vision/optimized_path + debug image
        ↓
  [depth_matcher]   →  adds depth to path points
        ↓
  [path_generator]  →  publishes /vision/welding_path (nav_msgs/Path)
```

**🚀 Launch Full Pipeline (stages 1-3)** starts Capture, then the mode node, then
Optimizer → Depth → Generator with staggered timers (200 ms apart).

**☠️ Stop All Nodes** stops every NodeButton-managed process.

---

### Stage 4 — Send to MoveIt

| Control | Action |
|---------|--------|
| **MoveIt Controller ▶ Start** | Runs `ros2 run parol6_vision moveit_controller` |
| **📡 Send Path → MoveIt** | Calls service `/moveit_controller/execute_welding_path` |
| **🌐 Launch vision_moveit.launch.py** | Launches the full MoveIt integration launch file |

---

## 4. Right-Side Tabs

### Visual Outputs Tab

Four live topic preview panels (each is a sub-tab):

| Sub-tab | Topic |
|---------|-------|
| 📷 Live Camera | `/kinect2/qhd/image_color_rect` |
| 📸 Captured Frame | `/vision/captured_image_raw` |
| 🔍 Processing Output | `/vision/processing_mode/annotated_image` |
| 📊 Path Optimizer Debug | `/path_optimizer/debug_image` |

Previews update at ~10 Hz. They show a placeholder until the topic starts publishing.

---

### Manual Red Line Tab

Lets you manually draw red lines over a captured image and publish the result as a ROS Image.

| Control | Action |
|---------|--------|
| **📂 Load Image** | Load a PNG/JPG from disk |
| **📥 Use Latest Cropped Frame** | Pull the most recently received `/vision/captured_image_color` frame into the canvas |
| **Brush (px)** | Brush radius for drawing |
| **🗑 Clear** | Erase all red-line strokes |
| **💾 Save Red-Line Image** | Save annotated image to disk |
| **📡 Publish as ROS Image** | Publish to `/vision/processing_mode/annotated_image` |

**Drawing:** left-click to place polygon vertices; right-click (or click near the first point) to close.  
**Escape key** clears the current polygon.

---

### ROS Launch Tab

Run complete launch files instead of individual nodes:

| Method | Launch file |
|--------|------------|
| Live Pipeline | `live_pipeline.launch.py` |
| Method 3: Fake Hardware | `vision_pipeline.launch.py` |
| Method 4: Real Hardware | `vision_moveit.launch.py` |
| Vision + MoveIt | `vision_moveit.launch.py` |

- **💉 Inject Test Path** — publishes a synthetic 3-point straight path to `/vision/welding_path`
  (useful for testing MoveIt without the full camera pipeline).
- **☠️ Kill All** — stops managed workers and then `pkill`s any stray ROS processes.

---

### Crop Image Tab

Interactive workspace mask / ROI definition for the crop_image_node.

**Left panel** — live raw frame from `/vision/captured_image_raw` with the saved polygon overlaid (blue dashed bounding box).  
**Right panel** — live masked/cropped output from `/vision/captured_image_color`.

#### Drawing a polygon

1. Left-click on the left panel to place vertices
2. Right-click **or** click near the first point to close the polygon (≥ 2 pts to close)
3. The polygon is shown in yellow; its bounding box is shown as a dashed outline
4. Press **Escape** to cancel and start over

#### Mask vs Crop mode

| Mode | What the node publishes | Depth coordinates |
|------|------------------------|------------------|
| 🛡 **Mask** *(recommended)* | Full-resolution image; pixels outside polygon filled with the mask colour | Preserved ✅ — depth_matcher works correctly |
| ✂️ **Crop** *(legacy)* | Smaller image cropped to the polygon's bounding box | Shifted ⚠️ — use only if downstream nodes don't need absolute coords |

#### Mask fill colour

The mask fill colour is picked independently of the polygon:

| Control | Action |
|---------|--------|
| **Coloured swatch** (e.g. `■`) | Click to open a colour picker dialog |
| **🔬 Eyedropper** | Click the button, then click any pixel on the left panel image — the sampled colour becomes the fill |

Choose a colour that blends into the background so the AI/colour-mode node is not confused by the masked region (e.g. match the table or floor colour).

#### Control bar buttons

| Button | Action |
|--------|--------|
| **🗑 Clear** | Discard the drawn polygon (does not touch the saved config) |
| **✅ Apply & Save** | Write `~/.parol6/crop_config.json`, call `~/reload_roi` service (mask mode) or push via `SetParameters` (crop mode) |
| **↩ Reset (Pass-through)** | Call `~/clear_roi` service → node disables masking and republishes the full frame |

> **Persistence:** The config is written to disk before the service call. If the node
> restarts it automatically reloads the last saved polygon and fill colour.

> **Retry logic:** If the node was just started when you click Apply & Save, the GUI
> retries every 500 ms for up to 4 s until the service is ready.

---

### Console Logs Tab

One sub-tab per node plus an **All Nodes** tab that receives everything.

**🗑 Clear All Logs** clears every tab simultaneously.

---

## 5. Full Workflow Walk-Through

```
1.  Source ROS and launch the GUI (see §1).
    Confirm header shows  ROS 2 ✅.

2.  Start camera.
    Sidebar → Stage 1 → Live Kinect Camera  ▶ Start
    (skip if using replayed bag data)

3.  Start capture node.
    Sidebar → Stage 1 → Capture Images Node  ▶ Start
    ↳  Crop Image Node starts automatically as well.

4.  [Optional] Define crop ROI.
    Tab → ✂️ Crop Image
    Draw polygon on left panel → ✅ Apply & Save
    Watch right panel update with the cropped frame.

5.  Capture a frame.
    Click  📷 Trigger Capture  (or press 's' if keyboard mode).
    Visual Outputs → 📸 Captured Frame  should update.

6.  Choose processing mode.
    Sidebar → Stage 2 → select radio button.
    Click  ▶ Start Selected Mode Node.

7.  Start backend pipeline.
    Sidebar → Stage 3 → start Optimizer, Depth Matcher, Path Generator.
    (or click 🚀 Launch Full Pipeline to do all of this at once)

8.  Send to MoveIt.
    Sidebar → Stage 4 → ▶ Start MoveIt Controller  →  📡 Send Path → MoveIt

9.  Stop everything.
    Sidebar → Stage 3 → ☠️ Stop All Nodes
```

---

## 6. ROS Topics & Services Reference

### Topics

| Topic | Type | Direction | Publisher |
|-------|------|-----------|-----------|
| `/vision/captured_image_raw` | `sensor_msgs/Image` | → | `capture_images` |
| `/vision/captured_image_color` | `sensor_msgs/Image` | → | `crop_image_node` |
| `/vision/capture_trigger` | `std_msgs/Empty` | → | GUI (fallback trigger) |
| `/vision/processing_mode/annotated_image` | `sensor_msgs/Image` | → | mode node / GUI (manual) |
| `/path_optimizer/debug_image` | `sensor_msgs/Image` | → | `path_optimizer` |
| `/vision/welding_path` | `nav_msgs/Path` | → | `path_generator` / injection |
| `/kinect2/qhd/image_color_rect` | `sensor_msgs/Image` | → | `kinect2_bridge` |

### Services

| Service | Type | Called by |
|---------|------|-----------|
| `/crop_image/reload_roi` | `std_srvs/srv/Trigger` | GUI — Apply & Save (mask mode + belt-and-suspenders for crop mode) |
| `/crop_image/clear_roi` | `std_srvs/srv/Trigger` | GUI — Reset (Pass-through) |
| `/crop_image/set_parameters` | `rcl_interfaces/srv/SetParameters` | GUI — Apply & Save (crop/legacy mode only) |
| `/moveit_controller/execute_welding_path` | `std_srvs/srv/Trigger` | GUI — Send Path |

### ROS Parameters — `/crop_image` node

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `/vision/captured_image_raw` | Raw image input |
| `output_topic` | string | `/vision/captured_image_color` | Cropped output |
| `config_path` | string | `~/.parol6/crop_config.json` | Config file path |
| `roi` | int[4] | *(unset)* | `[x, y, width, height]` in pixels |

---

## 7. Config Files

### Crop config — `~/.parol6/crop_config.json`

Written by the GUI on **Apply & Save**, read by `crop_image_node` on startup and on `~/reload_roi`.

#### Mask mode (recommended)

```json
{
  "enabled": true,
  "mode": "mask",
  "polygon": [[120, 80], [760, 80], [760, 480], [120, 480]],
  "mask_color": [180, 160, 140]
}
```

- `polygon` — list of `[x, y]` pairs in **original image pixel coords** (not scaled)
- `mask_color` — `[R, G, B]` fill for the masked region; `[0,0,0]` = black (default)

#### Crop mode (legacy bbox)

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

Set `"enabled": false` to force pass-through without clearing the saved coordinates.

### ROS log directory

All subprocesses launched by the GUI redirect their `ROS_LOG_DIR` to `/tmp/parol6_ros_logs/`
to avoid cluttering `~/.ros/log/`.

```bash
ls /tmp/parol6_ros_logs/   # individual node log files
```

---

## 8. Troubleshooting

### Header shows `ROS 2 offline`

**Symptom:** Badge is red; topic previews show placeholder; service calls fail.

**Causes & fixes:**

1. Launched the GUI without sourcing ROS:
   ```bash
   # Always launch via the WeldVision launcher, or source manually first:
   source /opt/ros/humble/setup.bash
   source ~/Desktop/PAROL6_URDF/install/setup.bash
   python3 parol6_vision/scripts/vision_pipeline_gui.py
   ```
2. Missing workspace build:
   ```bash
   cd ~/Desktop/PAROL6_URDF
   colcon build --packages-select parol6_vision
   source install/setup.bash
   ```

---

### Crop Image Node button shows ▶ Start but node is already running

After the node is stopped from outside the GUI (e.g. terminal `Ctrl-C`) the button
may not reflect the true state, since auto-polling was removed to keep the GUI
thread responsive.

**Fix:** Click **■ Stop** once (safe no-op if already dead), then **▶ Start** again.

To verify node state independently:
```bash
ros2 node list | grep crop_image
ros2 node info /crop_image
```

---

### Apply & Save shows "service not available" or keeps retrying

**Symptom:** Log shows `[Crop] Waiting for /crop_image service (N attempts left)…`
and eventually gives up.

**Diagnosis:**
```bash
ros2 node list | grep crop_image          # Is the node present?
ros2 service list | grep crop_image       # Is the service advertised?
ros2 node info /crop_image                # Shows publishers and services
```

**Fixes:**

1. Node not running — click **▶ Start** on **Crop Image Node** and wait ~3 s before applying.
2. Build is stale — rebuild and re-source:
   ```bash
   colcon build --packages-select parol6_vision && source install/setup.bash
   ```
3. Parameter type mismatch — the node expects `roi` as `INTEGER_ARRAY`.
   Test manually:
   ```bash
   ros2 param set /crop_image roi "[100,80,640,400]"
   ```

---

### Cropped output panel (right side of Crop tab) stays blank

**Symptom:** After Apply & Save the left panel shows the full frame but the right panel never updates.

**Diagnosis steps:**

```bash
# 1. Confirm /crop_image is publishing
ros2 topic info /vision/captured_image_color

# 2. Check frequency
ros2 topic hz /vision/captured_image_color

# 3. Grab one header
ros2 topic echo --once /vision/captured_image_color/header
```

**Common causes:**

| Cause | Fix |
|-------|-----|
| Node in pass-through but no raw frame has arrived yet | Trigger a capture first (press 📷), then apply ROI |
| ROI param was rejected (node logs a warning) | Check Console Logs → Crop Image Node tab for `⚠ Node rejected ROI param` |
| No publisher on `/vision/captured_image_color` | Stop and restart the Crop Image Node |

---

### Captured frame preview (Visual Outputs tab) stays blank

**Symptom:** `/vision/captured_image_raw` panel shows "Waiting…" even after a capture trigger.

**Diagnosis:**
```bash
ros2 topic info /vision/captured_image_raw
ros2 topic hz /vision/captured_image_raw
```

**Fixes:**
1. **Capture Images Node** is not running — start it from the sidebar.
2. Camera is not publishing — check **Live Kinect Camera** node is started.
3. `cv2` not installed — run `pip install opencv-python`.

---

### "📥 Use Latest Cropped Frame" button gives a dialog saying no image received

**Symptom:** Pop-up: "No image has been received yet on /vision/captured_image_color."

The GUI caches every frame received on `/vision/captured_image_color`.
If the topic has not published since the GUI started, the cache is empty.

**Fix:**
1. Ensure Capture Images Node and Crop Image Node are both running.
2. Trigger a capture — this causes `crop_image_node` to relay a frame on the output topic.
3. Click **📥 Use Latest Cropped Frame** again.

---

### Manual Red Line canvas stays empty after clicking "Use Latest Cropped Frame"

The button only loads if a frame has been received.  
If the crop node is in pass-through mode, the full raw frame is relayed — that is still valid.

Check that ROS 2 is online and `cv2` is installed:
```bash
python3 -c "import cv2; print(cv2.__version__)"
```

---

### GUI freezes or becomes unresponsive

The GUI spins ROS in a 10 ms `QTimer` (`rclpy.spin_once`). Heavy ROS traffic or
slow callbacks can stall this loop.

**Quick checks:**
```bash
# Check ROS graph load
ros2 topic hz /kinect2/qhd/image_color_rect   # should be ~30 Hz; higher → reduce quality
```

**Fixes:**
- Reduce Kinect stream quality in `kinect2_bridge_gpu.yaml` if CPU is saturated.
- Kill the camera and reconnect: **Kill Stale Camera** button.

---

### `pkill` / **Kill Stale Camera** doesn't stop the node

```bash
# Manually identify and kill the process
ps aux | grep kinect2_bridge
kill -INT <pid>

# Or by name
pkill -INT -f kinect2_bridge_node
```

### Live camera or captured frame previews do not update (silently fail)

**Symptom:** You click **📷 Trigger Capture** (or the camera is running), but the GUI's Visual Outputs tabs stay blank or do not refresh. The GUI does not crash, but the terminal might show no errors, or only `DeprecationWarning`s.

**Explanation:** If the GUI's PySide image callback attempts to construct a `QImage` using a Numpy `int64` type for the memory stride (instead of a native Python `int`), PySide strictly rejects it with a `TypeError`. Because this happens inside a background Qt slot, the GUI silently "swallows" the exception (stopping the image from drawing) without crashing the application. 
**Fix:** The latest version wraps all Numpy strides in explicit `int()` casts and adds rigorous traceback logging. Re-launch the GUI with the latest code.

---

### GUI crashes when clicking "Use Latest Cropped Frame" (Manual tab)

**Symptom:** You click the button to load the cached cropped frame, and the entire GUI window instantly closes (segmentation fault).

**Explanation:** This was a memory alignment issue. PySide's `QImage` generator requires exact memory stride measurements. If an image was cropped to a width that isn't a multiple of 4, OpenCV adds padding bytes to the rows. Older versions of the GUI assumed a strict `width * 3` stride, causing an out-of-bounds memory read when the padding existed.
**Fix:** This is fixed in the latest version by dynamically passing `image.strides[0]` to the `QImage` constructor. Pull the latest code and restart the GUI.

---

### Mask not applied automatically / mode nodes see the whole image / frame flickers

**Symptom:** The path optimizer draws bounding boxes in the black (masked) region, or the mask spontaneously disappears on the second capture, falling back to the full frame, or flickers wildly between different mask shapes and colors.

**Explanation:** There are two main reasons for this:
1. **Ghost Nodes (Multiple Instances)**: If you restarted the GUI or ran `live_pipeline.launch.py` multiple times without killing old processes, there might be 5–10 old `crop_image_node` instances still running in the background. Because they all subscribe to and publish on the same topics, they cause extreme race conditions! Some nodes have the old pass-through config, and some have the new mask config, sending a garbled mix of images to YOLO.
2. **Processing Exceptions**: By design, if `crop_image_node` throws a Python exception while processing a frame (e.g., from an invalid polygon or a Numpy array error), it catches the error and **passes the original unmasked image** to the downstream nodes rather than freezing the pipeline.

**Diagnosis & Fix:**
1. Click the big red **☠️ Kill All Background Nodes** button at the top-right of the GUI header. This runs `pkill` securely on all pipeline background processes to eliminate ghosts.
2. Check the **Crop Image Node** logs in the GUI Console for `Processing error: ...`
3. If the error persists, click **Clear** on the Crop tab, redraw a simple 4-point polygon, and click **Apply & Save**.

---

### Mask fill colour not applied (background still black)

**Symptom:** Custom fill colour was set and Applied, but the right panel still shows black background.

**Checks:**
```bash
cat ~/.parol6/crop_config.json | python3 -m json.tool   # Look for 'mask_color' field
```

If `mask_color` is missing or `[0,0,0]`, the config was saved before the colour was picked or
the node is running an old cached binary.

**Fix:**
1. Pick the colour in the GUI **before** clicking Apply & Save (the swatch must show the chosen colour).
2. Rebuild to pick up the node update:
   ```bash
   colcon build --packages-select parol6_vision
   source install/setup.bash
   ```
3. Restart the Crop Image Node and click Apply & Save again.

---

### Eyedropper picks the wrong colour

**Symptom:** After clicking the 🔬 Eyedropper and clicking the image, the colour swatch shows an unexpected colour.

- The eyedropper samples the **original image pixel** (not the scaled/displayed pixel), so it is accurate even when the display is resized.
- Make sure to click on the **left panel** (full raw frame), not the right panel.
- If the panel shows a stale frame, trigger a new capture first.

---

### ROS Launch tab — launch fails immediately

Check the Console Logs → All Nodes tab for the error.  
Common issue: launch file does not exist (package not built):
```bash
ros2 pkg prefix parol6_vision     # Should print a path
colcon build --packages-select parol6_vision
source install/setup.bash
```

---

### Camera running at ~0.3 Hz (Kinect GPU regression)

**Symptom:** The `/kinect2/qhd/image_color_rect` topic publishes at 0.3–1 Hz instead of ~15 Hz; everything downstream is extremely slow.

**Root cause:** `kinect2_bridge_gpu.yaml` can accidentally have `depth_method: "cpu"` which, combined with `bilateral_filter: true` and `edge_aware_filter: true`, reduces throughput to ~0.3 Hz because both filters serialize on CPU.

**Diagnosis:**
```bash
ros2 topic hz /kinect2/qhd/image_color_rect   # should be ~15 Hz
clinfo -l                                      # check OpenCL platform is listed
cat ~/Desktop/PAROL6_URDF/kinect2_bridge_gpu.yaml | grep -E 'depth_method|fps_limit|reg_method'
```

**Fix (GPU/OpenCL available — NVIDIA driver present):**
```yaml
# kinect2_bridge_gpu.yaml
depth_method: "opencl"       # was "cpu"
reg_method:   "opencl_cpu"   # was "cpu"
fps_limit:    15.0           # was 1.0
```

**Fix (CPU-only fallback):** If no OpenCL GPU is available, disable the expensive filters:
```yaml
depth_method:      "cpu"
reg_method:        "cpu"
fps_limit:         15.0
bilateral_filter:  false   # ← REQUIRED for CPU mode
edge_aware_filter: false   # ← REQUIRED for CPU mode
```

Restart the Kinect camera node from the GUI after editing the YAML.

---

### ☠️ Kill All — nodes still visible in `ros2 node list` after kill

**Old behaviour (before 2026-03-26):** The Kill All button used `pkill -9` (SIGKILL), which bypassed ROS shutdown code and left ghost entries in the graph for 30–60 s.

**Current behaviour:** Uses `pkill -INT` first (allows `rclpy.shutdown()`) then `pkill -TERM` after 1 s. Ghosts clear within a few seconds.

If you still see ghost nodes:
```bash
pkill -KILL -f 'ros2 run parol6'   # nuclear option — only if TERM didn't work
```

---

## Related Documentation

| Guide | Location |
|-------|----------|
| **Developer guide (architecture, nodes, handoff prompt)** | [`VISION_PIPELINE_DEVELOPER_GUIDE.md`](VISION_PIPELINE_DEVELOPER_GUIDE.md) |
| Capture & replay with bags | [`CAPTURE_REPLAY_GUIDE.md`](CAPTURE_REPLAY_GUIDE.md) |
| Camera calibration | [`CAMERA_CALIBRATION_GUIDE.md`](CAMERA_CALIBRATION_GUIDE.md) |
| Depth matcher deep-dive | [`DEPTH_MATCHER_GUIDE.md`](DEPTH_MATCHER_GUIDE.md) |
| Path generator | [`PATH_GENERATOR_GUIDE.md`](PATH_GENERATOR_GUIDE.md) |
| RViz visualization setup | [`RVIZ_SETUP_GUIDE.md`](RVIZ_SETUP_GUIDE.md) |
| Testing guide | [`TESTING_GUIDE.md`](TESTING_GUIDE.md) |

---

## Change Log

### Session 1 — 2026-03-24

| # | Change |
|---|--------|
| 1 | Removed blocking auto-poll for crop node button status |
| 2 | `_crop_apply_save` → async retry (8 × 500 ms) for reload_roi service |
| 3 | Manual tab: no auto-load from topic; explicit button only |
| 4 | `SetParameters` result checks `.successful` before logging |
| 5 | Log tab registration moved to `_build_tabs()` |
| 6 | `QImage` stride: `int(strides[0])` — prevents PySide6 TypeErrors |
| 7 | `crop_image_node`: numpy mask fill fixed (was crashing silently) |

### Session 2 — 2026-03-26

| # | Change |
|---|--------|
| 8 | `_crop_reset()` async retry — was blocking `wait_for_service` on Qt thread |
| 9 | `_manual_publish_ros()` publisher cached — was leaking a new publisher on every click |
| 10 | `_inject_test_path()` YAML corrected — was malformed for `ros2 topic pub` |
| 11 | `_kill_all()` SIGKILL → SIGINT+SIGTERM — allows clean ROS deregistration |
| 12 | `capture_images_node.py` docstring corrected (topic name) |
| 13 | **Kinect GPU regression fixed:** `kinect2_bridge_gpu.yaml` — `depth_method: opencl`, `fps_limit: 15.0` |
