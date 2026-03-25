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
| **Live Kinect Camera ▶ Start** | Runs `ros2 launch kinect2_bridge kinect2_bridge_launch.yaml` |
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

It loads its ROI from `~/.parol6/crop_config.json` on startup.  
If no config exists the node runs in **pass-through** mode (full frame).

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

Interactive ROI definition for the crop_image_node.

**Left panel** — shows the live raw frame from `/vision/captured_image_raw` with the saved ROI overlaid (blue dashed box).

**Right panel** — shows the live cropped output from `/vision/captured_image_color`.

**Drawing an ROI:**
1. Left-click to place polygon vertices on the left panel image
2. Right-click (or click near the first point) to close the polygon
3. The bounding box of the polygon becomes the ROI (shown as a yellow dashed box)
4. Click **✅ Apply & Save** to persist and push to `/crop_image`

| Button | Action |
|--------|--------|
| **🗑 Clear ROI** | Discard the drawn shape (does not touch the saved config) |
| **✅ Apply & Save** | Write `~/.parol6/crop_config.json`, push ROI to node via `SetParameters`, then trigger `~/reload_roi` |
| **↩ Reset (Pass-through)** | Call `~/clear_roi` service → node disables crop and republishes the full frame |

> If the Crop Image Node was just started when you click Apply & Save, the GUI
> will retry every 500 ms for up to 4 seconds until the node's parameter service is
> ready before sending the ROI.

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
| `/crop_image/set_parameters` | `rcl_interfaces/srv/SetParameters` | GUI Apply & Save |
| `/crop_image/reload_roi` | `std_srvs/srv/Trigger` | GUI (after Apply & Save) |
| `/crop_image/clear_roi` | `std_srvs/srv/Trigger` | GUI Reset |
| `/moveit_controller/execute_welding_path` | `std_srvs/srv/Trigger` | GUI Send Path |

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

Written by the GUI when you click **✅ Apply & Save**, and read by `crop_image_node` on startup and on `~/reload_roi`.

```json
{
  "enabled": true,
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
- Reduce Kinect stream quality in `kinect2_bridge_launch.yaml` if CPU is saturated.
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

## Related Documentation

| Guide | Location |
|-------|----------|
| Capture & replay with bags | [`CAPTURE_REPLAY_GUIDE.md`](CAPTURE_REPLAY_GUIDE.md) |
| Camera calibration | [`CAMERA_CALIBRATION_GUIDE.md`](CAMERA_CALIBRATION_GUIDE.md) |
| Depth matcher deep-dive | [`DEPTH_MATCHER_GUIDE.md`](DEPTH_MATCHER_GUIDE.md) |
| Path generator | [`PATH_GENERATOR_GUIDE.md`](PATH_GENERATOR_GUIDE.md) |
| RViz visualization setup | [`RVIZ_SETUP_GUIDE.md`](RVIZ_SETUP_GUIDE.md) |
| Testing guide | [`TESTING_GUIDE.md`](TESTING_GUIDE.md) |
