# Capture GUI — Developer Guide

## Overview

The Capture GUI is a Qt5 testing tool that lets you test the vision
pipeline **one frame at a time** — you freeze a single image from the
live camera, run the full red-line detection on it, and optionally
trigger 3D depth matching, all without the pipeline running continuously
and flooding MoveIt.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    capture_gui  (ROS2 Node)                     │
│                                                                 │
│  ┌────────────────┐        ┌───────────────────────────────┐   │
│  │ Qt5 Main Window │ ←──── │ ROSSignals (Qt signals)       │   │
│  └────────────────┘        └───────────────────────────────┘   │
│                                        ▲                        │
│                              CaptureGUINode (background thread) │
│                              subscribes/publishes/service calls │
└─────────────────────────────────────────────────────────────────┘
         │                          │                    │
         ▼                          ▼                    ▼
 /kinect2/qhd/         /capture_gui/frozen_frame    /kinect2/qhd/
 image_color_rect       (relay topic)                image_depth_rect
         │                          │
         │               ┌──────────┴──────────┐
         │               │  red_line_detector   │
         │               │  (capture_mode=True) │
         │               │  input remapped to   │
         │               │  frozen_frame topic  │
         │               └──────────────────────┘
         │                          │
         │                /vision/weld_lines_2d
         │                          │
         │               ┌──────────┴──────────┐
         │               │    depth_matcher     │
         │               │  (capture_mode=True) │
         │               └──────────────────────┘
         │                          │
         │                /vision/weld_lines_3d
```

### Capture Mode vs Streaming Mode

| Feature | Streaming (default) | Capture Mode |
|---|---|---|
| Activation | `--ros-args -p capture_mode:=false` (default) | `--ros-args -p capture_mode:=true` |
| Trigger | Continuous image callback | `/red_line_detector/capture` Trigger service |
| Frames processed | Every arriving frame | Exactly ONE per service call |
| MoveIt impact | Continuous, may queue up | None until explicitly triggered |
| Used by | Production pipeline | Testing GUI |

---

## Files Added / Modified

| File | Change |
|---|---|
| `parol6_vision/red_line_detector.py` | `capture_mode` param + `/red_line_detector/capture` service |
| `parol6_vision/depth_matcher.py` | `capture_mode` param + `/depth_matcher/capture` service |
| `parol6_vision/capture_gui.py` | **New** — Qt5 GUI node |
| `launch/capture_mode.launch.py` | **New** — convenience launch |
| `setup.py` | Registered `capture_gui` entry point |

---

## Usage

### Option A — Full launch (recommended)

```bash
# Inside Docker, ROS2 sourced
# Terminal 1: start infrastructure (Kinect bridge assumed running)
ros2 launch parol6_vision capture_mode.launch.py

# Terminal 2: start the GUI
ros2 run parol6_vision capture_gui
```

### Option B — Manual node control (GUI handles it)

```bash
# Just launch the GUI — use Start/Stop buttons inside it
ros2 run parol6_vision capture_gui
```

### Building

```bash
cd /workspace
colcon build --packages-select parol6_vision --symlink-install
source install/setup.bash
```

---

## GUI Walkthrough

```
┌──────────────────────────────────────────────────────────────┐
│ 🤖 PAROL6 Vision — Capture Testing GUI                      │
├──[Hz bars: Camera | Debug | 2D Lines | 3D Lines]─────────────┤
├──────────────┬───────────────────────────────────────────────┤
│ Node Controls│  [📷 Live Preview] [🧊 Frozen Frame]          │
│  ▶ Start Det │  [🔍 Detection Result]    ← tab panel        │
│  ■ Stop Det  │                                               │
│  ▶ Start DM  │  (camera stream or frozen/detection images)   │
│  ■ Stop DM   │                                               │
├──────────────┤                                               │
│ Capture &    │                                               │
│  Detection   │                                               │
│ [📷 Capture] │                                               │
│ [🔁 Re-detect│                                               │
│ [🔍 Depth]   │                                               │
│ [💾 Save]    │                                               │
├──────────────┤                                               │
│ Stats table  │                                               │
├──────────────┤                                               │
│ 3D Result    │                                               │
├──────────────┴───────────────────────────────────────────────┤
│ [Log: timestamped node/GUI/service log lines]                │
└──────────────────────────────────────────────────────────────┘
│ Status: Detector 🟢 Running | DM ⬛ Stopped | Captures: 3  │
```

### Typical testing workflow

1. **Start the Kinect bridge** (outside the GUI, as usual).
2. Click **▶ Start Detector** — launches `red_line_detector` with
   `capture_mode=true`, remapped to read from the frozen-frame relay topic.
3. Point the camera at your red marker / test scene.
4. Click **📷 Capture Frame** — the GUI grabs the next arriving camera
   frame, publishes it to `/capture_gui/frozen_frame`, and immediately
   calls `/red_line_detector/capture`.  
   The detector processes **only that one frozen frame** and publishes
   its detections.
5. The **🔍 Detection Result** tab updates with the debug overlay image.
   The **Detection Stats** table shows confidence, pixel count, bbox.
6. Click **🔁 Re-detect (same frame)** to re-run detection on the exact
   same image with different parameters (e.g., after tuning HSV in YAML).
7. Click **▶ Start Depth Matcher** if you also want 3D output.
8. Click **🔍 Match Depth** — calls `/depth_matcher/capture`.  
   The depth matcher reads the current `/vision/weld_lines_2d` and the
   live depth topic (depth of the scene hasn't changed) and publishes
   `/vision/weld_lines_3d`.
9. Click **💾 Save Frozen Frame** to write the captured PNG for later.
10. Click **📷 Capture Frame** again for a new scene.

---

## Service API (CLI testing without the GUI)

```bash
# Trigger red-line detection on next frame
ros2 service call /red_line_detector/capture std_srvs/srv/Trigger {}

# Trigger depth matching
ros2 service call /depth_matcher/capture std_srvs/srv/Trigger {}
```

---

## Hz Indicator Colours

| Colour | Meaning |
|---|---|
| 🟢 Green  | ≥ 20 Hz (healthy) |
| 🟡 Yellow | 5–20 Hz |
| 🟠 Orange | 0–5 Hz (slow) |
| ⬛ Grey   | 0 Hz (no messages) |

---

## Troubleshooting

**Capture button does nothing / detection times out**  
→ Check that the detector node is running (▶ Start Detector button).  
→ Check that `/kinect2/qhd/image_color_rect` is at > 0 Hz in the Hz bar.

**Detection result shows nothing / 0 lines**  
→ Check HSV parameters in `config/detection_params.yaml` and re-tune.  
→ Use `ros2 run parol6_vision hsv_inspector` to sample HSV from the image.

**Match Depth fails / timeout on `lines`**  
→ Ensure a successful capture + detection was done first (weld_lines_2d must have data).  
→ Check that `/kinect2/qhd/image_depth_rect` is publishing.

**GUI crashes / `No module named 'PyQt5'`**  
```bash
apt-get install -y python3-pyqt5
```

**`capture_gui` not found after build**  
```bash
colcon build --packages-select parol6_vision --symlink-install
source install/setup.bash
```
