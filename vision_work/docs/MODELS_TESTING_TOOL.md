# Models Testing Tool
**WeldVision GUI Tester Architecture — v3**

A unified R&D testing environment for vision models operating inside the `parol6_dev` Docker container via X11 forwarding.

---

## ⚡ Quick Start for Teammates

> **New here? Jump straight to the [Dependencies & Installation](#0-dependencies--installation) section below.**

```bash
# Launch the tool hub
cd /workspace/vision_work
python3 launcher.py
```

---

## 0. Dependencies & Installation

All tools run inside the `parol6_dev` Docker container. After entering the container, install the following.

### Core (required by every tool)

```bash
pip install PySide6 opencv-python numpy
```

| Package | Why |
|---|---|
| `PySide6` | All next-gen GUI tools are built on this |
| `opencv-python` | Image loading, drawing, HSV masking |
| `numpy` | Array ops, mask maths |

> ⚠️ **NumPy 2.x and `cv_bridge` are incompatible** — the ROS `cv_bridge` package was compiled against NumPy 1.x. If you see `_ARRAY_API not found`, run:
> ```bash
> pip install "numpy<2"
> ```

---

### ML Inference (YOLO / segmentation tools)

```bash
pip install ultralytics
```

| Package | Why |
|---|---|
| `ultralytics` | YOLO v8/v11 detection + segmentation |

> YOLO Inspector, Mask Pipeline Tester, Batch YOLO Exporter, Annotation Studio, Pipeline Studio, Pipeline Prototyper all need this.

---

### Seam Inspector (ResUNet)

```bash
pip install torch torchvision scikit-image
```

| Package | Why |
|---|---|
| `torch` + `torchvision` | ResUNet model inference |
| `scikit-image` | Skeleton extraction (falls back to OpenCV if missing) |

---

### ROS 2 Features (optional — degrades gracefully if missing)

Source ROS 2 Humble and make sure `cv_bridge` is installed:

```bash
source /opt/ros/humble/setup.bash
sudo apt install ros-humble-cv-bridge
pip install "numpy<2"   # fix NumPy 2.x ABI incompatibility
```

> Tools with ROS features: **Pipeline Prototyper**, **Pipeline Studio**, **Mask Pipeline Tester**.  
> All three run fine in offline mode if `rclpy` is not available — ROS controls are simply greyed out.

---

### Legacy Tools (Tkinter — already in container)

```bash
# No extra installs needed — Tkinter ships with Python 3
# But matplotlib is needed for the ResUNet 3-panel export:
pip install matplotlib
```

---

### Full one-liner (everything at once)

```bash
pip install PySide6 opencv-python "numpy<2" ultralytics torch torchvision scikit-image matplotlib
```

---

## Tool Suite Overview

| # | Tool | File | Framework | Requires |
|---|---|---|---|---|
| — | **Universal Launcher** | `launcher.py` | Tkinter | nothing extra |
| 1 | YOLO Tester *(legacy, frozen)* | `yolo_training/yolo_gui.py` | Tkinter | `ultralytics` |
| 2 | ResUNet Tester *(legacy, frozen)* | `resunet_training/weld_seam_gui.py` | Tkinter | `torch`, `matplotlib` |
| 3 | Manual Path Annotator *(legacy, frozen)* | `tools/manual_annotator.py` | Tkinter | `opencv-python` |
| 4 | **Pipeline Prototyper** | `tools/pipeline_prototyper.py` | PySide6 | `ultralytics`, ROS optional |
| 5 | **Script Sandbox** | `tools/script_sandbox.py` | PySide6 | `opencv-python` |
| 6 | **Mask Painter** | `tools/mask_painter.py` | PySide6 | `opencv-python` |
| 7 | **YOLO Inspector** | `tools/yolo_inspector.py` | PySide6 | `ultralytics` |
| 8 | **Seam Inspector** | `tools/seam_inspector.py` | PySide6 | `torch`, `scikit-image` |
| 9 | **Mask Pipeline Tester** | `tools/mask_pipeline_tester.py` | PySide6 | `ultralytics`, ROS optional |
| 10 | **Pipeline Studio** | `tools/pipeline_studio.py` | PySide6 | `ultralytics`, ROS optional |
| 11 | **Annotation Studio** | `tools/annotation_studio.py` | PySide6 | `ultralytics` |
| 12 | **Batch YOLO Exporter** | `tools/batch_yolo.py` | PySide6 | `ultralytics` |

All PySide6 tools share `BaseVisionApp` from `vision_work/core/qt_base_gui.py`.

---

## 1. Universal Launcher (`vision_work/launcher.py`)
The scrollable entry point for the entire toolkit.
- Two sections: **LEGACY TOOLS** (frozen Tkinter apps) and **NEXT-GEN TOOLS** (PySide6 apps).
- Mouse-wheel scrolling supported — add as many tools as needed.
- Each tool is launched as an independent subprocess so crashes don't affect the launcher.

---

## 2. YOLO Tester — legacy (`yolo_training/yolo_gui.py`)

> ⚠️ **Frozen — do not modify.** Use **YOLO Inspector** for new work.

A Tkinter GUI for testing YOLO bounding box + segmentation models.

**Key Features:**
- **Target Tag Filter** — show only classes matching a keyword
- **Semantic ID Labels** — every box is prefixed `[ID:i]`  
- **Visual Color Picker** — "Pick" button opens OS color wheel, auto-converts to BGR
- **Auto Multi-Color** — auto-assigns palette colours by object ID
- **Segmentation Polygon Mode** — clean polygon overlays from `masks.xyn`
- **Advanced Ops Tab** — batch crop, YOLO bbox/seg annotation export, dual colour mask export

---

## 3. ResUNet Tester — legacy (`resunet_training/weld_seam_gui.py`)

> ⚠️ **Frozen — do not modify.** Use **Seam Inspector** for new work.

Tkinter GUI for ResUNet weld-seam segmentation.

**Key Features:**
- View toggles: Original / Heatmap / Mask / Skeleton / Overlay
- Matplotlib 3-panel export (Original + Mask + Centerline)
- Batch mask generator across a folder

---

## 4. Pipeline Prototyper (`tools/pipeline_prototyper.py`)
Test complete vision pipelines end-to-end in 4 slots without writing ROS nodes.

| Slot | Input | Description |
|---|---|---|
| 1 — Input | File / Topic | Local image or live ROS topic |
| 2 — ML | `.pt` file | YOLO / ResUNet inference |
| 3 — Script | `.py` file | Any Python script — auto-detects `process_image()` and `segment_blocks()` |
| 4 — ROS Out | Topic name | Publish result to ROS |

Real-time FPS + latency readout. Graceful ROS degradation if `rclpy` not sourced.

---

## 5. Script Sandbox (`tools/script_sandbox.py`)
Load **any** Python image-processing script and run it on a single image.  
Live console panel shows stdout/stderr. Tries `process_image()` / `segment_blocks()` first, falls back to subprocess.

---

## 6. Mask Painter (`tools/mask_painter.py`)
Manual RGBA brush for painting colour masks on images.

- Adjustable brush size
- Quick colour buttons (Green / Red / Blue) + full QColorDialog
- Undo stack (up to 20 strokes)
- Export composite image or raw RGBA mask PNG

---

## 7. YOLO Inspector (`tools/yolo_inspector.py`)
Modern PySide6 replacement for the legacy YOLO Tester.

- Load any `.pt` / `.onnx` YOLO weights
- Real-time confidence threshold slider
- View modes: Original / Bounding Boxes / Segmentation Mask / Polygon / Solid Mask
- Per-object-ID colour picker (double-click any detection in the list)
- **Hide Labels / ID Text** checkbox — suppresses all overlay text
- Auto Multi-Color palette
- Live inference latency readout

---

## 8. Seam Inspector (`tools/seam_inspector.py`)
Modern PySide6 replacement for the legacy ResUNet Tester.

- Load `.pth` ResUNet weights (same format as legacy tool)
- Binary threshold slider (live)
- View modes: Original / Heatmap / Binary Mask / Skeleton / Colour Overlay
- Batch folder → masks export
- Performance readout (ms per frame)

---

## 9. Mask Pipeline Tester (`tools/mask_pipeline_tester.py`)
End-to-end pipeline: **YOLO → Colour Mask → HSV Path Detection**.

Implements the exact same logic as `detect_path.py` inline — no modification of the teammate's script needed.

**Pipeline steps:**
1. Load image (file or grab from ROS topic)
2. YOLO segmentation — each detected ID is painted its assigned colour
   - Default: ID 0 = Green (side A), ID 1 = Red (side B)
3. HSV masking extracts green and red regions
4. Bounding-box intersection = seam path
5. Shown in a 4-panel canvas: Original+Masks / G mask / R mask / Result

**Path Visualisation modes:**

| Mode | What is drawn |
|---|---|
| **Rectangle** | Yellow bounding box around the intersection |
| **Centerline** | Single horizontal line through the bbox centre |
| **Band** | Filled horizontal strip, configurable pixel width |

**Advanced ROS section** (opt-in, greyed out if rclpy missing):
- **Grab One Frame** — subscribes to a topic, receives one frame, auto-unsubscribes
- **Publish result** — publishes the annotated result on every pipeline run

**Display Options:**
- **Hide Labels / ID Text** — suppresses all text overlays (blocks, seam label, ID text)

---

## 10. Pipeline Studio (`tools/pipeline_studio.py`)
Enhanced Pipeline Prototyper with intermediate previews and A/B comparison.

- **Tabbed canvas** — separate tabs for Input / Model A / Model B / Script Out / Final
- **A/B Comparison** — load two YOLO weights and compare outputs side-by-side
- Per-slot latency breakdown in the sidebar
- Full two-way ROS 2 integration (subscribe + publish)

---

## 11. Annotation Studio (`tools/annotation_studio.py`)
YOLO-assisted semi-automatic label editor.

1. Load image → Run YOLO → boxes are pre-filled on the canvas
2. Click to select a box; drag to move it
3. Draw new boxes manually (drag on canvas)
4. Assign class name + ID per box from the sidebar
5. Export as standard **YOLO `.txt`** annotation or annotated PNG

---

## 12. Batch YOLO Exporter (`tools/batch_yolo.py`)
Select a folder of images, apply YOLO, and batch-save results.

**Target ID filter** — type which detection IDs to export (e.g. `0,1`) or leave blank for all.

**Export modes:**

| Mode | Output |
|---|---|
| **Crop** | Each detected object cropped and saved individually, sorted by class name |
| **Solid Mask** | Full image with the bbox region filled in a chosen colour |
| **Seg Mask (B&W)** | Per-pixel segmentation mask as grayscale PNG |
| **Colour Mask** | Segmentation region painted in a chosen colour on black background |
| **Annotated Image** | Full image with boxes + labels drawn |

Progress bar + background thread — UI never freezes.

---

## Shared Design Philosophy

All next-gen PySide6 tools share:
- **`BaseVisionApp`** base class (`core/qt_base_gui.py`) — consistent dark sidebar + canvas layout
- **Graceful degradation** — ROS, PyTorch, Ultralytics, scikit-image are all optional imports; missing any one does not crash the tool
- **Independent subprocess launch** — tools never share state with the launcher
- **Modular, single-responsibility** — each tool does one thing well
