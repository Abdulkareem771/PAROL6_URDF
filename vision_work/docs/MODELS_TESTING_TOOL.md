# Models Testing Tool
**WeldVision GUI Tester Architecture — v2**

A unified R&D testing environment for vision models operating inside the `parol6_dev` Docker container via X11 forwarding.

## Tool Suite Overview

| Tool | File | Type |
|---|---|---|
| YOLO Tester (legacy) | `yolo_training/yolo_gui.py` | Tkinter — frozen |
| ResUNet Tester (legacy) | `resunet_training/weld_seam_gui.py` | Tkinter — frozen |
| Manual Annotator (legacy) | `tools/manual_annotator.py` | Tkinter — frozen |
| **Pipeline Prototyper** | `tools/pipeline_prototyper.py` | PySide6 |
| **Script Sandbox** | `tools/script_sandbox.py` | PySide6 |
| **Mask Painter** | `tools/mask_painter.py` | PySide6 |
| **YOLO Inspector** | `tools/yolo_inspector.py` | PySide6 |
| **Seam Inspector** | `tools/seam_inspector.py` | PySide6 |
| **Mask Pipeline Tester** | `tools/mask_pipeline_tester.py` | PySide6 |
| **Pipeline Studio** | `tools/pipeline_studio.py` | PySide6 |
| **Annotation Studio** | `tools/annotation_studio.py` | PySide6 |

All PySide6 tools share `BaseVisionApp` from `vision_work/core/qt_base_gui.py`.

## 1. Universal Launcher (`vision_work/launcher.py`)
The universal entry point for the toolkit.
- Displays two sections: legacy Tkinter tools and the new PySide6 Next-Gen tools.
- Invokes each tool as an independent subprocess, so tools never interfere with each other.


## 2. YOLO Tester (`vision_work/yolo_training/yolo_gui.py`)
A custom PyTorch/Ultralytics GUI for isolated testing of bounding box and segmentation models on individual images or large batches.

**Key Features (Main Tester Tab):**
- **Target Tag Filter:** Only displays classes that match the given text string (e.g., typing "seam" hides all other detected objects).
- **Semantic ID Labels:** Every detected object bounding box label is prefixed with its index (e.g., `[ID: 1] Seam 0.95`) allowing for specific targeting.
- **Dynamic Mask Coloring:** 
  - *Global Color:* A text input box `Color (B,G,R)` lets users dynamically paint all bounding boxes and segmentation masks in any RGB color (e.g., `0,255,0` for Green) without re-running the model.
  - *Visual Color Picker:* A 🎨 button beside the color text box opens the native OS color wheel. The chosen color is automatically converted to BGR and written into the entry field, eliminating the need to manually type values.
  - *Dictionary Mapping:* Users can assign specific colors to specific object IDs using a dictionary syntax (e.g., `0: 255,0,0; 1: 0,255,0` paints object 0 red and object 1 green).
  - *Auto Multi-Color Toggle:* A simple checkbox that auto-assigns a visually distinct, hardcoded color palette to all objects based on their IDs for immediate visual separation.
- **Solid Color Mask & Batches:** Supports painting the object totally solid in bounding box or segmentation format, and allows batch-cropping objects across an entire directory.
- **Segmentation Polygon Mode:** Synthesizes raw `masks.xyn` geometry into geometrically sharp, semi-transparent polygon overlays natively drawn with OpenCV, providing a cleaner visualization than rasterized masks.

**Key Features (Advanced Ops Tab):**
- **Dual Tag Filtering:** Track exactly 2 separate tags (`Tag 1` and `Tag 2`) simultaneously with customizable individual RGB colors. Requires switching the View Mode to `Dual Tag Mask`.
- **Advanced Batch Data Generation:** Allows using the loaded YOLO model to mass-generate synthetic data formats for other machine learning architectures:
    * `Crop Objects (Tag 1 & 2)`: Sequentially crops dual-classes across a folder.
    * `Export YOLO BBox Annotations (.txt)`: Auto-annotates a folder producing standard YOLO bounding box labels (`class_id x_center y_center w h`).
    * `Export YOLO Seg Annotations (.txt)`: Auto-annotates a folder producing standard YOLO segmentation polygon labels (`class_id x1 y1 x2 y2 ...`).
    * `Export Dual Color Masks`: Outputs completely solid RGB masks of your detected objects on a black background (ideal for UNet dataset generation).
    * `Export Binary Masks`: Same as above but outputs strict Black/White boolean masks.

## 3. ResUNet Tester (`vision_work/resunet_training/weld_seam_gui.py`)
A dedicated custom GUI for the ResUNet weld-seam segmentation pipeline.

**Key Features:**
- **Interactive View Toggles:** Allows switching the live canvas between Original, Heat Map, Mask, Skeleton, and colored Overlay.
- **Matplotlib 3-Panel Export:** Saves the results as a professional academic-style 3-panel figure showing Original, Seam Mask Overlay, and the extracted Centerline.
- **Batch Mask Generator:** Analyzes an entire folder of input images and exports the results sequentially. The user can specify a custom `(B,G,R)` color and choose between rendering the **1-px Skeleton** centerline or the **Whole Mask Area** to the output folder.

## Shared Philosophy & Design
Both tools share the same core structural design to ensure the user experience is identical regardless of the underlying model:
- **V1 Single-Pane GUI:** A wide dark sidebar on the left and a single large viewing canvas on the right.
- **Clipboard Access:** Both tools bypass standard Tkinter limitations by invoking `xclip` directly, allowing users in Linux/X11 containerized environments to seamlessly use `Ctrl+V` to paste images directly from their Host OS clipboard into the tool.
- **Live Sliders:** Confidence threshold sliders trigger instant canvas redraws without forcing the underlying heavy ML models to re-evaluate the image data.

## 4. Pipeline Prototyper (`vision_work/tools/pipeline_prototyper.py`)
A modern PySide6 tool built on `BaseVisionApp` for testing complete vision pipelines end-to-end without writing boilerplate ROS 2 C++ nodes.

### 4-Slot Architecture
Each stage is independently toggleable. Slots you don't need are simply left empty.

| Slot | Widget | Description |
|---|---|---|
| **Input** | File Browser / Text Box | Load a local image folder **or** type a live ROS 2 topic (e.g. `/kinect2/image_raw`) to subscribe to a live camera stream |
| **ML Inference** | `.pt` File Picker | Load any Ultralytics YOLO or ResUNet model. Inference is triggered automatically when an image is loaded or a ROS frame arrives |
| **Script Injector** | `.py` File Picker | Dynamically load any Python script as a black-box processing node. Zero modifications needed. Auto-detects `segment_blocks()` and `process_image()` function signatures |
| **ROS Output** | Text Box + Button | Type a ROS 2 topic name and click "Publish" to stream the final processed canvas directly to the robot network |

### Real-time Profiling
A live FPS and latency counter in the sidebar measures the combined Slot 2+3 execution time, giving exact data for whether the pipeline can run on the target hardware (Jetson / Mini PC) in real-time.

### Design Notes
- **Zero teammate changes required:** Uses `importlib.util` dynamic loading — teammates never need to add ROS boilerpate or UI code to their scripts.
- **Graceful ROS degradation:** If `rclpy` is not sourced, all 4 slots still work in pure offline image mode.
- **Launched from:** The main `launcher.py` hub via the 🔮 Pipeline Prototyper button.
