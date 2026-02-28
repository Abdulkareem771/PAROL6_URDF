# Models Testing Tool
**WeldVision GUI Tester Architecture**

This tool provides a unified, cross-OS, GUI-based environment for testing and isolating computer vision models. It operates within a Dockerized ROS2 runtime (via `parol6_dev`) but utilizes X11 forwarding to render its UI back to the host machine.

## 1. Universal Launcher (`vision_work/launcher.py`)
The universal entry point for the toolkit.
- Provides a simple Tkinter window to launch either the **YOLO Detector** or the **ResUNet Segmenter**.
- Built to be expandable; new models or testing tools can easily be attached here.
- Invokes the chosen sub-scripts using `subprocess.run()`.

## 2. YOLO Tester (`vision_work/yolo_training/yolo_gui.py`)
A custom PyTorch/Ultralytics GUI for isolated testing of bounding box and segmentation models on individual images or large batches.

**Key Features:**
- **Target Tag Filter:** Only displays classes that match the given text string (e.g., typing "seam" hides all other detected objects).
- **Dynamic Mask Coloring:** A text input box `Drawing Color (B,G,R)` lets users dynamically paint all bounding boxes and segmentation masks in any RGB color (e.g., `0,255,0` for Green) without re-running the model.
- **Batch Cropping:** Takes an entire directory of mixed images and exports *only* the matching detected objects into a time-stamped `yolo_results_...` folder. The outputs are cleanly sequentially numbered `0001_original_name.jpg`.

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
