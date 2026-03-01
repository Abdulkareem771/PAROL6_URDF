# Vision Pipeline Automation & Testing Roadmap

## 1. User Pipeline Goals & Exploration Paths

The project has entered an advanced R&D phase requiring the capability to rapidly test, verify, and switch out computer vision approaches for locating the optimal welding path. The current overarching strategies are:

### Approach A: Physical Masks vs. AI Masks (The "Blind" Image Processing Node)
1. **Physical Ground Truth Validation:** Apply physical distinct colored masks to the real workpieces in the lab. Use a pure OpenCV/Image Processing node to extract the weld path based exclusively on color thresholds.
2. **AI Simulation Substitution:** Train a YOLO segmentation model to identify the two bare workpieces (e.g., class: `workpiece`).
3. **Synthetic Masking:** Have the AI artificially "paint" the detected workpieces with the exact same RGB mask colors used in Step 1.
4. **Integration:** Feed the AI-colored image into the exact same OpenCV Image Processing node from Step 1. The node should successfully extract the path without knowing whether the colors were painted physically in the lab or digitally by the AI.

### Approach B: Hierarchical AI Crop & Refine (The "Funnel" Method)
1. **Macro Detection:** Train YOLO to identify the general `workpiece` objects within the raw camera feed.
2. **Data Generation Sandbox:** Run YOLO over a massive video feed and mass-extract cropped regions of just the workpieces.
3. **Micro Detection:** Train a *secondary* model (either another YOLO or a ResUNet) specifically on those cropped images to identify the exact "Region of Contact" (the seam area).
4. **Path Extraction:** Run ResUNet strictly on the "Region of Contact" crop to extract the exact contour/centerline.
5. **Red Line Injection:** Artificially draw a red line over the predicted path and pass it to a pre-existing "Red Line Detector" node.

### Approach C: Semi-Supervised Human-in-the-Loop
1. Import a raw image into a GUI.
2. A human operator *manually* draws a red line over the weld seam (or selects the "best" prediction out of multiple YOLO candidates).
3. The GUI passes this human-verified image directly to the downstream execution node.
4. **Purpose:** Fast prototyping of downstream robot kinematics without waiting for perfect ML models to train.

---

## 2. Tooling Status

| Feature | Status | Tool |
|---|---|---|
| Manual Drawing / Paint Mode | ✅ Implemented | `tools/manual_annotator.py` |
| Multi-Model Pipeline Chaining | ✅ Implemented | `tools/pipeline_prototyper.py` |
| Two-Way Live ROS Topic Streaming | ✅ Implemented | `tools/pipeline_prototyper.py` |
| External Script Injection (black-box node) | ✅ Implemented | `tools/pipeline_prototyper.py` |
| Live FPS / Latency Profiling | ✅ Implemented | `tools/pipeline_prototyper.py` |
| Interactive Clickable Detections | 🔲 Planned | Future add-on to prototyper |
| BB / Polygon Nudging | 🔲 Planned | Future annotation tool |
| Automated Data Augmentation Exporter | 🔲 Planned | YOLO Advanced Tab add-on |

---

## 3. Architectural Decision: Modular Launcher Pattern ✅

**The Question:** *Should we add all these features into a single, massive "God Tool", or keep the YOLO/ResUNet/etc. tools separate and simply add more buttons to `launcher.py`?*

**Decision: Modular Launcher Pattern — Implemented.**

We leave `yolo_gui.py` strictly as the "YOLO Expert" and `weld_seam_gui.py` as the "ResUNet Expert". Each R&D need gets its own lean, specialized tool registered in the Universal Launcher.

**Current tools in the suite (`launcher.py`):**
- 🔍 `yolo_training/yolo_gui.py` — YOLO object detection tester
- 〰️ `resunet_training/weld_seam_gui.py` — ResUNet seam segmenter
- 🖍️ `tools/manual_annotator.py` — Human-in-the-loop red line painter
- 🔮 `tools/pipeline_prototyper.py` — 4-slot linear pipeline testing harness with two-way ROS 2 integration

All tools share a common `BaseVisionApp` PySide6 base class in `core/qt_base_gui.py`.
