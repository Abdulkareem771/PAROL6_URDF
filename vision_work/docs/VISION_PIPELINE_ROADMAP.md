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

## 2. Recommended Tooling Additions

To support this extensive R&D phase, the testing environment needs to evolve from a "Viewer" into an "Interactive Pipeline Prototyping Engine." 

### Required Features:
1. **Manual Drawing / 'Paint' Mode:** 
   * Ability to load a raw image and manually draw/paint red lines or colored masks natively on the canvas. 
   * Essential for Approach A (testing Image Processing limits without AI) and Approach C (human-in-the-loop).
2. **Multi-Model Pipeline Chaining (The "Pass-to-UNet" Button):**
   * Currently, YOLO and ResUNet are tested in absolute isolation. We need a way to take a YOLO generated crop and instantly pipe it into a loaded ResUNet model on the exact same screen to test the "Hierarchical Pipeline" (Approach B).
3. **Interactive "Clickable" Detections:**
   * If YOLO detects 3 seams, the GUI should pause. The user clicks the correct one on the canvas, and *only that specific crop* gets passed to the next stage or saved.
4. **Automated Data Augmentation Exporter:**
   * Automatically generate rotated/brightened/flipped versions of the YOLO bounding boxes during the Advanced Batch Auto-Annotation to instantly 5x the dataset size.
5. **Bounding Box / Polygon Nudging:**
   * The ability to drag the corners of a YOLO prediction *before* saving it to disk as an annotation. This turns YOLO into an AI-Assisted annotation tool for building future datasets.

---

## 3. Architectural Decision: Monolithic Tool vs. Launcher Modularity

**The Question:** *Should we add all these features into a single, massive "God Tool", or keep the YOLO/ResUNet/etc. tools separate and simply add more buttons to `launcher.py`?*

**The Recommendation: The Modular Launcher Pattern (Keep them separate!)**

Your instincts are exactly right. Building "one tool to do everything" (Monolithic) is a bad idea for R&D. 
1. **Code Bloat:** A single Python file trying to manage PyTorch, Ultralytics YOLO, ResUNet architectures, OpenCV painting, and Tkinter UI state simultaneously will become a massive, unmaintainable nightmare (3000+ lines of code).
2. **Dependency Clashes:** In the future, you might want to test a model that requires completely different Python versions or conflicting library versions (e.g., TensorFlow vs PyTorch). 
3. **The Unix Philosophy:** "Do one thing and do it well." 

**How we proceed:**
We will leave `yolo_gui.py` strictly as the "YOLO Expert" and `weld_seam_gui.py` as the "ResUNet Expert". 
Instead, we will build **New, Specialized Mini-Tools** and attach them as new buttons to the `launcher.py`. 

For example, we will build a dedicated `pipeline_prototyper.py` that imports the weights of *both* models, but features a UI specifically designed for chaining them together and drawing on them. Or a dedicated `manual_annotator.py` designed purely for Approach C. 

The Universal Launcher becomes the "Desktop Environment" for your R&D suite.
