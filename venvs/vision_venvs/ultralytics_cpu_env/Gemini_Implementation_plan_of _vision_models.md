Below is the complete implementation plan formatted as a `.md` documentation file. You can copy this directly into your project repository (e.g., `README.md` or `VISION_DOC.md`) to serve as the "Source of Truth" for your engineering team.

---

# Robotic Welding Vision System: Implementation Roadmap

## 1. Project Overview

This project aims to automate welding path detection using a two-stage computer vision pipeline. The system will detect a "workpiece" (an assembly of two metal parts) and identify the precise "seam" or path where the weld must occur.

* **Camera Configuration:** Static Overhead (1.5m Height).
* **Target Joints:** Butt, Tee, Corner, and Edge.
* **Methodology:** Wood-to-Metal Transfer Learning using a Two-Stage Pipeline.

---

## 2. System Architecture

We utilize a **Pipeline Approach** to maximize precision. Instead of one model doing everything, we split the task:

| Stage | Name | Model Type | Input | Output |
| --- | --- | --- | --- | --- |
| **Stage 1** | **ROI Detector** | Object Detection (YOLO) | Full Workspace Image | Bounding Box of Assembly |
| **Stage 2** | **Path Estimator** | Keypoint Detection | Cropped ROI | Start/End Points + Vector |

---

## 3. Phase 1: Data Collection Strategy

To overcome the lack of initial metal data, we will utilize **Domain Randomization** with wood.

### 3.1 Dataset Composition

1. **Wood Base Set (500 Images):** * Mix of natural wood and wood spray-painted with grey primer.
* Purpose: Teach the model the geometric "logic" of intersections.


2. **Metal Anchor Set (100 Images):** * Actual metal workpieces under factory lighting.
* Purpose: Fine-tune the model to recognize metallic specularity and grain.



### 3.2 Environmental Variables

* **Camera:** Use a Circular Polarizing Filter (CPL) to reduce glare.
* **Lighting:** Capture images under varying overhead light positions to simulate different times of day/shadows.

---

## 4. Phase 2: Labeling SOP (Standard Operating Procedure)

Consistency in labeling is critical for robot precision.

### 4.1 Task A: ROI (Bounding Box)

* **Label Name:** `workpiece_assembly`
* **Boundary:** Include a 5-10% padding around the pieces.
* **Negative Samples:** Label 50 images of an empty welding table as `background`.

### 4.2 Task B: Seam (Keypoints & Polylines)

* **Keypoint 1:** `seam_start`
* **Keypoint 2:** `seam_end`
* **Polyline:** `weld_path`
* **Critical Rule:** Annotators must zoom to **400%**. For **Tee Joints**, the path must be exactly at the intersection of the two planes, NOT on the shadow cast by the vertical piece.

---

## 5. Phase 3: Data Augmentation

To multiply our small dataset, the following augmentations must be applied in the training script:

* **Rotation:** 0° to 360° (Robot needs to find joints at any orientation).
* **Color:** 50% probability of **Grayscale** conversion (bridges the gap between wood and metal).
* **Lighting:** Random Brightness and Contrast adjustments (simulates metal reflections).
* **Noise:** Gaussian Blur (simulates camera vibration during robot operation).

---

## 4. Phase 4: Training & Fine-Tuning

1. **Stage 1: Base Training:** Train on the Wood Dataset for 100+ epochs.
2. **Stage 2: Weight Transfer:** Save the best weights.
3. **Stage 3: Fine-Tuning:** Initialize training with wood weights, then train on the **Metal Anchor Set** for 30-50 epochs with a low Learning Rate ().

---

## 5. Phase 5: Robot Integration

The output of Stage 2 provides pixel coordinates . These must be converted to Robot Coordinates .

1. **Calibration:** Use a Checkerboard pattern to calculate the **Homography Matrix**.
2. **Depth:** Since the camera is at a fixed 1.5m,  is constant (Table Height) unless the workpiece height changes.
3. **Validation:** If the distance between `seam_start` and `seam_end` exceeds expected part dimensions by , trigger a safety stop.

---

## 6. Maintenance & Scalability

* **Step 1:** Start with Straight Lines (Butt/Tee).
* **Step 2:** Once accuracy , introduce curved workpieces.
* **Step 3:** Collect "Edge Case" images (e.g., rusted metal, oil-coated parts) to further harden the model.

---

**Would you like me to generate the `train.py` script that handles the Grayscale and Rotation augmentations discussed in Phase 3?**