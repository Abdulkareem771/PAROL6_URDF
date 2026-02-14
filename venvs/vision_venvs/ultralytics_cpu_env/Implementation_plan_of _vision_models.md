This implementation plan is designed to move your team from the data collection phase to a production-ready model. It bridges the gap between the "Wood Prototype" and the "Metal Reality."

---

# **Implementation Plan: Robotic Welding Vision System**

**Project Goal:** High-precision detection of welding seams using a two-stage computer vision pipeline.

---

## **Phase 1: The Data Pipeline (Weeks 1-3)**

The foundation of the project is the **Wood-to-Metal Transfer** strategy.

### **1.1 Data Collection (The Physical Setup)**

* **The Wood Set (Base):** Create 500 assemblies.
* **50% Natural:** Raw wood to teach structural geometry.
* **50% Primed:** Spray wood with flat grey/silver primer to simulate the monochromatic look of steel.


* **The Metal Anchor Set (Fine-Tuning):** Create 100 high-quality assemblies of actual workpieces (Butt, Tee, Corner, Edge).
* **Environment:** All photos must be taken from the 1.5m overhead mount to maintain the correct **Ground Sample Distance (GSD)**.

### **1.2 Data Augmentation (The Multiplier)**

Your script must programmatically "stretch" the dataset.

* **Geometric:** Random rotations (360°), horizontal/vertical flips, and slight shears.
* **Photometric:** Random brightness/contrast (simulating shop lights) and **Grayscale conversion** (essential for wood-to-metal transition).

---

## **Phase 2: Labeling SOP (Weeks 2-4)**

Consistency is more important than quantity. Use a tool like CVAT or LabelStudio.

### **2.1 Task A: ROI Detection (The "Where")**

* **Label:** Bounding Box.
* **Rule:** Tight boxes around the entire "workpiece" (both parts together).
* **Purpose:** This model crops the 1.5m overhead image into a high-resolution "Close-up" for the next stage.

### **2.2 Task B: Seam Extraction (The "How")**

* **Label:** Keypoint + Polyline.
* **Keypoint 1:** "Seam_Start"
* **Keypoint 2:** "Seam_End"
* **Polyline:** Trace the contact line between the parts.
* **Precision Rule:** Annotators must zoom to **400%**. In a Tee-joint, the line must be at the **root** of the intersection, not on the shadow.

---

## **Phase 3: The "Two-Stage" Model Architecture (Weeks 4-6)**

| Stage | Model Type | Input | Output |
| --- | --- | --- | --- |
| **Stage 1: Detector** | YOLOv11-OBB | Full 1.5m Image | Bounding Box Coordinates |
| **Stage 2: Estimator** | YOLOv11-Pose | Cropped ROI | Start/End Keypoints + Vector |

---

## **Phase 4: Training & Fine-Tuning (Weeks 6-8)**

1. **Step 1: Pre-training:** Train both models on the **Wood Dataset** for 100 epochs. This establishes the "concept" of a joint.
2. **Step 2: Transfer:** Freeze the "backbone" (the early layers) of the model.
3. **Step 3: Fine-Tuning:** Train for another 50 epochs using the **Metal Anchor Set** with a very low learning rate (). This "specializes" the model for metallic reflections and grain.

---

## **Phase 5: Deployment & Robot Integration (Weeks 8+)**

### **5.1 Pixel-to-Millimeter Calibration**

Since the camera is static at 1.5m, you must use a **Homography Matrix**:

1. Place a checkerboard on the welding table.
2. Map the pixel  to the robot's  coordinates.
3. The Vision System sends the **Start Point** and **End Point** as a vector to the Robot Controller.

### **5.2 Validation Logic**

The system should only trigger the weld if:

* The **Confidence Score** of the ROI is .
* The **Distance** between Start and End points matches the expected part length.

---

## **Next Steps for the Team**

* **Hardware:** Mount the camera at 1.5m and ensure the workspace is fully illuminated.
* **Software:** Initialize a GitHub repository and set up the `ultralytics` environment.

**Would you like me to write the Python script that performs the "Crop-and-Detect" pipeline logic for your Stage 1 and Stage 2 models?**
