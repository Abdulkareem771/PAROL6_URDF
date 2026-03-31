# ArUco & Eye-to-Hand Calibration Guide

## Overview
The PAROL6 vision pipeline uses an **Eye-to-Hand** camera setup, meaning the Kinect v2 camera is mounted statically in the workspace, looking at the robot. 

To map what the camera sees (pixels) to where the robot can move (meters), we must precisely determine the geometric relationship between the robot's base (`base_link`) and the camera's lenses (`kinect2_link`). We accomplish this automatically using an **ArUco marker** and a specialized calibration node string.

---

## 1. How It Works

The calibration routine mathematically solves the transformation **`base_link` $\to$ `kinect2_link`**.
It does this in real-time by linking three vectors:

1. **`base_link` $\to$ `detected_marker_frame` (Known Physical Position)**
   The user physically places an ArUco marker in the robot workspace and measures its exact distance from the robot's zero-point (`base_link`) using a ruler or CAD, and inputs this into the calibrator. 
   *(e.g., X=0.623m, Y=0.080m, Z=0.234m)*

2. **`kinect2_ir_optical_frame` $\to$ `detected_marker_frame` (Detected Camera Position)**
   The `aruco_detector` node scans the live camera feed, finds the marker, and publishes its 3D pose relative to the camera lens.

3. **`base_link` $\to$ `kinect2` (Computed Camera Base Transform)**
   The `eye_to_hand_calibrator` node averages $N$ samples of the detected marker. It then algebraically computes where the camera *must* be located in order to see the marker at that angle and distance. 
   Finally, it factors in the internal geometry of the Kinect v2 (`kinect2_bridge` applies a -90° roll from the mount to the lens) to ensure the result is perfectly level.

The output is written to `~/.parol6/camera_tf.yaml` and is instantly applied by the `camera_tf_enforcer` node.

---

## 2. The Components 

The calibration system consists of three distinct components:

### A. The Detector (`aruco_detector.py`)
A lightweight, dependency-free Python node that uses OpenCV's `cv2.aruco` library.
* **Input Topics:** `/kinect2/sd/image_color_rect`, `/kinect2/sd/camera_info`
* **Output:** Publishes a live TF from `kinect2_ir_optical_frame` $\to$ `detected_marker_frame`.
* **Features:** Sub-pixel corner refinement, adaptive thresholding, and built-in image rectification checks.

> [!NOTE] 
> The system also supports using the external ROS package `aruco_ros single` as a drop-in replacement via the GUI toggle. However, the custom `aruco_detector.py` is recommended for guaranteed Docker compatibility.

### B. The Calibrator (`eye_to_hand_calibrator.py`)
The mathematical engine of the calibration process.
* **Input:** Subscribes to the live TF tree to watch the marker frame.
* **Behavior:** Collects $N$ samples (default: 20) to filter out camera depth noise. It uses `scipy.spatial.transform.Rotation` to compute a mathematically valid *Fréchet mean* of the quaternions (standard arithmetic averaging would corrupt the rotation manifold).
* **Output:** Safely saves the final matrix to `~/.parol6/camera_tf.yaml`.

### C. The Enforcer (`camera_tf_enforcer.py`)
A purely operational node that runs in the background of your live pipeline.
* **Behavior:** Continually reads `camera_tf.yaml` and broadcasts it to the TF tree using a `TransformBroadcaster` at 100 Hz.
* **Purpose:** This dynamically links the isolated camera TF tree to the robot's MoveIt TF tree. By using an enforcer rather than a static launch file, we can re-calibrate and update the camera's position on-the-fly without ever restarting the system.

---

## 3. Performing a Calibration via GUI

The **Vision Pipeline GUI** (`vision_pipeline_gui.py`) provides a dedicated **"📷 Cam Calibrate"** tab that automates the entire process.

### Step 1: Physical Setup
1. Print a standard ArUco marker (DICT_ARUCO_ORIGINAL) and attach it to a flat surface in the workspace.
2. Measure its exact center relative to the robot's base.
3. Keep the marker well-lit and unobstructed.

### Step 2: Configure the GUI
Open the PAROL6 Vision GUI and navigate to the **"📷 Cam Calibrate"** tab. Scroll down to the **"🎯 ArUco Auto-Calibration"** box.
1. **Backend Node:** Choose "Custom OpenCV (`aruco_detector`)".
2. **Marker ID:** Enter the ID printed on your ArUco tag (e.g., `206`).
3. **Size (mm):** Enter the exact physical dimension of the printed black square (e.g., `156.8`).
4. **Known marker pos:** Input your ruler measurements (X, Y, Z in meters) from Step 1.

### Step 3: Run Calibration
1. Click **"▶️ Run ArUco Calibration"**.
2. The GUI will launch both the detector and calibrator. Watch the progress bar as it collects image samples.
3. Once completed, the raw math results will print in the log window.

### Step 4: Save & Apply
1. Click **"✅ Save & Enforce New Frame"**.
2. The GUI will write the result to your YAML file and immediately start the `camera_tf_enforcer`.
3. Check RViz — the point cloud and camera frames should snap perfectly into alignment with the robot base.

---

## 4. Troubleshooting & Best Practices

> [!WARNING]
> **TF Toggling / Frame Jittering in RViz**
> If your camera frame appears to jump rapidly between two positions in RViz, you have a duplicate parent loop. 
> **Fix:** Ensure `publish_camera_tf:=false` is set in your `demo.launch.py` so the launch file does not fight the live `camera_tf_enforcer`.

> [!TIP]
> **Tilted Point Clouds**
> If your point cloud appears perfectly aligned in X/Z but is pitched or rolled by ~90 degrees, it means the calibration math assumed an Identity transform instead of respecting the camera bridge's internal rotation. The current `eye_to_hand_calibrator.py` handles this automatically via live TF lookups (`5s` timeout). Make sure `kinect2_bridge` is running *before* calibrating so the node can query the lens offset!

**Command Line Fallback**
If the GUI is unavailable, you can run the pipeline manually:
```bash
# 1. Start the detector
ros2 run parol6_vision aruco_detector --ros-args -p marker_id:=206 -p marker_size:=0.1568
# 2. Run the calibrator
ros2 run parol6_vision eye_to_hand_calibrator --ros-args -p marker_x:=0.623 -p marker_y:=0.080 -p marker_z:=0.234
# 3. Start the enforcer
ros2 run parol6_vision camera_tf_enforcer
```
