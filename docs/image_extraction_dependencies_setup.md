# AI Dataset Preparation Guide (RGB + Depth Extraction)

This documentation provides **clear, step-by-step instructions** for preparing your environment, recording Kinect v2 data, extracting RGB + Depth images, and troubleshooting common issues.

It is designed so that every team member can work smoothly, avoid repeated problems, and generate datasets in a consistent, professional way. (RGB + Depth) for ROS 2 Humble

This document provides **clear, detailed, step‑by‑step instructions** for installing all required libraries and tools needed for **RGB + Depth data extraction** from Kinect v2 rosbag files under **ROS 2 Humble + Python 3.10**.

These instructions ensure that all developers on the team work in harmony and avoid repeated troubleshooting. Follow this exactly when preparing your workspace or Docker image.

---
# 📸 Step 1 — Recording Rosbag for Dataset Creation
Recording a rosbag is the first step before extracting RGB and Depth images. A rosbag file stores raw camera streams so they can be processed later.

### ✔️ 1. Ensure Kinect v2 is publishing topics
Inside the container:
```bash
ros2 topic list
```
You should see:
```
/kinect2/qhd/image_color
/kinect2/sd/image_depth
```
If not, launch the Kinect driver.

### ✔️ 2. Choose where to save the rosbag
We recommend saving directly to a host-mounted folder:

On host:
```bash
mkdir -p ~/parol6_rosbags
```

Start container with a mount (already configured in most workflows):
```bash
-v ~/parol6_rosbags:/workspace/rosbags
```
Note: if it's not already configured, you must add the previous command (i.e., -v ~/parol6_rosbags:/workspace/rosbags) into your container's start commands.

### ✔️ 3. Record RGB + Depth topics
Inside the container, and after finished from starting the camera: 
```bash
ros2 bag record -o /workspace/vision_work/rosbags/ai_dataset_$(date +%Y-%m-%d_%I-%M-%S_%p) \
  /kinect2/hd/image_color \
  /kinect2/hd/image_depth

```

Press **Ctrl + C** to stop recording.

### ✔️ Output structure
```
ai_dataset/
  metadata.yaml
  ai_dataset_0.db3
```

This `.db3` file is used in Step 2.

---
# 🖼 Step 2 — Extraction Workflow (RGB + Depth)
Once the rosbag is recorded, you must extract image files that will be used for AI training.

Extraction has three stages:
1. **Reading rosbag messages using rosbag2_py**
2. **Converting ROS Image → OpenCV image using CvBridge**
3. **Saving RGB and Depth frames with synchronized numbering**

Below is the recommended extraction script.

### ✔️ 1. Create folders for extracted images
Inside container:
```bash
mkdir -p /workspace/extracted_dataset/rgb
mkdir -p /workspace/extracted_dataset/depth
```

### ✔️ 2. Use the offline extraction script:
Create: `extract_from_bag.py`

```python
import os
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
# Path to the bag folder (the folder that contains metadata.yaml + .db3)
BAG_FOLDER = "/workspace/rosbags/ai_dataset"   # adjust to your bag location
OUT_RGB = "/workspace/extracted_dataset/rgb"
OUT_DEPTH = "/workspace/extracted_dataset/depth"

os.makedirs(OUT_RGB, exist_ok=True)
os.makedirs(OUT_DEPTH, exist_ok=True)

bridge = CvBridge()

storage_options = StorageOptions(uri=BAG_FOLDER, storage_id="sqlite3")
converter_options = ConverterOptions("", "")

reader = SequentialReader()
reader.open(storage_options, converter_options)

# prepare topic name variables
rgb_topic = "/kinect2/qhd/image_color"
depth_topic = "/kinect2/sd/image_depth"

count = 0
while reader.has_next():
    (topic, data, t) = reader.read_next()
    # t is a builtin_interfaces/Time in serialized form; rosbag returns nanoseconds int typically
    try:
        msg = deserialize_message(data, Image)
    except Exception as e:
        print("Failed to deserialize message:", e)
        continue

    # Use the header stamp for filenames when available
    stamp = None
    try:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    except Exception:
        # fallback to counter
        stamp = count

    if topic == rgb_topic:
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(f"{OUT_RGB}/{count}.png", cv_img)
    elif topic == depth_topic:
        # Depth often encoded as 16UC1 or 32FC1; use passthrough and save as PNG (scaled if needed)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Optionally normalize for visualization:
        # cv2.normalize(cv_img, cv_img, 0, 65535, cv2.NORM_MINMAX)
        cv2.imwrite(f"{OUT_DEPTH}/{count}.png", cv_img)

    count += 1

print("Done. Extracted", count, "messages.")
```

Run:
```bash
python3 extract_from_bag.py
```

### ✔️ Output structure
```
extracted_dataset/
  rgb/
    0.png
    1.png
    2.png
  depth/
    0.png
    1.png
    2.png
```

The filenames match counter values → ensuring RGB & Depth pairs stay synchronized.



---
# ⚙️ Step 3 — Environmental Setup (Dependencies)
To make Steps 1 and 2 work **consistently across all machines**, install the required tools inside the Docker image.

## 🧩 Overview
For AI dataset creation (RGB + Depth images) you must install:

| Component | Purpose |
|----------|---------|
| **cv_bridge** | Convert ROS images → OpenCV images |
| **vision_opencv** | Provides matching OpenCV libraries needed by CvBridge |
| **rosbag2_python** | Python reader for `.db3` rosbag files |
| **OpenCV (apt + pip)** | Save & process RGB/Depth images |
| **NumPy** | Required by OpenCV + CvBridge |

These tools allow scripts to:
- Extract RGB frames from Kinect rosbag
- Extract depth maps
- Synchronize RGB + depth
- Save images as `.png` for AI training


---
## ✔️ Section 1 — Installation Inside Dockerfile (Recommended)
This is the **proper, permanent, reproducible** installation method.
Add the following block to your Dockerfile:

```dockerfile
# Install dependencies for images extraction (RGB + Depth extraction tools)
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-default-plugins \
    python3-opencv \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*
```

### 📌 Why this is the recommended block?
- Ensures **CvBridge loads correctly** with Python 3.10
- Ensures **rosbag2_py** works to read `.db3` bag files
- Ensures **OpenCV is modern enough** (pip version preferred)
- Ensures **depth extraction** works without errors
- Prevents common dependency conflicts

---
## ✔️ Section 2 — Rebuild the Docker Image
After modifying the Dockerfile:

```bash
docker build -t parol6-ultimate:latest .
```

Make sure the build finishes without errors.

---
## ✔️ Section 3 — Verifying Installation
Inside the container, run the following tests.

### Test OpenCV
```bash
python3 - <<EOF
import cv2
print("OpenCV loaded:", cv2.__version__)
EOF
```

Expected: **a version number appears** (4.x.x).

### Test CvBridge
```bash
python3 - <<EOF
from cv_bridge import CvBridge
print("CvBridge OK")
EOF
```

Expected: `CvBridge OK` with no errors.

### Test rosbag2_py
```bash
python3 - <<EOF
import rosbag2_py
print("rosbag2_py OK")
EOF
```

Expected: `rosbag2_py OK`.

If all 3 tests pass → your environment is ready for RGB + Depth extraction.

---
## ✔️ Section 4 — Installing Inside a Running Container (Alternative)
If someone cannot edit the Dockerfile, they may install inside the container:

```bash
apt-get update
apt-get install -y \
  ros-humble-cv-bridge \
  ros-humble-vision-opencv \
  ros-humble-rosbag2-python \
  python3-opencv \
  python3-numpy
pip3 install --no-cache-dir opencv-python
```

To **save this state permanently**, they must run:

```bash
docker commit parol6_dev parol6-ultimate:with-extraction-libs
```

This creates a new image containing the installed libraries.

---
## 🛑 Troubleshooting Guide
This section covers the most common issues the team may face.

---
## ❌ Issue 1 — `ImportError: cannot import name CvBridge`
### ✔️ Fix
Install missing ROS OpenCV bridges:

```bash
apt-get install -y ros-humble-cv-bridge ros-humble-vision-opencv
```

This resolves 99% of CvBridge import issues.

---
## ❌ Issue 2 — `ImportError: libopencv_imgproc.so not found`
**Cause:** The system OpenCV version doesn’t match what CvBridge expects.

### ✔️ Fix
Install vision_opencv:
```bash
apt-get install -y ros-humble-vision-opencv
```
And install pip OpenCV:
```bash
pip3 install opencv-python
```

---
## ❌ Issue 3 — `ModuleNotFoundError: No module named rosbag2_py`
### ✔️ Fix
Install rosbag2 python bindings:
```bash
apt-get install -y ros-humble-rosbag2-python
```

---
## ❌ Issue 4 — Depth images save incorrectly (black or blank images)
### ✔️ Fix
Use:
```python
cv_img = bridge.imgmsg_to_cv2(msg, "passthrough")
```
Do NOT use `bgr8` for depth.

Also ensure depth is saved in PNG format (preserves 16‑bit).

---
## ❌ Issue 5 — `cv2.error: function not implemented`
### ✔️ Fix
Install the pip version of OpenCV:
```bash
pip3 install --no-cache-dir opencv-python
```
This resolves miscompiled or missing cv2 modules.

---
## ❌ Issue 6 — Extraction script crashes when reading rosbag
### Possible causes:
- rosbag path is incorrect
- depth topic name mismatches
- rosbag recorded using wrong ROS2 version

### ✔️ Verify topics with:
```bash
ros2 bag info your_bag_folder
```

Ensure topics include:
```
/kinect2/qhd/image_color
/kinect2/sd/image_depth
```

---
## ❌ Issue 7 — Timestamp filenames do not match between RGB and Depth
### ✔️ Fix
Use the message header timestamp:
```python
stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
```
This guarantees synchronized filenames.

---
## ✔️ Section 5 — Summary for Team Members
If you follow this documentation:
- Your Docker environment will match the team standard.
- All extraction scripts will run without errors.
- You will produce consistent RGB + Depth datasets.
- No wasted time troubleshooting missing dependencies.

We strongly recommend installing the dependencies **inside the Dockerfile** for long‑term stability.

If you face any issues not covered here, report them so we can update this documentation.
