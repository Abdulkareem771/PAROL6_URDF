# Vision System Deployment Guide - PAROL6 Team

**Final Robust Installation for Production Use**

This is the **authoritative guide** for setting up the vision environment.  
All team members should follow these steps for consistency.

---

## 📋 Overview

**What We're Building:**
- Vision-guided welding system
- YOLO object detection in Docker
- ROS 2 Humble integration
- Offline-capable (wheels-based)
- Python 3.10 (ROS Humble compatible)

**Key Decisions:**
- ✅ Virtual environment INSIDE Docker (isolation)
- ✅ Wheels for offline installation (reproducibility)
- ✅ Python 3.10 (ROS Humble native version)
- ✅ Single container (`parol6-ultimate`)

---

## 🖥️ Part 0: Host Machine Setup (Prerequisites)

**Before starting Docker, ensure your host machine is ready.**

### 0.1 Install NVIDIA Container Toolkit (For GPU)
*Reference from teammate's notes*

If you have an NVIDIA GPU, you need the toolkit to let Docker use it.

```bash
# On Host Machine (NOT inside Docker)
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker
```

**Verify Host GPU:**
```bash
nvidia-smi
```

---

## 🎯 Quick Reference

| Component | Version | Location |
|-----------|---------|----------|
| Docker Image | `parol6-ultimate:latest` | Shared image |
| Python | 3.10 | Inside Docker |
| ROS 2 | Humble | `/opt/ros/humble/` |
| Vision venv | 3.10 | `/workspace/venv_vision/` |
| Wheels | - | `/workspace/wheels/` |
| YOLO | YOLOv8/v11 | Via ultralytics |
| PyTorch | 2.0+ (CPU or CUDA) | Auto-detected |

---

## 📦 Dependencies & Setup

### Critical Requirement: Python 3.10
Since we are using **ROS 2 Humble**, we **MUST use Python 3.10**.
- The Docker container uses Python 3.10 by default.
- Any external wheels must be built for `cp310` (Python 3.10).
- Windows wheels (cp313, win_amd64) **WILL NOT WORK** on the robot.

---

### Option A: Online Setup (Direct Download)
*Best for fast setup if the robot has internet.*

Run the setup script inside the container. It will automatically download the correct versions from PyPI.

```bash
# Inside Docker container
./setup_vision_env.sh
```

---

### Option B: Offline Setup (Using Linux Wheels)
*Best for air-gapped robots or slow internet.*

1. **On a PC with Internet:**
   Run the download script to fetch Linux-compatible wheels:
   ```bash
   ./download_vision_wheels.sh
   # Result: Creates 'wheels_linux_py310' folder (~2GB)
   ```

2. **Transfer**:
   Copy the `wheels_linux_py310` folder to your robot's workspace.

3. **Install on Robot:**
   The setup script will automatically detect the local folder and install from it.
   ```bash
   ./setup_vision_env.sh
   ```

---

## 🔧 Setup Script (`setup_vision_env.sh`)

This script handles everything automatically:
1. Creates virtual environment (`venv_vision`)
2. Activates it
3. Checks for local wheels folder
   - If found: Installs offline
   - If missing: Downloads online
4. Installs **Necessary Libraries Only**:
   - `ultralytics` (YOLO)
   - `opencv-python`
   - `torch` & `torchvision`
   - `scipy`

---

## 🤖 Part 3: ROS Integration

### Step 3.1: Create Vision Package

```bash
cd /workspace
source /opt/ros/humble/setup.bash

# Create package
ros2 pkg create parol6_vision \
  --build-type ament_python \
  --dependencies rclpy sensor_msgs vision_msgs geometry_msgs
```

---

### Step 3.2: Add Vision Node

Create `parol6_vision/parol6_vision/yolo_detector.py`:

```python
#!/usr/bin/env python3
import sys
import os

# CRITICAL: Add venv to path FIRST
venv_site = '/workspace/venv_vision/lib/python3.10/site-packages'
if os.path.exists(venv_site):
    sys.path.insert(0, venv_site)

# Now import vision libs (from venv)
from ultralytics import YOLO
import cv2
import torch

# Import ROS (from system)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence', 0.7)
        
        model_path = self.get_parameter('model_path').value
        self.confidence = self.get_parameter('confidence').value
        
        # Load YOLO
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/kinect2/sd/image_color_rect',
            self.image_callback, 10
        )
        
        # Publishers
        self.det_pub = self.create_publisher(
            Detection2DArray, '/yolo/detections', 10
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'YOLO Detector started on {device}')
        self.get_logger().info(f'Model: {model_path}')
    
    def image_callback(self, msg):
        # Convert ROS → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run YOLO
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        # Convert to Detection2DArray
        detections = Detection2DArray()
        detections.header = msg.header
        
        for r in results[0].boxes:
            det = Detection2D()
            # ... (populate detection message)
            detections.detections.append(det)
        
        # Publish
        self.det_pub.publish(detections)

def main():
    rclpy.init()
    node = YOLODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

### Step 3.3: Update setup.py

```python
entry_points={
    'console_scripts': [
        'yolo_detector = parol6_vision.yolo_detector:main',
        'depth_matcher = parol6_vision.depth_matcher:main',
    ],
},
```

---

### Step 3.4: Build Package

```bash
cd /workspace
source /opt/ros/humble/setup.bash

# Build
colcon build --symlink-install --packages-select parol6_vision

# Source
source install/setup.bash
```

---

### Step 3.5: Test ROS Node

```bash
# Terminal 1: Start Kinect
ros2 launch kinect2_ros2 driver.launch.py

# Terminal 2: Run YOLO detector
source venv_vision/bin/activate  # Important!
ros2 run parol6_vision yolo_detector
```

**Verify:**
```bash
ros2 topic echo /yolo/detections
```

---

## 🚀 Part 4: Launch File Integration

Create `parol6_vision/launch/vision_pipeline.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='parol6_vision',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': '/workspace/models/custom_workpiece.pt',
                'confidence': 0.8
            }],
            # Set PYTHONPATH for venv
            additional_env={
                'PYTHONPATH': '/workspace/venv_vision/lib/python3.10/site-packages'
            }
        ),
        Node(
            package='parol6_vision',
            executable='depth_matcher',
            name='depth_matcher',
            output='screen',
            additional_env={
                'PYTHONPATH': '/workspace/venv_vision/lib/python3.10/site-packages'
            }
        ),
    ])
```

**Run:**
```bash
ros2 launch parol6_vision vision_pipeline.launch.py
```

---

## 🎓 Part 5: Training Custom YOLO Model

### Step 5.1: Prepare Dataset

**Required Folder Structure:**
```
dataset/
 ├── images/
 │   ├── train/  # Training images (.jpg/.png)
 │   └── val/    # Validation images
 └── labels/
     ├── train/  # YOLO labels (.txt)
     └── val/
```

**Commands to create:**
```bash
mkdir -p dataset/images/{train,val}
mkdir -p dataset/labels/{train,val}
```

**Label Format:**
`class_id x_center y_center width height` (Normalized 0-1)

---

### Step 5.2: Create data.yaml

```yaml
train: dataset/images/train
val: dataset/images/val

nc: 1  # Number of classes
names: ['workpiece']  # Class names
```

---

### Step 5.3: Train

```bash
source venv_vision/bin/activate

yolo train \
  model=yolov8n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  device=0  # Use GPU 0 (or 'cpu')

# Results in: runs/train/exp/weights/best.pt
```

---

### Step 5.4: Test Custom Model

```bash
# Update launch file model_path
ros2 launch parol6_vision vision_pipeline.launch.py \
  -p model_path:=/workspace/runs/train/exp/weights/best.pt
```

---

## 👥 Part 6: Team Workflow

### For New Team Members

1. **Get repository**:
   ```bash
   git clone <repo>
   cd PAROL6_URDF
   ```

2. **Get wheels** (from teammate or download):
   ```bash
   # Option A: Get tar from teammate
   tar -xzf vision_wheels.tar.gz
   
   # Option B: Download fresh
   ./download_vision_wheels.sh
   ```

3. **Start container**:
   ```bash
   docker pull parol6-ultimate:latest
   docker run -d --name parol6_dev ... # (Step 1.1)
   docker exec -it parol6_dev bash
   ```

4. **Setup vision**:
   ```bash
   cd /workspace
   ./setup_vision_env.sh
   ```

5. **Build & test**:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ros2 launch parol6_vision vision_pipeline.launch.py
   ```

**Total time**: 15-20 minutes

---

### Daily Development Workflow

```bash
# 1. Start container (if not running)
docker start parol6_dev

# 2. Enter container
docker exec -it parol6_dev bash

# 3. Navigate
cd /workspace

# 4. Activate venv (for vision work)
source venv_vision/bin/activate

# 5. Code/test
# ... work on vision nodes ...

# 6. Build
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select parol6_vision

# 7. Run
source install/setup.bash
ros2 launch parol6_vision vision_pipeline.launch.py
```

---

## 🔍 Part 7: Verification Checklist

After setup, verify:

- [ ] Python version is 3.10.x
- [ ] Venv activates without errors
- [ ] `from ultralytics import YOLO` works
- [ ] `import torch; torch.cuda.is_available()` returns True (if GPU)
- [ ] ROS 2 nodes can import vision libraries
- [ ] Launch file runs without errors
- [ ] `/yolo/detections` topic publishes data
- [ ] RViz visualizes detections

---

## ⚠️ Troubleshooting

### Issue: ModuleNotFoundError: ultralytics

**Cause**: venv not activated or PYTHONPATH not set

**Fix**:
```bash
# Option 1: Activate venv
source venv_vision/bin/activate

# Option 2: Set in launch file
additional_env={'PYTHONPATH': '/workspace/venv_vision/lib/python3.10/site-packages'}
```

---

### Issue: Camera Not Working / Not Found

**Symptom**: `[ERROR] [kinect_node]: Cannot open device`

**Cause**: Docker container lacks permissions or `--privileged` flag missing.

**Fix**:
1. **On Host Machine**: Give permission to video device
   ```bash
   sudo chmod 666 /dev/video0
   # Or specific kinect USB device rules
   ```
2. **Docker Run Command**: Ensure it has:
   ```bash
   --privileged --network host -v /dev:/dev
   ```

---

### Issue: CUDA not available

**Check host GPU**:
```bash
nvidia-smi
```

**Check Docker access**:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Fix**: Ensure container started with `--gpus all`

---

### Issue: ROS package conflicts

**Symptom**: ImportError with catkin_pkg, etc.

**Cause**: System pip install polluted ROS Python

**Fix**: Use venv (as documented)

---

### Issue: Wheels not found

**Symptom**: setup_vision_env.sh downloads from internet

**Fix**:
```bash
# Verify wheels exist
ls wheels/*.whl | wc -l  # Should be 40+

# If missing, download
./download_vision_wheels.sh
```

---

## 📚 Related Documentation

- **[PARALLEL_WORK_GUIDE.md](PARALLEL_WORK_GUIDE.md)** - Team task distribution
- **[VISION_COLLEAGUE_PLAN.md](VISION_COLLEAGUE_PLAN.md)** - Depth matcher implementation
- **[VISION_ENV_STRATEGY.md](VISION_ENV_STRATEGY.md)** - Technical deep-dive
- **[docs/YOLO_Model_Install.md](docs/YOLO_Model_Install.md)** - Legacy reference
- **[docs/KINECT_INTEGRATION.md](docs/KINECT_INTEGRATION.md)** - Camera setup

---

## ✅ Success Criteria

**Environment is ready when:**
1. ✅ YOLO detects objects in test images
2. ✅ ROS node publishes detections at 5-10Hz
3. ✅ GPU utilized (if available)
4. ✅ No import errors
5. ✅ Reproducible across all teammates

---

## 🎓 Best Practices Summary

1. **Always use venv** - Don't install vision libs system-wide
2. **Use wheels** - Faster, reproducible, offline-capable
3. **Python 3.10** - Match ROS Humble exactly
4. **Single container** - Don't create yolo_cpu, yolo_gpu, etc.
5. **PYTHONPATH in launch** - Don't forget to set in launch files
6. **Test incrementally** - venv → imports → ROS node → launch
7. **Share wheels** - Don't make everyone download 2GB

---

## 📞 Support

**Questions?** Check:
1. This guide first
2. VISION_ENV_STRATEGY.md (technical details)
3. Ask team lead (Kareem)

**Report issues with:**
- Command that failed
- Full error message
- Python version (`python3 --version`)
- Venv activated? (yes/no)

---

**Last Updated**: January 2026  
**Maintained by**: PAROL6 Vision Team
