# Vision Environment Strategy - Docker + Wheels

## 🎯 Your Requirements Analysis

**What you need:**
1. ✅ Vision libraries (YOLO, OpenCV, PyTorch) in Docker
2. ✅ Python version compatibility (ROS Humble uses Python 3.10)
3. ✅ Offline installation via wheels (shareable, reproducible)
4. ✅ Avoid future conflicts with ROS Python packages
5. ✅ Work inside Docker container (all teammates have it)

---

## 📊 Options Comparison

### Option 1: venv Inside Docker ✅ **RECOMMENDED**

**Pros:**
- ✅ Isolated from ROS Python packages
- ✅ Easy to recreate (delete + rebuild venv)
- ✅ No Docker image bloat
- ✅ Fast iteration (no Docker rebuild)
- ✅ Works with Python 3.10 (matches ROS Humble)

**Cons:**
- ⚠️ Must activate venv in each terminal session
- ⚠️ Need to set PYTHONPATH in launch files

**Verdict**: ⭐⭐⭐⭐⭐ Best for development

---

### Option 2: System-wide pip in Docker ❌ **NOT RECOMMENDED**

**Pros:**
- Simple (just `pip install`)

**Cons:**
- ❌ Conflicts with ROS packages (catkin_pkg, etc.)
- ❌ Hard to rollback
- ❌ Pollutes Docker image
- ❌ Teammates get different versions

**Verdict**: ⭐⭐ Avoid

---

### Option 3: UV (Modern Tool) 🆕 **INTERESTING**

**What is uv?**
- Ultra-fast Python package installer (Rust-based)
- 10-100x faster than pip
- Better dependency resolution
- Built-in virtual environment support

**Pros:**
- ✅ Very fast
- ✅ Better caching
- ✅ Modern tooling

**Cons:**
- ⚠️ Newer tool (less tested in production)
- ⚠️ Learning curve for team
- ⚠️ Might have compatibility issues

**Verdict**: ⭐⭐⭐⭐ Good for future, but stick with venv for stability now

---

## ✅ Recommended Solution: venv + Wheels

### Architecture

```
Docker Container (parol6-ultimate:latest)
├── /opt/ros/humble/           # ROS 2 (Python 3.10)
├── /workspace/                # Your code
│   ├── venv_vision/          # Virtual environment
│   │   ├── bin/
│   │   ├── lib/python3.10/
│   │   └── pyvenv.cfg
│   ├── wheels/               # Downloaded .whl files
│   │   ├── torch-*.whl
│   │   ├── ultralytics-*.whl
│   │   └── opencv_python-*.whl
│   └── setup_vision_env.sh   # Automated setup
```

**Why This Works:**
1. ROS uses system Python 3.10 → venv uses same Python 3.10 ✅
2. Wheels are version-specific → guaranteed compatibility ✅
3. Offline install → share wheels/ folder → teammates don't re-download ✅
4. Isolated → no ROS conflicts ✅

---

## 🛠️ Implementation: Wheels-Based Setup

### Step 1: Download Wheels (One-Time, On Good Internet)

```bash
#!/bin/bash
# download_vision_wheels.sh

# Navigate to workspace
cd /workspace

# Create wheels directory
mkdir -p wheels

# Activate temporary venv to download
python3 -m venv temp_venv
source temp_venv/bin/activate

# Download wheels (platform-specific: linux_x86_64, Python 3.10)
pip download \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --dest wheels/ \
    ultralytics \
    opencv-python \
    scipy \
    torch \
    torchvision

# Clean up
deactivate
rm -rf temp_venv

echo "✓ Wheels downloaded to wheels/"
echo "  You can now share this folder or install offline"
```

---

### Step 2: Install from Wheels (Offline-Ready)

```bash
#!/bin/bash
# setup_vision_env.sh (REVISED for wheels)

set -e

VENV_DIR="venv_vision"
WHEELS_DIR="wheels"

echo "=========================================="
echo "  Vision Environment Setup (Wheels)"
echo "=========================================="

# Check if wheels exist
if [ ! -d "$WHEELS_DIR" ]; then
    echo "❌ Wheels directory not found!"
    echo "   Run: ./download_vision_wheels.sh first"
    exit 1
fi

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
echo "✓ Virtual environment created"

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install from wheels (OFFLINE mode)
echo "Installing vision libraries from wheels..."
pip install --no-index --find-links="$WHEELS_DIR" \
    ultralytics \
    opencv-python \
    scipy \
    torch \
    torchvision

echo "✓ Vision libraries installed"

# Save requirements
pip freeze > requirements_vision.txt

# Deactivate
deactivate

echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate:"
echo "  source venv_vision/bin/activate"
echo ""
echo "To share with teammates:"
echo "  1. Share wheels/ folder"
echo "  2. They run: ./setup_vision_env.sh"
echo ""
```

---

### Step 3: ROS Node Integration

```python
#!/usr/bin/env python3
# parol6_vision/yolo_detector.py

import sys
import os

# CRITICAL: Add venv to Python path BEFORE other imports
venv_path = '/workspace/venv_vision/lib/python3.10/site-packages'
if os.path.exists(venv_path):
    # Insert at beginning to prioritize venv packages
    sys.path.insert(0, venv_path)

# Now import vision libraries (from venv)
from ultralytics import YOLO
import cv2
import torch

# Import ROS (from system Python - no conflict!)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Load YOLO model (from venv)
        self.model = YOLO('yolov8n.pt')
        
        # ROS subscriber (uses system rclpy)
        self.image_sub = self.create_subscription(
            Image, '/kinect2/sd/image_color_rect',
            self.image_callback, 10
        )
        
        self.get_logger().info("YOLO Detector initialized")
    
    def image_callback(self, msg):
        # Process with YOLO
        # ...
        pass

def main():
    rclpy.init()
    node = YOLODetector()
    rclpy.spin(node)
```

**Key Point**: Python path manipulation works because:
- System Python 3.10 runs the script
- We add venv site-packages to path
- Both use same Python version → binary compatibility ✅

---

### Step 4: Launch File Integration

```python
# vision_pipeline.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='parol6_vision',
            executable='yolo_detector',
            output='screen',
            # Set PYTHONPATH to include venv
            additional_env={'PYTHONPATH': '/workspace/venv_vision/lib/python3.10/site-packages'},
        ),
    ])
```

---

## 📦 Sharing with Teammates

### Method 1: Share Wheels Folder
```bash
# On your machine (with internet)
./download_vision_wheels.sh  # Creates wheels/

# Compress
tar -czf vision_wheels.tar.gz wheels/

# Share vision_wheels.tar.gz (Google Drive, USB, etc.)
```

**Teammate setup:**
```bash
# Extract wheels
tar -xzf vision_wheels.tar.gz

# Install (offline!)
./setup_vision_env.sh
```

**Size**: ~1.5-2GB (PyTorch is large)

---

### Method 2: Commit to Git LFS (If Using)
```bash
# If repo has Git LFS
git lfs track "wheels/*.whl"
git add wheels/ .gitattributes
git commit -m "Add vision library wheels"
git push
```

**Teammates:**
```bash
git pull
./setup_vision_env.sh  # Installs from committed wheels
```

---

## 🐍 Python Version Compatibility

**ROS 2 Humble → Python 3.10** ✅

**Vision Libraries Compatibility:**
| Library | Python 3.10 Support |
|---------|---------------------|
| PyTorch | ✅ Yes (1.12+) |
| Ultralytics (YOLO) | ✅ Yes (8.0+) |
| OpenCV | ✅ Yes (4.5+) |
| scipy | ✅ Yes (1.7+) |

**Verdict**: ✅ All good! Python 3.10 is well-supported.

---

## ⚠️ Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Activate venv
**Problem**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
# Always activate before running
source venv_vision/bin/activate
ros2 run parol6_vision yolo_detector
```

Or use launch file with `additional_env` (shown above).

---

### Pitfall 2: Mixing System and venv Packages
**Problem**: Import errors, version conflicts

**Solution**: Always use `sys.path.insert(0, venv_path)` at top of scripts

---

### Pitfall 3: Wheels for Wrong Platform
**Problem**: Binary incompatibility

**Solution**: Download wheels matching your system:
```bash
# Check platform
python3 -c "import platform; print(platform.machine())"
# Output: x86_64

# Download for that platform
pip download --platform manylinux2014_x86_64 ...
```

---

## 🔄 Alternative: UV (For Future Reference)

If you want to try `uv` later:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with uv (10x faster)
uv venv venv_vision

# Install packages (Rust speed!)
uv pip install ultralytics opencv-python scipy torch

# Works exactly like venv
source venv_vision/bin/activate
```

**Recommendation**: Stick with venv+wheels for now, consider `uv` for future projects.

---

## ✅ Updated Documentation (For Colleagues)

### In Docker Container Workflow

```bash
# 1. Enter Docker container
docker exec -it parol6_dev bash

# 2. Navigate to workspace
cd /workspace

# 3. Activate vision environment
source venv_vision/bin/activate

# 4. Verify installation
python -c "from ultralytics import YOLO; print('YOLO ready!')"

# 5. Run vision nodes
ros2 run parol6_vision yolo_detector

# 6. When done
deactivate
```

**All teammates work inside Docker - consistent environment!** ✅

---

## 📝 Action Plan

1. **Merge remote xbox_camera** → See colleague's progress
2. **Download wheels** → Run `download_vision_wheels.sh`
3. **Setup venv** → Run `setup_vision_env.sh`
4. **Test** → Import YOLO, verify it works
5. **Commit wheels** → Share with team
6. **Update docs** → Add to PARALLEL_WORK_GUIDE.md

**Ready to check remote xbox_camera branch?**
