# 🧠 YOLO + ROS 2 Humble + Docker (Ubuntu 22.04) — Team Guide

## 📌 Purpose of This Guide
This document explains **how to install and use the YOLOv11 object detection model inside an existing Docker container running ROS 2 Humble, without modifying the Dockerfile.**

✔ Run **YOLO object detection inside Docker**  
✔ Use it on **CPU or GPU**  
✔ Integrate it with **ROS 2 Humble**  
✔ **Train YOLO on your own dataset**  
✔ Avoid common problems we’ve already faced 

Instead of changing the image, we:

- Use a **Python virtual environment (venv)** inside the container.
- Install YOLOv11 and its dependencies locally.
- Keep the Docker image **clean, reusable, and consistent.**
- Avoid dependency conflicts with system Python and ROS 2.

This guide is written so that:
    
- New team members can follow it end-to-end.
- Everyone installs things the same way.
- We minimize debugging time and environment drift.


Following the same steps keeps our environments consistent and reduces wasted debugging time.

---

# 🚀 1. System Requirements

This guide assumes:

- **Host OS:** Ubuntu 22.04
- **ROS Version:** ROS 2 Humble
- **Docker:** Already installed and working
- **Docker Image:** Already contains ROS 2 Humble
- **No Dockerfile changes allowed**
- **Internet access inside container** (for pip installs)

---
# 2. Why Use a Python Virtual Environment Inside Docker?

Even though Docker is already an isolated environment, **ROS 2 Humble relies on system Python**, and installing ML libraries globally can cause:

- Conflicts with ROS Python packages
- Broken `rclpy` or OpenCV bindings
- Hard-to-reproduce bugs across machines

Using `venv`:
- Keeps YOLO dependencies isolated
- Makes rollback easy
- Avoids breaking ROS
- Works without touching Dockerfile

---

# 3. Start and Enter the Docker Container

Start your container as usual:

```bash
docker start parol6_dev
docker exec -it parol6_dev bash
```
Verify ROS is available:

```bash 
source /opt/ros/humble/setup.bash
ros2 --help
```

---

# 4. Install Required System Packages (Inside Container)

These packages are safe to install and do NOT affect the Dockerfile.

```bash
apt update && apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0
```

Why these are needed:

- `python3-venv`: create virtual environments

- `python3-dev`: compile Python wheels

- `libgl1, libglib2.0-0`: OpenCV GUI & image support


---

# 5. Create a Python Virtual Environment

Choose a location **outside ROS workspace**:

```bash
mkdir -p /opt/venvs
python3 -m venv /opt/venvs/yolo
```

Activate the environment:

```bash
source /opt/venvs/yolo/bin/activate
```
Confirm:

```bash
which python
# Should point to /opt/venvs/yolo/bin/python
```
Upgrade pip tools:

```bash
pip install --upgrade pip setuptools wheel
```


---

# 6. Install YOLOv11 and Dependencies
## 6.1 Install YOLO (Ultralytics)
```bash
pip install ultralytics
```

Verify installation:

```bash
yolo --help
```

## 6.2 Install PyTorch (CPU or GPU)

**CPU-only (recommended unless CUDA is configured):**
```bash
pip install torch torchvision torchaudio
```

**GPU (CUDA 11.8 example):**

```bash 
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

Verify PyTorch:

```bash
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```


# 7. Download and Test YOLOv11
## 7.1 Quick Sanity Test
```bash
yolo predict model=yolo11n.pt source=https://ultralytics.com/images/bus.jpg
```
Expected:

- Image downloaded

- Detection output printed

- Annotated image saved

--- 

# 8. Using YOLOv11 in Python
## 8.1 Minimal Python Test
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")

for r in results:
    print(r.boxes.cls)
```

---

# 9. Using YOLOv11 with ROS 2
## 9.1 Important Rule

**Always source ROS first, then activate the virtual environment**

Correct order:
```bash
source /opt/ros/humble/setup.bash
source /opt/venvs/yolo/bin/activate
```
## 9.2 ROS 2 Python Node Example
```python 
import rclpy
from rclpy.node import Node
from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.model = YOLO("yolo11n.pt")
        self.get_logger().info("YOLOv11 loaded")

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

Run it with:

```bash
python3 yolo_node.py
```

---

# 10. Make Activation Easier (Recommended)
Add this alias:
```bash
echo "alias yoloenv='source /opt/ros/humble/setup.bash && source /opt/venvs/yolo/bin/activate'" >> ~/.bashrc
source ~/.bashrc
```
Now simply run:

```bash
yoloenv
```

---

# 11. Troubleshooting (Common Issues We Faced)
### ❌ `ModuleNotFoundError: ultralytics`

Cause:

-  Virtual environment not activated

Fix:

```bash

source /opt/venvs/yolo/bin/activate
```
---

### ❌ `ImportError: libGL.so.1 not found`
Fix:

```bash 
apt install -y libgl1
```

---

### ❌ `cv2.imshow()` crashes

Cause:

- No display in Docker

Fix:

- Use `cv2.imwrite()`

- Or run container with X11 forwarding

- Or use headless inference only




### ❌ `ROS node crashes after installing PyTorch`
Cause:

- Installed packages globally instead of venv

Fix:

```bash
pip uninstall torch torchvision
# Reinstall inside venv
```

---

### ❌ `CUDA not detected`

Check:

```bash
nvidia-smi
```

Ensure:

- NVIDIA Container Toolkit installed on host

- Docker run command includes `--gpus all`

---

### ❌ `rclpy` not found inside venv

Cause:

- ROS not sourced before venv

Fix:

```bash
source /opt/ros/humble/setup.bash
source /opt/venvs/yolo/bin/activate
```

---

# 12. Best Practices for Team Consistency

- ❗ Never install ML libraries with `apt`

- ❗ Never `pip install` without activating venv

- ✔ Always use `/opt/venvs/yolo`

- ✔ Keep ROS Python clean

- ✔ Document any new dependency

---

## 13. Summary

This workflow:

- Requires no Dockerfile changes

- Is safe for ROS 2 Humble

- Is repeatable across machines

- Keeps ML and ROS dependencies isolated

- Minimizes environment-related bugs

If everyone follows this document, we avoid:

- Broken ROS installs

- Dependency mismatches

- “Works on my machine” issues

 
---
````
    


    

````

# 🟦 3. Run YOLO Inside Docker — CPU-Only (Simplest Setup)

This works on any machine — no GPU required.

Start a ROS Humble container:
```bash
docker run -it --name yolo_cpu --net=host --ipc=host ros:humble bash
```

You are now **inside the container shell**.

---

## 🔧 3.1 Install Dependencies (Inside the Container)

```bash
apt update
apt install -y python3 python3-pip python3-opencv git
pip3 install --upgrade pip
```

---

## 🤖 3.2 Install YOLO (Ultralytics — YOLOv11)

```bash
pip3 install ultralytics
```

Verify:
```bash
yolo
```

You should see the YOLO CLI help menu.

---

## 🧪 3.3 Test YOLO Detection

```bash
yolo predict model=yolov11n.pt source='https://ultralytics.com/images/bus.jpg'
```

Results appear in:

```
runs/predict/
```

🎉 **YOLO now runs on CPU inside Docker**

---

# ⚡ 4. Run YOLO Inside Docker — With GPU (NVIDIA)

Recommended for **real-time performance**.

---

## 🔹 4.1 Install NVIDIA Docker Runtime (Host Machine Only)

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker
```

Verify GPU access inside Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU details.

---

## 🔹 4.2 Start ROS Container WITH GPU

```bash
docker run -it --gpus all --name yolo_gpu --net=host --ipc=host ros:humble bash
```

---

## 🔹 4.3 Install Dependencies (Inside Container)

```bash
apt update
apt install -y python3 python3-pip python3-opencv git
pip3 install --upgrade pip
```

---

## 🔹 4.4 Install PyTorch With CUDA Support

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Expected output:
```
True
```

---

## 🔹 4.5 Install YOLO

```bash
pip3 install ultralytics
```

---

## 🔹 4.6 Run YOLO (GPU Accelerated)

```bash
yolo predict model=yolov11n.pt source=0
```

GPU will be used automatically 🎉

---

# 🤝 5. Using YOLO With ROS 2

Install ROS–OpenCV bridge inside the container:

```bash
apt install -y ros-humble-cv-bridge ros-humble-image-transport
pip3 install numpy
```

---

## Example ROS2 YOLO Node (`yolo_node.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.model = YOLO('yolov11n.pt')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame)
        annotated = results[0].plot()
        cv2.imshow("YOLOv11", annotated)
        cv2.waitKey(1)

rclpy.init()
rclpy.spin(YoloNode())
```

You may later:
✔ change the topic  
✔ publish results  
✔ add filtering  

---

# 🎯 6. Training YOLO on Our Dataset

This works on CPU or GPU.

---

## 📁 6.1 Dataset Folder Structure

```
dataset/
 ├── images/train
 ├── images/val
 ├── labels/train
 └── labels/val
```

Labels use format:
```
class x_center y_center width height
```
(all values normalized 0–1)

---

## 📝 6.2 Create `data.yaml`

```yaml
train: dataset/images/train
val: dataset/images/val

nc: 3
names: ['class1', 'class2', 'class3']
```

---

## 🏋️ 6.3 Start Training

```bash
yolo train model=yolov11n.pt data=data.yaml epochs=100 imgsz=640
```

Training output:
```
runs/train/
```

---

# 📦 7. Save Container State

So we don’t reinstall every time:

```bash
exit
docker commit yolo_gpu yolo_ready
```

Next use:

```bash
docker run -it --gpus all --net=host --ipc=host yolo_ready bash
```

---

# 🛑 8. Troubleshooting & Common Issues

## 🟡 GPU Not Detected

Check host:
```bash
nvidia-smi
```

Check container:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If `False`:
- run container with `--gpus all`
- reinstall CUDA-enabled PyTorch
- restart Docker
- confirm NVIDIA drivers installed

---

## 🔴 Camera Not Working

Give permission:
```bash
sudo chmod 666 /dev/video0
```

Ensure you ran container with:
```
--net=host
--ipc=host
```

---

## 🔴 ROS Image Conversion Errors

Use:
```
bgr8
```

in cv_bridge.

---

## 🔴 Poor Performance

Try:
```
yolov11n.pt  (fastest)
```

or reduce image size:
```bash
imgsz=480
```

---

## 🔴 Python / pip Errors

```bash
pip3 install --upgrade pip
```

---

## 🔴 Changes Lost After Restart

Commit the container:
```bash
docker commit <container> <new_image_name>
```

---

## 🔴 Docker Permission Denied

Run:
```bash
sudo usermod -aG docker $USER
```

Logout + login.

---

# 💡 9. Why We Use YOLOv11 (Ultralytics)

✔ Easy installation  
✔ Active support  
✔ Works on CPU & GPU  
✔ Well-structured API  
✔ Good ROS integration  

---

# 🙏 10. Team Rules (Consistency Matters)

Please follow:

✔ Use YOLO **inside Docker**  
✔ Prefer **YOLOv11** unless discussed  
✔ Keep Python version **3.8–3.11**  
✔ Save working containers using `docker commit`  
✔ Share consistent dataset structure  

---

# 📩 11. Reporting Issues

When something fails, please share:

• command you ran  
• error text  
• CPU or GPU  
• container name  

This helps us support each other faster.

---

# 🎉 Done!

You now know how to:

✔ Run YOLO in Docker  
✔ Use CPU or GPU  
✔ Train your own data  
✔ Debug common problems  

---
