# 🧠 YOLO + ROS 2 Humble + Docker (Ubuntu 22.04) — Team Guide

> **📌 UPDATED STRATEGY (January 2026)**  
> This document describes the initial YOLO setup approach and is kept as reference.  
> For the **current recommended workflow** using virtual environments and offline wheels,  
> see: [VISION_DEPLOYMENT_GUIDE.md](../VISION_DEPLOYMENT_GUIDE.md)
>
> **Use this doc for:**
> - Understanding YOLO basics
> - GPU setup reference  
> - Training workflow concepts
> - Troubleshooting hardware issues

---

## 📌 Purpose of This Guide
This document explains how to:

✔ Run **YOLO object detection inside Docker**  
✔ Use it on **CPU or GPU**  
✔ Integrate it with **ROS 2 Humble**  
✔ **Train YOLO on your own dataset**  
✔ Avoid common problems we’ve already faced  

Following the same steps keeps our environments consistent and reduces wasted debugging time.

---

# 🚀 1. System Requirements

### ✅ Host System
- Ubuntu **22.04**
- ROS 2 **Humble**
- Docker installed

Install Docker if needed:
```bash
sudo apt install docker.io
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

Logout and log back in.

Confirm Docker works:
```bash
docker run hello-world
```

---

# 🧩 2. Pull Base ROS 2 Docker Image

```bash
docker pull ros:humble
```

Verify:
```bash
docker images
```

---

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
