# Xbox Kinect v2 Integration Guide

This document details the integration of the Xbox Kinect v2 camera into the PAROL6 ROS 2 environment.

## 1. Architecture
We use a **Single Container** approach. The camera drivers and ROS 2 nodes are installed directly into the main `parol6-ultimate` image.
- **Driver**: `libfreenect2` (Open source drivers for Kinect v2)
- **ROS 2 Package**: `krepa098/kinect2_ros2` (Bridge between driver and ROS 2)

## 2. Installation
We have provided a script to install the drivers and ROS 2 package inside the container.

1.  **Start the Container**:
    ```bash
    ./start_ignition.sh
    ```
2.  **Run the Install Script**:
    Open a new terminal and run:
    ```bash
    docker exec -u 0 -it parol6_dev /workspace/scripts/install_kinect.sh
    ```
3.  **Save the Image (Optional but Recommended)**:
    To make the installation permanent so you don't have to run the script again:
    ```bash
    docker commit parol6_dev parol6-ultimate:latest
    ```

## 3. How to Use
### Running the Camera
To start the camera node, open a new terminal in the container and run:
```bash
source /opt/kinect_ws/install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge_launch.py
```
*Note: You may need to adjust the launch file name depending on the specific package contents.*

### Verifying Data
To see the list of topics published by the camera:
```bash
ros2 topic list
```
You should see topics like `/kinect2/qhd/image_color`, `/kinect2/sd/image_depth`, etc.

### Using with PAROL6 Project
Since the camera workspace is an "overlay", you can use it alongside your robot workspace:
```bash
# Source both workspaces
source /opt/kinect_ws/install/setup.bash
source /home/kareem/Desktop/PAROL6_URDF/install/setup.bash

# Now you can launch your robot and the camera, or nodes that use both.
```

## 4. AI Data Collection
To collect data for training AI models (e.g., object detection, grasping):

1.  **Record Data (Rosbag)**:
    Use `ros2 bag` to record raw data streams.
    ```bash
    ros2 bag record -o my_training_data /kinect2/qhd/image_color /kinect2/sd/image_depth
    ```
2.  **Exporting**:
    The bag files are saved in the container. You can copy them to your host machine or mount a volume to save them directly to your host.
3.  **Processing**:
    You can write a Python script using `cv_bridge` to extract images from the bag files or subscribe directly to the topics to save frames as `.jpg` or `.png` files for your dataset.

## 5. Deployment & Sharing
### Sharing with Colleagues
To share this setup **without** requiring them to download/build everything again:
1.  **Build the Image**: You build the image on your machine.
    ```bash
    docker build -t parol6-ultimate:latest .
    ```
2.  **Save to File**: Export the built image to a single file.
    ```bash
    docker save parol6-ultimate:latest | gzip > parol6-ultimate-with-kinect.tar.gz
    ```
3.  **Share**: Send the `.tar.gz` file to your colleagues (USB, Drive, etc.).
4.  **Load**: They load it directly into their Docker.
    ```bash
    docker load < parol6-ultimate-with-kinect.tar.gz
    ```
**Result**: They get the exact same environment, drivers, and compiled code instantly, with zero downloads.
