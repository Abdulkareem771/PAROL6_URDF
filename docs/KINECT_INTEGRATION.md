# Xbox Kinect v2 Integration Guide

This document details the integration of the Xbox Kinect v2 camera into the PAROL6 ROS 2 environment.

## 1. Architecture
We use a **Single Container** approach. The camera drivers and ROS 2 nodes are installed directly into the main `parol6-ultimate` image.
- **Driver**: `libfreenect2` (Open source drivers for Kinect v2)
- **ROS 2 Package**: `krepa098/kinect2_ros2` (Bridge between driver and ROS 2)

## 2. Prerequisites
Before installing or using the camera:
1.  **Container must be running**: Start your container with `./start_ignition.sh`
2.  **Camera connection**: The Xbox Kinect v2 must be **plugged into a USB 3.0 port** on your computer
3.  **USB passthrough**: The container must have access to USB devices (already configured in `start_ignition.sh` with `-v /dev:/dev --privileged`)

## 3. Installation
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
**IMPORTANT**: All commands below must be run **INSIDE** the container, not on your host machine.

1.  **Enter the container**:
    ```bash
    docker exec -it parol6_dev bash
    ```
2.  **Start the camera node**:
    ```bash
    source /opt/kinect_ws/install/setup.bash
    ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
    ```
*Note: You may need to adjust the launch file name depending on the specific package contents.*

### Verifying Data
To see the list of topics published by the camera (run **inside the container**):
```bash
ros2 topic list
```
You should see topics like `/kinect2/qhd/image_color`, `/kinect2/sd/image_depth`, etc.

### Previewing Camera Data
**Option 1: Using RViz2 (Recommended)**
1.  **Launch RViz2** (in a new terminal inside the container):
    ```bash
    docker exec -it parol6_dev bash
    source /opt/kinect_ws/install/setup.bash
    rviz2
    ```
2.  **Add Camera Display**:
    - Click "Add" button (bottom left)
    - Select "By topic" tab
    - Expand `/kinect2/qhd/image_color` and select "Image"
    - Click OK
3.  **Add PointCloud Display** (for depth data):
    - Click "Add" again
    - Expand `/kinect2/sd/points` and select "PointCloud2"
    - In the PointCloud2 settings, set "Fixed Frame" to the appropriate frame (usually `kinect2_link` or similar)

**Option 2: Using image_view (Quick preview with selecting topics, and manibulating the image arguments)**
```bash
ros2 run rqt_image_view rqt_image_view
```
This will open a window showing the RGB camera feed.

### Using with PAROL6 Project
Since the camera workspace is an "overlay", you can use it alongside your robot workspace:
```bash
# Source both workspaces
source /opt/kinect_ws/install/setup.bash
source /home/kareem/Desktop/PAROL6_URDF/install/setup.bash

# Now you can launch your robot and the camera, or nodes that use both.
```

## 4. Quick Test (Plug and Play)
We've provided a test script for easy verification:
```bash
# Inside the container
docker exec -it parol6_dev bash
/workspace/scripts/test_kinect.sh
```
This will:
- Launch the camera node
- Display available topics
- Show you how to preview the video

**To preview in a separate terminal:**
```bash
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash
ros2 run image_view image_view --ros-args --remap /image:=/kinect2/qhd/image_color
```

## 5. AI Data Collection
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

## 6. Deployment & Sharing
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
