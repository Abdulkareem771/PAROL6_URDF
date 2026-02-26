# Kinect2 ROS2 Driver User Guide

This guide provides comprehensive instructions for using the improved `kinect2_ros2` driver, including new features like **Digital IR Auto-Exposure** and **Depth Hole Filling**.

## 1. Quick Start

### Launching the Driver

Use the provided launch file to start the bridge with default settings:

```bash
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
```

### Viewing Data

Open `rqt` or `rviz2` to inspect the topics:

- **Color:** `/kinect2/hd/image_color`, `/kinect2/qhd/image_color`
- **Depth:** `/kinect2/hd/image_depth_rect`, `/kinect2/sd/image_depth`
- **IR:** `/kinect2/sd/image_ir`

### Integration: RTAB-Map

To generate 3D scans using RTAB-Map:

```bash
ros2 launch kinect2_bridge rtabmap.launch.py
```

---

## 2. Configuration & Parameters

You can configure the node using standard ROS2 launch arguments.

### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_name` | `kinect2` | Namespace for all topics. |
| `fps_limit` | `-1.0` | Limit the frame rate. -1.0 = unlimited (max 30). |
| `publish_tf` | `true` | Publish TF transforms (camera_link, optical_frames). |
| `min_depth` | `0.1` | Minimum depth range in meters. |
| `max_depth` | `4.0` | Maximum depth range in meters. |

### Advanced Features (New)

#### 1. Depth Hole Filling

Fills invalid pixels (black holes) in the depth map using neighboring values.

- **Parameter:** `hole_fill_radius` (int)
- **Default:** `0` (Disabled)
- **Recommended:** `1` to `3`
- **Usage:**

  ```bash
  ros2 launch kinect2_bridge kinect2_bridge_launch.yaml hole_fill_radius:=3
  ```

#### 2. IR Digital Auto-Exposure

Enables software-based dynamic range compression for the IR camera. **Crucial for calibration** in indoor lighting.

- **Parameter:** `ir_auto_exposure` (bool)
- **Default:** `false` (Disabled)
- **Usage:**

  ```bash
  ros2 launch kinect2_bridge kinect2_bridge_launch.yaml ir_auto_exposure:=true
  ```

#### 3. Processing Method

Choose between CPU and OpenCL (GPU) processing.

- **Parameter:** `reg_method`
- **Values:** `cpu` (default for stability), `opencl` (requires GPU drivers)
- **Usage:**

  ```bash
  ros2 launch kinect2_bridge kinect2_bridge_launch.yaml reg_method:=opencl
  ```

---

## 3. Calibration Guide

For accurate point clouds and depth alignment, you must calibrate the camera.

### Step 1: Preparation

1. **Enter the Container**:

    ```bash
    docker exec -it parol6_dev bash
    source /opt/kinect_ws/install/setup.bash
    ```

2. **Create Calibration Directory**:
    Create a folder to store your raw calibration data:

    ```bash
    mkdir ~/kinect_cal_data; cd ~/kinect_cal_data
    ```

3. **Print Pattern**: Use a chess pattern (e.g., 5x7 with 0.03m squares: `chess5x7x0.03`).
    *Note: Ensure you measure your square size accurately!*

### Step 2: Comprehensive Calibration Sequence

For the best accuracy, perform the calibration in the following order:

#### 1. Color Camera Intrinsics

Record and calibrate the RGB camera.

```bash
# Record
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 record color
# Calibrate
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate color
```

#### 2. IR Camera Intrinsics

Record and calibrate the Infrared camera.

```bash
# Record
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 record ir
# Calibrate
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate ir
```

#### 3. Extrinsic Calibration (Sync)

Calibrate the pose between Color and IR cameras.

```bash
# Record Synchronized
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 record sync
# Calibrate
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate sync
```

#### 4. Depth Calibration

Calibrate depth bias (requires computed extrinsics).

```bash
ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate depth
```

### Step 3: Installation

1. **Find Serial Number**:
    Look for the `device serial: XXXXXXXXXXXX` line in the `kinect2_bridge` launch output.
    *Example Serial: 018436651247*

2. **Create Device Directory**:

    ```bash
    ros2 cd kinect2_bridge/data
    mkdir 018436651247  # Replace with YOUR serial
    ```

3. **Copy Calibration Files**:
    Copy the generated YAML files from your recording directory to the bridge folder:

    ```bash
    cp ~/kinect_cal_data/calib_color.yaml /opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/data/018436651247/
    cp ~/kinect_cal_data/calib_depth.yaml /opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/data/018436651247/
    cp ~/kinect_cal_data/calib_ir.yaml /opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/data/018436651247/
    cp ~/kinect_cal_data/calib_pose.yaml /opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/data/018436651247/
    ```

### Step 4: Verification & Persistence

1. **Verify**: Restart the bridge.

    ```bash
    ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
    ```

    Check alignment in `rqt_image_view`.

2. **Save Changes (Important)**:
    Since you are in a Docker container, you must commit your changes to save the calibration files permanently.

    ```bash
    # On Host Machine (outside container)
    docker commit parol6_dev parol6-ultimate:latest
    docker stop parol6_dev
    ```

### Calibration Quality Engine (New)

The improved calibration tool provides real-time and post-process feedback to ensure high-quality results.

#### 1. Coverage Map (Real-time)

During recording (`record` mode), a 3x3 grid is displayed in the terminal:

```
[OK] [  ] [OK]
[  ] [OK] [  ]
[OK] [  ] [OK]
```

- **Goal:** Get an `[OK]` in **all 9 cells**.
- **Visual AR:** Look at the **Video Window** (Color OR IR)! The grid is drawn directly on the screen.
  - **Green Line/Text:** Good coverage in that zone.
  - **Red Line/Text:** Zone needs more samples.

#### 2. Hands-Free "Smart Capture"

- **Old Way:** Press `s` on keyboard.
- **New Way (Smart):**
  1. Hold the board in a valid position.
  2. Hold it **steady** for 2 seconds.
  3. **BEEP!** (Visual Log: "AUTO-CAPTURE TRIGGERED!")
  4. Move to next position.
- **Result:** You can calibrate the whole camera without touching the keyboard.
- **Modes:** Works in `record color`, `record ir`, and `record sync`.

#### 3. Per-Image Analysis (Post-process)

After calibration (`calibrate` mode), every image is checked:

```
Img 5: 0.12 px [OK]
Img 6: 1.54 px [OUTLIER]
```

- **OK:** Error < 1.0 pixel. Good data.
- **OUTLIER:** Error > 1.0 pixel. Bad data (blurry, movement).
- **Action:** If you have many outliers, improve lighting and hold the board steadier.

#### 3. Quality Score

A final grade (0-100) combining three factors:

1. **Coverage (40%):** Did you fill the 3x3 map?
2. **Diversity (30%):** Did you capture enough frames (>30)?
3. **Accuracy (30%):** Is the reprojection error low?

**Target Score:** > 80/100.
**Passing Score:** > 60/100.

---

## 4. Troubleshooting

**Q: Integration with rtabmap or other packages?**
A: Remap topics in your launch file. The bridge publishes standard `sensor_msgs/Image` and `sensor_msgs/CameraInfo`.

**Q: OpenCL crashes on start?**
A: Ensure your host machine has OpenCL drivers (Intel Neo, NVIDIA CUDA) and they are passed to the Docker container (e.g., `--gpus all` or `--device /dev/dri`). If not available, stick to `reg_method:=cpu` (default).

**Q: "No device connected"?**
A: Ensure the Kinect v2 USB is plugged in and the container has USB access (`--privileged` or `-v /dev/bus/usb:/dev/bus/usb`).

**Q: "Permission denied" when accessing USB?**
A: Ensure you have added your user to the `plugdev` group and setup udev rules, or simply run the container with `--privileged`.
