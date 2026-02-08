# Kinect2 ROS2 Calibration Pack - Package Analysis

## Executive Summary

✅ **Status**: All dependencies are installed and the package is ready to use  
📦 **Package Version**: kinect2_ros2_calibration_pack  
🔧 **Installation Status**: Complete with all runtime dependencies satisfied

---

## Package Structure

The calibration pack consists of three ROS 2 packages:

### 1. kinect2_registration
**Purpose**: Depth registration and alignment between IR and RGB cameras

**Dependencies**:
- `rclcpp` - ROS 2 C++ client library ✅
- `std_msgs` - Standard message types ✅
- `Eigen3` - Linear algebra library ✅ (v3.4.0)

**Build Requirements**:
- OpenCL headers and libraries ✅ (v3.0, installed during workspace rebuild)
- C++17 compiler ✅

**Processing Methods**:
- CPU-based registration (default)
- OpenCL-based registration (GPU-accelerated, optional)

---

### 2. kinect2_bridge
**Purpose**: Main driver node that interfaces with libfreenect2 and publishes ROS topics

**Dependencies**:
- `rclcpp` ✅
- `std_msgs` ✅
- `sensor_msgs` ✅
- `tf2` / `tf2_ros` / `tf2_geometry_msgs` ✅
- `kinect2_registration` ✅
- `depth_image_proc` ✅
- `diagnostic_msgs` ✅
- `ament_index_cpp` ✅

**External Dependencies**:
- `libfreenect2` ✅ (installed at `/usr/local/lib/libfreenect2.so`)

**Launch Files**:
1. [`kinect2_bridge_launch.yaml`](file:///opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/launch/kinect2_bridge_launch.yaml) - Main driver launch
2. [`rtabmap.launch.py`](file:///opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/launch/rtabmap.launch.py) - RTAB-Map SLAM integration

---

### 3. kinect2_calibration
**Purpose**: Camera calibration tools with enhanced quality analysis

**Dependencies**:
- `rclcpp` ✅
- `std_msgs` ✅
- `sensor_msgs` ✅
- `message_filters` ✅
- `image_transport` ✅
- `compressed_image_transport` ✅
- `compressed_depth_image_transport` ✅
- `kinect2_registration` ✅
- `kinect2_bridge` ✅
- `cv_bridge` ✅

**Calibration Assets**:
- `color images to calibrate/` - 914 pre-collected calibration images
- Chess pattern files in `patterns/` directory
- Calibration scripts

---

## ✨ New Features in Calibration Pack

### 1. Digital IR Auto-Exposure
- **Feature**: Software-based dynamic range compression for IR camera
- **Use Case**: Essential for calibration in indoor lighting conditions
- **Parameter**: `ir_auto_exposure` (default: `false`)
- **Launch Example**:
  ```bash
  ros2 launch kinect2_bridge kinect2_bridge_launch.yaml ir_auto_exposure:=true
  ```

### 2. Depth Hole Filling
- **Feature**: Fills invalid pixels (black holes) in depth maps using neighboring values
- **Parameter**: `hole_fill_radius` (default: `0` = disabled, recommended: `1-3`)
- **Launch Example**:
  ```bash
  ros2 launch kinect2_bridge kinect2_bridge_launch.yaml hole_fill_radius:=3
  ```

### 3. Smart Auto-Capture
- **Feature**: Hands-free calibration - automatically captures images when board is held steady
- **Benefit**: No need to press keys during calibration
- **Trigger**: Hold chess pattern steady for 2 seconds
- **Visual Feedback**: Green/red overlay on video window

### 4. Coverage Map Visualization
- **Feature**: Real-time 3x3 grid showing calibration coverage
- **Display**: Terminal + video window overlay
- **Goal**: Achieve `[OK]` in all 9 cells for optimal calibration

### 5. Calibration Quality Engine
- **Feature**: Post-process analysis of each calibration image
- **Metrics**:
  - Per-image reprojection error (< 1.0 px = OK)
  - Overall quality score (0-100)
  - Outlier detection
- **Target Score**: > 80/100 (Minimum: > 60/100)

### 6. RTAB-Map Integration
- **Feature**: Ready-to-use SLAM configuration
- **Components**: Visual odometry + SLAM + Visualization
- **Launch Command**:
  ```bash
  ros2 launch kinect2_bridge rtabmap.launch.py
  ```

---

## Runtime Dependencies Status

### ROS 2 Packages (All Installed ✅)

| Package | Status | Purpose |
|---------|--------|---------|
| `rtabmap_ros` | ✅ | RTAB-Map SLAM |
| `rtabmap_slam` | ✅ | SLAM node |
| `rtabmap_odom` | ✅ | Visual odometry |
| `rtabmap_viz` | ✅ | Visualization |
| `depth_image_proc` | ✅ | Point cloud generation |
| `image_transport` | ✅ | Image transport |
| `compressed_image_transport` | ✅ | Compressed images |
| `compressed_depth_image_transport` | ✅ | Compressed depth |
| `cv_bridge` | ✅ | OpenCV-ROS bridge |

### System Libraries (All Installed ✅)

| Library | Version | Status | Purpose |
|---------|---------|--------|---------|
| `libfreenect2` | Latest | ✅ | Kinect v2 driver |
| `Eigen3` | 3.4.0 | ✅ | Linear algebra |
| `OpenCL headers` | 3.0 | ✅ | GPU acceleration |
| `ocl-icd-opencl-dev` | 2.2.14-3 | ✅ | OpenCL development |

### Visualization Tools (All Available ✅)

| Tool | Status | Purpose |
|------|--------|---------|
| `rviz2` | ✅ | 3D visualization |
| `rqt` | ✅ | ROS GUI framework |
| `rqt_image_view` | ✅ | Image viewer |

---

## Configuration Parameters

### Basic Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_name` | `kinect2` | Topic namespace |
| `fps_limit` | `30.0` | Frame rate limit (-1 = unlimited) |
| `publish_tf` | `true` | Publish TF transforms |
| `min_depth` | `0.1` | Minimum depth (meters) |
| `max_depth` | `4.0` | Maximum depth (meters) |
| `queue_size` | `5` | Message queue size |
| `worker_threads` | `4` | Processing threads |

### Advanced Parameters
| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `depth_method` | `cpu` | `cpu`, `opencl` | Depth processing method |
| `reg_method` | `cpu` | `cpu`, `opencl` | Registration method |
| `bilateral_filter` | `true` | bool | Edge-preserving filter |
| `edge_aware_filter` | `true` | bool | Edge-aware filtering |
| `hole_fill_radius` | `0` | `0-5` | Hole filling radius |
| `ir_auto_exposure` | `false` | bool | IR auto-exposure |

---

## Calibration Workflow

### Complete Calibration Sequence

1. **Color Camera Intrinsics**
   ```bash
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 record color
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate color
   ```

2. **IR Camera Intrinsics** (with auto-exposure recommended)
   ```bash
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 record ir
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate ir
   ```

3. **Extrinsic Calibration (Sync)**
   ```bash
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 record sync
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate sync
   ```

4. **Depth Calibration**
   ```bash
   ros2 run kinect2_calibration kinect2_calibration_node chess5x7x0.03 calibrate depth
   ```

### Calibration File Installation

After calibration, copy files to:
```
/opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/data/<DEVICE_SERIAL>/
```

Required files:
- `calib_color.yaml`
- `calib_ir.yaml`
- `calib_pose.yaml`
- `calib_depth.yaml`

---

## Published Topics

### RGB Camera
- `/kinect2/hd/image_color` - Full HD RGB (1920x1080)
- `/kinect2/qhd/image_color` - Quarter HD RGB (960x540)
- `/kinect2/qhd/image_color_rect` - Rectified QHD
- `/kinect2/hd/camera_info` - Camera intrinsics
- `/kinect2/qhd/camera_info` - Camera intrinsics

### Depth Camera
- `/kinect2/sd/image_depth` - Standard depth (512x424)
- `/kinect2/hd/image_depth_rect` - Registered HD depth
- `/kinect2/qhd/image_depth_rect` - Registered QHD depth

### IR Camera
- `/kinect2/sd/image_ir` - Infrared image

### Point Clouds
- `/kinect2/qhd/points` - XYZRGB point cloud
- `/kinect2/sd/points` - Depth-only point cloud

### TF Frames
- `kinect2_link` - Base frame
- `kinect2_rgb_optical_frame` - RGB camera frame
- `kinect2_ir_optical_frame` - IR camera frame

---

## ⚠️ Setup Requirements

### 1. Hardware Connection
- Kinect v2 must be connected to **USB 3.0 port**
- Container has USB access via `--privileged` flag ✅ (configured in `start_container.sh`)

### 2. Container Permissions
- Container running with privileged access ✅
- USB device access via `/dev:/dev` mount ✅

### 3. Calibration Files (User Action Required)
> [!WARNING]
> Default calibration files may not be accurate for your specific device. For optimal performance, you should:
> 1. Run the complete calibration sequence
> 2. Install calibration files for your device serial number
> 3. Commit the container to save calibration data

### 4. OpenCL for GPU Acceleration (Optional)
- OpenCL headers installed ✅
- To use GPU acceleration (`reg_method:=opencl`):
  - Requires host GPU drivers
  - Requires `--gpus all` or `--device /dev/dri` in Docker run
  - Current setup: CPU mode (stable and sufficient)

---

## 🚀 Quick Start Commands

### Basic Usage
```bash
# Inside container
docker exec -it parol6_dev bash
source /opt/kinect_ws/install/setup.bash

# Launch bridge (basic)
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml

# Launch with enhanced features
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml \
  ir_auto_exposure:=true \
  hole_fill_radius:=3 \
  fps_limit:=15
```

### Visualization
```bash
# RViz2
rviz2

# Image viewer
rqt_image_view

# Point cloud in RViz
# Add -> By topic -> /kinect2/qhd/points -> PointCloud2
```

### RTAB-Map SLAM
```bash
ros2 launch kinect2_bridge rtabmap.launch.py name:=kinect2
```

---

## 📋 Recommendations

### 1. Immediate Actions
- ✅ All dependencies installed
- ✅ Package built successfully  
- ⚠️ **Recommended**: Run camera calibration for your specific device
- ⚠️ **Recommended**: Commit container after calibration

### 2. Performance Optimization
- Use `fps_limit:=15` for most applications (reduces CPU load)
- Enable `hole_fill_radius:=3` for cleaner point clouds
- Use `reg_method:=cpu` (default) for stability
- Only use `reg_method:=opencl` if you have proper GPU setup

### 3. For Calibration
- Enable `ir_auto_exposure:=true` during calibration recording
- Use the coverage map to ensure complete 3x3 grid coverage
- Aim for quality score > 80/100
- Hold pattern steady for 2 seconds to trigger auto-capture

### 4. Data Persistence
> [!IMPORTANT]
> Remember to commit the container after:
> - Running calibration
> - Making any configuration changes
> ```bash
> docker commit parol6_dev parol6-ultimate:latest
> ```

---

## 🔍 Missing Components

### None Found ✅

All required dependencies and tools are installed:
- ✅ libfreenect2 driver
- ✅ All ROS 2 package dependencies
- ✅ OpenCL development headers
- ✅ RTAB-Map suite
- ✅ Visualization tools (RViz2, rqt)
- ✅ Image transport plugins

---

## 📚 Additional Resources

### User Guide Location
Inside container: `/opt/kinect_ws/src/kinect2_ros2/user_guide.md`

### Calibration Images
Pre-collected: `/opt/kinect_ws/src/kinect2_ros2/color images to calibrate/` (914 images)

### Launch Files
- Main bridge: `/opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/launch/kinect2_bridge_launch.yaml`
- RTAB-Map: `/opt/kinect_ws/src/kinect2_ros2/kinect2_bridge/launch/rtabmap.launch.py`

---

## Summary

✅ **Ready to Use**: The kinect2_ros2_calibration_pack is fully installed with all dependencies satisfied  
✨ **Enhanced Features**: IR auto-exposure, depth hole filling, smart calibration, RTAB-Map integration  
📦 **Complete Package**: All three packages built and functional  
🎯 **Next Step**: Run calibration sequence for your specific Kinect device to achieve optimal performance
