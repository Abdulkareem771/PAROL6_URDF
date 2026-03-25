# parol6_vision - Vision-Guided Welding Path Detection

**Package:** `parol6_vision`  
**Type:** ROS2 Python Package  
**Purpose:** Complete perception → planning → execution pipeline for red line welding path detection

---

## Overview

This package implements the vision system for the PAROL6 robot to detect red marker lines, project them into 3D space, generate smooth welding trajectories, and execute them using MoveIt2.

### System Architecture

```
Kinect v2 → Red Line Detector → Depth Matcher → Path Generator → MoveIt Controller → Robot
            (WeldLineArray)      (WeldLine3DArray)    (Path)        (Trajectory)
```

---

## Nodes

### 1. red_line_detector
Detects red marker lines using computer vision (HSV color segmentation, morphology, skeletonization).

**Published Topics:**
- `/vision/weld_lines_2d` (parol6_msgs/WeldLineArray)
- `/red_line_detector/debug_image` (sensor_msgs/Image)
- `/red_line_detector/markers` (visualization_msgs/MarkerArray)

### 2. depth_matcher
Projects 2D detections to 3D using depth camera data.

**Published Topics:**
- `/vision/weld_lines_3d` (parol6_msgs/WeldLine3DArray)

### 3. path_generator
Generates smooth welding trajectories from 3D points.

**Published Topics:**
- `/vision/welding_path` (nav_msgs/Path)

### 4. moveit_controller
Executes welding paths using MoveIt2 with fallback strategies.

**Services:**
- `~/execute_welding_path` (std_srvs/Trigger)

---

## Building the Package

### Prerequisites
- `parol6_msgs` package built and sourced
- ROS2 Humble
- Python dependencies: opencv-python, numpy, scipy, scikit-image

### Build Steps

```bash
# Inside Docker container
cd /workspace

# Build
colcon build --packages-select parol6_vision

# Source
source install/setup.bash

# Verify
ros2 pkg list | grep parol6_vision
```

---

## Configuration

Edit configuration files in `config/`:

1. **camera_params.yaml** - Update after camera calibration
2. **detection_params.yaml** - Tune HSV ranges for your red marker
3. **path_params.yaml** - Adjust welding parameters

---

## Launching

```bash
# Full pipeline
ros2 launch parol6_vision vision_pipeline.launch.py

# Detector only (for tuning)
ros2 launch parol6_vision red_detector_only.launch.py

# With RViz
ros2 launch parol6_vision vision_pipeline.launch.py use_rviz:=true
```

---

## Testing

See [VISION_DEVELOPER_GUIDE.md](file:///home/osama/Desktop/PAROL6_URDF/docs/VISION_DEVELOPER_GUIDE.md) for comprehensive testing procedures.

---

## Documentation

- **[Vision Developer Guide](file:///home/osama/Desktop/PAROL6_URDF/docs/VISION_DEVELOPER_GUIDE.md)** - Complete implementation reference
- **[Implementation Plan](file:///home/osama/.gemini/antigravity/brain/d2ceb7f4-79cd-48c2-8f8b-a2b6c9627e7c/implementation_plan.md)** - System design and architecture
- **[parol6_msgs README](file:///home/osama/Desktop/PAROL6_URDF/parol6_msgs/README.md)** - Custom message documentation

---

**Status:** Package structure complete, nodes implementation in progress  
**Version:** 1.0.0
