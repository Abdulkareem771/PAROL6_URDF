# PAROL6 Vision Package Documentation

## 📍 Quick Navigation

### 🔭 Vision Pipeline GUI
**Full reference:** [`VISION_PIPELINE_GUI_GUIDE.md`](VISION_PIPELINE_GUI_GUIDE.md)

All-in-one PySide6 launcher — start nodes, preview topics, define crop ROI,
draw manual red-lines, and send paths to MoveIt.  
Includes a full **Troubleshooting** section.

### 🎥 For Teammates with Camera (Snapshot Capture)
**START HERE:** [`rosbag/TEAMMATE_START_HERE.md`](rosbag/TEAMMATE_START_HERE.md)

Complete snapshot capture workflow:
- Camera setup and positioning
- Recording ROS bag snapshot
- Sharing with team

### 📦 ROS Bag Snapshot System
**Directory:** [`rosbag/`](rosbag/)

- **`TEAMMATE_START_HERE.md`** - Quick entry point for camera capture
- **`TEAMMATE_CAPTURE_SNAPSHOT.md`** - Complete step-by-step guide
- **`KINECT_SNAPSHOT_GUIDE.md`** - Replay and development workflow

### 🖥️ RViz & Visualization
- **`RVIZ_SETUP_GUIDE.md`** - Setting up RViz for vision debugging
- Configuration files in `parol6_vision/config/vision_debug.rviz`

### 🧪 Testing
- **`TESTING_GUIDE.md`** - Unit and integration tests for vision pipeline
- **`GAZEBO_PATH_EXEC_CHECKLIST.md`** - End-to-end validation (Path Generator → MoveIt → Gazebo)
- **`MOVEIT_CONTROLLER_DEBUG_CONTEXT.md`** - Known execution failure signatures and recovery

### 📋 Implementation & Planning
- **`implementation_plan.md`** - Original vision pipeline architecture

---

## 📂 Documentation Structure

```
parol6_vision/docs/
├── README.md (this file)          # Navigation hub
├── rosbag/                        # ROS bag snapshot system
│   ├── TEAMMATE_START_HERE.md    # ← Start here if you have camera
│   ├── TEAMMATE_CAPTURE_SNAPSHOT.md
│   └── KINECT_SNAPSHOT_GUIDE.md
├── RVIZ_SETUP_GUIDE.md           # Visualization setup
├── TESTING_GUIDE.md              # Test procedures
└── implementation_plan.md        # Architecture details
```

---

## 🎯 Common Workflows

### I have the camera - capture snapshot
```bash
# Read this first
cat parol6_vision/docs/rosbag/TEAMMATE_START_HERE.md
```

### I want to develop without camera
```bash
# Get snapshot from teammate, then:
ros2 bag play test_data/kinect_snapshot_* --loop
ros2 launch parol6_vision camera_setup.launch.py
```

### I want to visualize in RViz
```bash
# Read setup guide
cat parol6_vision/docs/RVIZ_SETUP_GUIDE.md
```

### I want to validate path execution in Gazebo
```bash
cat parol6_vision/docs/GAZEBO_PATH_EXEC_CHECKLIST.md
```

---

## 📞 Questions?

- **Camera/Kinect issues:** See `rosbag/` directory
- **RViz problems:** See `RVIZ_SETUP_GUIDE.md`
- **Testing:** See `TESTING_GUIDE.md`
- **Architecture:** See `implementation_plan.md`
