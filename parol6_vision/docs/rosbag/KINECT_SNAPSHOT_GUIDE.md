# Kinect Snapshot Dataset - Quick Start Guide

## 📋 Overview
This guide shows how to capture and replay a frozen Kinect sensor dataset for camera-less development.

## 🎯 Quick Capture (If You Have Camera)

### Step 1: Launch Kinect Bridge
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge.launch.py
```

Wait ~10 seconds for camera to initialize.

### Step 2: Record Snapshot (New Terminal)
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
./src/parol6_vision/scripts/record_kinect_snapshot.sh 3 kinect_snapshot
```

This captures 3 seconds of data to `/workspace/test_data/kinect_snapshot_<timestamp>/`

### Step 3: Stop Kinect
Press `Ctrl+C` in the kinect2_bridge terminal.

---

## 🔁 Replay Mode (For Development)

### Start Replay
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash

# Find your snapshot directory
ls test_data/

# Replay in loop
ros2 bag play test_data/kinect_snapshot_YYYYMMDD_HHMMSS --loop
```

### Launch Vision Pipeline (New Terminal)
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

**The pipeline now runs exactly as if the camera was connected!**

---

## 📤 Sharing with Teammates

### Compress Dataset
```bash
cd /workspace/test_data
tar czf kinect_snapshot_YYYYMMDD_HHMMSS.tar.gz kinect_snapshot_YYYYMMDD_HHMMSS/
```

### Share via:
- Google Drive
- USB drive
- University network
- Email (if < 25 MB)

⚠️ **Do NOT commit .bag files to git** - they're too large.

---

## 👥 Teammate Usage (No Camera Required)

### Step 1: Extract Dataset
```bash
# Copy tar.gz file to Docker container
docker cp kinect_snapshot.tar.gz parol6_dev:/workspace/test_data/

# Inside container:
docker exec -it parol6_dev bash
cd /workspace/test_data
tar xzf kinect_snapshot.tar.gz
```

### Step 2: Replay + Develop
```bash
# Terminal 1: Replay sensor data
ros2 bag play test_data/kinect_snapshot_YYYYMMDD_HHMMSS --loop

# Terminal 2: Run vision pipeline
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

✅ **Works without any camera hardware!**

---

## 🛠️ Validation Commands

### Check Topics Are Publishing
```bash
ros2 topic list | grep kinect
ros2 topic hz /kinect2/qhd/image_color_rect
```

### View Bag Info
```bash
ros2 bag info test_data/kinect_snapshot_YYYYMMDD_HHMMSS
```

### View Image in RViz
RViz should show the camera feed in the "Camera" display automatically.

---

## 📊 Troubleshooting

### "No topics available"
- Make sure `ros2 bag play --loop` is running
- Check topic names: `ros2 topic list`

### "Different message timestamp warnings"
- Normal for bags - ignore these warnings

### "Bag file not found"
- Check path: `ls -la test_data/`
- Ensure you're in `/workspace` directory

---

## 🎓 For Your Thesis

You can state:

> "To enable reproducible development and testing without hardware dependencies, a 3-second sensor snapshot was captured using ROS 2 bag and replayed in loop mode. This frozen dataset preserves synchronized RGB-D images, camera calibration, and TF transforms, allowing the entire vision → planning → simulation pipeline to execute deterministically."

---

## 📁 File Locations

```
/workspace/
├── test_data/
│   ├── kinect_snapshot_20260124_020000/  # Bag directory
│   │   ├── metadata.json                  # Dataset metadata
│   │   └── *.db3                          # Bag data files
│   └── kinect_snapshot_20260124_020000.tar.gz  # Compressed for sharing
└── src/parol6_vision/
    └── scripts/
        └── record_kinect_snapshot.sh      # Recording script
```
