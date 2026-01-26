# 📸 TEAMMATE: Capture Kinect Snapshot - Complete Guide

**Purpose:** You have the Kinect camera. Follow these steps to capture a sensor snapshot and share it with the team.

---

## ✅ Prerequisites Checklist

Before starting, verify:
- [ ] You must to have the Red-Marker ROS bag "rosbag2_2026_01_26-23_26_59.zip"
- [ ] Kinect v2 camera physically connected to USB 3.0 port
- [ ] Docker container `parol6_dev` is running
- [ ] You have the latest code from the `Red-Marker-Detection` branch


---

## 📋 Step-by-Step Instructions

### Step 1: Start Docker Container
```bash
cd ~/Desktop/PAROL6_URDF  # Or wherever your repo is
./start_container.sh       # Start the container
```

### Step 2: Check Kinect Installation
```bash
docker exec -it parol6_dev bash -c "test -f /opt/kinect_ws/install/setup.bash && echo '✅ Kinect installed' || echo '❌ Need to install Kinect'"
```

**If you see "❌ Need to install Kinect"**, run:
```bash
docker exec -u 0 -it parol6_dev /workspace/scripts/install_kinect.sh
```
⏱️ This takes 10-20 minutes. Wait for completion.

### Step 3: Position the Camera
**IMPORTANT:** Set up your camera to view the workspace:

```
Camera Position (example):
┌─────────────┐
│   Kinect    │  ← Mount ~1m above workspace
│   Camera    │     Looking DOWN at ~45° angle
└─────────────┘
       ↓ 45°
  ┌──────────┐
  │ Workspace│  ← Target area with red lines
  │  (robot) │
  └──────────┘
```

**Tips:**
- Camera should be 0.5-1.0m above the workspace
- Tilted DOWN to see the work surface
- Red welding lines should be visible in the frame
- Stable mount (no wobbling)

### Step 4: Launch Kinect Bridge

**Terminal 1:**
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/kinect_ws/install/setup.bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch camera
ros2 launch kinect2_bridge kinect2_bridge.launch.py
```

**Wait 10-15 seconds** for camera initialization.

You should see:
```
[kinect2_bridge-1] [INFO] ... device serial: ...
[kinect2_bridge-1] [INFO] ... initializing device ...
[kinect2_bridge-1] [INFO] ... device initialized ...
```

### Step 5: Verify Camera is Working

**Terminal 2:**
```bash
docker exec -it parol6_dev bash
source /opt/ros/humble/setup.bash

# Check topics
ros2 topic list | grep kinect

# Check image is publishing
ros2 topic hz /kinect2/qhd/image_color_rect
```

**Expected output:**
- You should see `/kinect2/qhd/image_color_rect` and other topics
- Hz should show ~10-30 (meaning images are publishing)

**Optional - Preview Camera View:**
```bash
source /opt/kinect_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view
```
Select `/kinect2/qhd/image_color_rect` from dropdown. Verify red lines are visible.

### Step 6: Record Snapshot

**Terminal 2 (or new terminal 3):**
```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash

# Record 3-second snapshot
./parol6_vision/scripts/record_kinect_snapshot.sh 3 kinect_snapshot
```

**You have to manually press Ca**


**You'll see:**
```
==========================================
Kinect Snapshot Bag Recorder
==========================================
Duration: 3 seconds
Output: /workspace/test_data/kinect_snapshot_YYYYMMDD_HHMMSS

Checking topic availability...
  ✓ /kinect2/qhd/image_color_rect
  ✓ /kinect2/qhd/image_depth_rect
  ✓ /kinect2/qhd/camera_info
  ✓ /tf
  ✓ /tf_static

Starting recording in 2 seconds...
🔴 RECORDING...
✅ Recording complete!
```

### Step 7: Verify Recording

```bash
# Check bag was created
ls -lh /workspace/test_data/

# View bag info
ros2 bag info /workspace/test_data/kinect_snapshot_YYYYMMDD_HHMMSS
```

**Expected:**
- Directory created: `kinect_snapshot_YYYYMMDD_HHMMSS/`
- Size: 10-50 MB (depends on scene complexity)
- Topics: 5 (image_color, image_depth, camera_info, tf, tf_static)
- Messages: ~100-300 total

### Step 8: Test Replay Locally

**Stop Kinect (Terminal 1):**
Press `Ctrl+C`

**Terminal 1 - Replay:**
```bash
cd /workspace
source /opt/ros/humble/setup.bash
ros2 bag play test_data/kinect_snapshot_YYYYMMDD_HHMMSS --loop
```

**Terminal 2 - Verify:**
```bash
ros2 topic hz /kinect2/qhd/image_color_rect
```

Should show ~10 Hz. If yes, replay works! ✅

### Step 9: Compress for Sharing

**Stop bag replay (Ctrl+C), then:**
```bash
cd /workspace/test_data
tar czf kinect_snapshot_YYYYMMDD_HHMMSS.tar.gz kinect_snapshot_YYYYMMDD_HHMMSS/

# Check size
ls -lh kinect_snapshot_*.tar.gz
```

**Expected:** 5-30 MB compressed file

### Step 10: Copy to Host Machine

**On your host (outside Docker):**
```bash
docker cp parol6_dev:/workspace/test_data/kinect_snapshot_YYYYMMDD_HHMMSS.tar.gz ~/Desktop/
```

### Step 11: Share with Team

Upload `kinect_snapshot_YYYYMMDD_HHMMSS.tar.gz` to:
- **Google Drive** (recommended - easy sharing)
- **GitHub Release** (if < 25 MB)
- **USB drive**
- **University network share**

Send the download link to your teammates with a message like:

```
Hey team! 

I've captured a Kinect snapshot for camera-less development.

Download: [Your Google Drive Link]
File: kinect_snapshot_20260124_023000.tar.gz

To use it:
1. docker cp kinect_snapshot_20260124_023000.tar.gz parol6_dev:/workspace/test_data/
2. docker exec -it parol6_dev bash
3. cd /workspace/test_data && tar xzf kinect_snapshot_20260124_023000.tar.gz
4. ros2 bag play test_data/kinect_snapshot_20260124_023000 --loop

Then run the vision pipeline normally - no camera needed!
```

---

## 🎯 Camera TF Configuration (Reference)

The camera transform is already configured in the code at:
`parol6_vision/launch/camera_setup.launch.py`

**Current configuration:**
```python
# Camera positioned:
# - 0.5m forward (X)
# - 0.0m sideways (Y)  
# - 1.0m up (Z)
# - Pitched down 45° to look at workspace

arguments=['--x', '0.5', '--y', '0.0', '--z', '1.0',
           '--roll', '0.0', '--pitch', '-0.785', '--yaw', '0.0',
           '--frame-id', 'base_link', '--child-frame-id', 'kinect2_link']
```

**If you position the camera differently:**
1. Measure the actual position relative to robot base
2. Update these values in the launch file
3. Rebuild: `colcon build --packages-select parol6_vision`
4. Include updated launch file when sharing

---

## 🆘 Troubleshooting

### "kinect2_bridge: command not found"
**Fix:** Source the kinect workspace first:
```bash
source /opt/kinect_ws/install/setup.bash
```

### "No topics available" when checking
**Fix:** Wait longer (15-20 seconds) for camera initialization

### "Cannot find recording script"
**Fix:** Ensure you're in `/workspace` and pull latest code:
```bash
cd /workspace
git pull origin Red-Marker-Detection
```

### Camera shows black image
**Fix:**
- Check USB cable connection (must be USB 3.0 - blue port)
- Ensure camera power LED is on
- Try unplugging and replugging camera

### Recording shows "Topic not available" warnings
**Fix:** Launch kinect_bridge first and wait for stable publishing before recording

---

## ✅ Success Criteria

You've succeeded when:
- ✅ Bag file created (~10-50 MB)
- ✅ Local replay works
- ✅ Compressed .tar.gz created
- ✅ File copied to host machine
- ✅ Shared with team

---

## 📞 Need Help?

If stuck:
1. Check `/workspace/src/parol6_vision/docs/KINECT_SNAPSHOT_GUIDE.md` for details
2. Contact team lead (Kareem)
3. Share error messages + terminal output

---

**Thank you for capturing the snapshot! This will enable the entire team to develop without camera hardware. 🎉**
