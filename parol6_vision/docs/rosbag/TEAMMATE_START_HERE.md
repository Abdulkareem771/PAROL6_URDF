# 🔵 For Teammate with Camera - START HERE

## 📍 Your Task
You have the Kinect camera. Capture a sensor snapshot for the team.

## 📖 Complete Guide Location
**Read this file:** `docs/TEAMMATE_CAPTURE_SNAPSHOT.md`

Open it and follow Steps 1-11.

## ⚡ Quick Summary (5 minutes)

```bash
# 1. Start container
./start_container.sh

# 2. Launch Kinect (Terminal 1)
docker exec -it parol6_dev bash
cd /workspace
source /opt/kinect_ws/install/setup.bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# 3. Record snapshot (Terminal 2 - wait 10 sec after step 2)
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
./src/parol6_vision/scripts/record_kinect_snapshot.sh 3 kinect_snapshot

# 4. Compress
cd /workspace/test_data
tar czf kinect_snapshot_*.tar.gz kinect_snapshot_*/

# 5. Copy to host
docker cp parol6_dev:/workspace/test_data/kinect_snapshot_*.tar.gz ~/Desktop/

# 6. Share on Google Drive (or USB)
```

## ✅ What's Included

**Everything is in the guide:**
- ✅ Kinect installation check
- ✅ Camera positioning instructions  
- ✅ Camera TF configuration (already in code)
- ✅ Step-by-step recording process
- ✅ Verification steps
- ✅ Compression and sharing
- ✅ Troubleshooting section

## 🎯 Camera Position Reference

```
     Kinect
       ↓ (mounted ~1m high, angled down 45°)
   ┌────────┐
   │ Robot  │  ← Red lines visible in frame
   │ Base   │
   └────────┘
```

Camera TF is **already configured** in:
`parol6_vision/launch/camera_setup.launch.py`

(No changes needed unless camera position is very different)

## 📞 Questions?
- Full guide: `docs/TEAMMATE_CAPTURE_SNAPSHOT.md`
- Technical details: `parol6_vision/docs/KINECT_SNAPSHOT_GUIDE.md`
- Kinect setup: `docs/KINECT_INTEGRATION.md`

**After capturing, share the .tar.gz file with the team via Google Drive!**
