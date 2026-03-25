# Docker Image Export & Gazebo Quick Start

## 📦 Docker Image Export Status

**Command Running:**
```bash
docker save parol6-ultimate:latest | gzip > ~/Desktop/parol6-ultimate-20260124.tar.gz
```

**Expected Size:** ~6-8 GB compressed (original 13.5 GB)  
**Time Estimate:** 10-20 minutes depending on disk speed

### Check Export Progress:

```bash
# Watch file size grow
watch -n 5 "ls -lh ~/Desktop/parol6-ultimate-*.tar.gz"

# Check if process is still running
ps aux | grep "docker save"
```

### When Export Completes:

The file `~/Desktop/parol6-ultimate-20260124.tar.gz` will be ready to share with teammates.

**To load on another machine:**
```bash
docker load < parol6-ultimate-20260124.tar.gz
```

---

## 🚀 Gazebo Quick Start (While Export Runs)

### The Issue You Had:

❌ **Wrong:** `ros2 launch parol6 gazebo.launch.py` (without sourcing workspace)  
✅ **Correct:** See below

### Solution 1: One-Line Launch

```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6 gazebo.launch.py"
```

### Solution 2: Interactive Session (Recommended)

```bash
# Step 1: Enter container
docker exec -it parol6_dev bash

# Step 2: Source workspace (CRITICAL!)
cd /workspace
source install/setup.bash

# Step 3: Launch Gazebo
ros2 launch parol6 gazebo.launch.py
```

### Solution 3: Use Existing Script

```bash
# From host machine
cd ~/Desktop/PAROL6_URDF
./start_ignition.sh
```

---

## 🎯 Next Steps Recommendation

Based on your Gazebo validation plan, here's the recommended workflow:

### Step 1: Test Gazebo Alone

```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 gazebo.launch.py
```

**Verify:**
- Gazebo window opens
- Robot model appears
- No error messages

### Step 2: Test Gazebo + MoveIt

```bash
# Terminal 1: Gazebo
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 gazebo.launch.py

# Terminal 2: MoveIt (wait 10 seconds)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**Verify:**
- RViz opens with robot model
- Interactive markers appear
- Can plan and execute motions

### Step 3: Full Vision Pipeline (Later)

Only after teammate captures Kinect snapshot:

```bash
# Terminal 1: Bag replay
ros2 bag play test_data/kinect_snapshot_* --loop

# Terminal 2: Gazebo
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 gazebo.launch.py

# Terminal 3: Vision
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

---

## 📚 Full Documentation

See `docs/GAZEBO_SETUP_GUIDE.md` for:
- Complete workflow explanations
- Troubleshooting common issues
- Mode switching (Gazebo vs. Real Robot)
- Integration with vision pipeline

---

## ⏰ Export Time Estimate

Started: ~17:30  
Expected completion: ~17:45 - 18:00

You can continue working with Gazebo while the export runs in the background!
