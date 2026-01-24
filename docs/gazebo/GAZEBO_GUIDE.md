# Gazebo Simulation Guide for PAROL6

## 🎯 Purpose

Gazebo provides **visual simulation** of the robot for:
- Testing motion planning before real hardware
- Visualizing welding path execution
- Validating vision pipeline integration
- Demonstrating the system for thesis documentation

**Important:** Gazebo is **independent** from your other RViz windows:
- You can run MoveIt RViz (`demo.launch.py`) separately
- You can run Camera RViz (`camera_setup.launch.py`) separately
- Gazebo just adds a physics simulation view

---

## 🔧 Gazebo vs. Ignition - Which to Use?

### Ignition Gazebo (Recommended ✅)

**Launch file:** `ros2 launch parol6 ignition.launch.py`

**Pros:**
- Modern architecture
- Better performance
- Future-proof
- Works reliably in your Docker setup

**Cons:**
- May try to download models from fuel.ignitionrobotics.org on first launch
- More complex architecture (but well-supported)

### Standard Gazebo (Alternative)

**Launch file:** `ros2 launch parol6 gazebo.launch.py`

**Pros:**
- Simpler architecture
- Well-documented legacy support

**Cons:**
- Older technology
- Being deprecated
- May have compatibility issues with current setup
- Will be fully replaced by Ignition

**Recommendation for Your Thesis:** Use **Ignition Gazebo** (`ignition.launch.py`) for best compatibility with your Docker environment.

---

## 🚀 Quick Start

### Step 1: Source Environment

```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
```

###Step 2: Launch Gazebo

```bash
ros2 launch parol6 ignition.launch.py
```

**Expected:** Ignition Gazebo window opens with empty world

**Wait 10-15 seconds** for full initialization

### Step 3: Verify Robot is Loaded

```bash
# In a new terminal
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash

# Check if robot model is published
ros2 topic echo /robot_description --once

# Check controllers
ros2 control list_controllers
```

**Expected output:**
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
```

---

## 🎨 Usage Scenarios

### Scenario A: Gazebo + MoveIt (Motion Planning Visualization)

**Terminal 1: Gazebo**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**Terminal 2: MoveIt + RViz** (after Gazebo loads)
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**What you get:**
- Plan motions in RViz
- Execute them in Gazebo simulation
- See visual robot movement

---

### Scenario B: Gazebo + Vision Pipeline (Your Validation Plan)

**Terminal 1: ROS Bag (frozen camera data)**
```bash
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_* --loop
```

**Terminal 2: Gazebo**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**Terminal 3: Camera Visualization**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

**Terminal 4: MoveIt (optional, for execution)**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**What you get:**
- Vision pipeline detects red lines
- Generates welding paths
- Can execute paths in Gazebo
- All visualized simultaneously

---

### Scenario C: Gazebo Only (Quick Visual Check)

Just want to see the robot model in 3D simulation:

```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**No RViz needed** - Gazebo has its own GUI

---

## ⚠️ Troubleshooting

### Issue 1: "Package 'parol6' not found"

**Cause:** Workspace not sourced

**Fix:**
```bash
cd /workspace && source install/setup.bash
```

---

### Issue 2: Gazebo Hangs at "Preparing World"

**Symptoms:**
- Terminal shows "waiting for service /controller_manager"
- Gazebo window blank or doesn't appear

**Cause:** Network trying to download models, or controller manager timeout

**Fix Option A** - Kill and restart:
```bash
# In another terminal
docker exec parol6_dev bash -c "ps aux | grep gazebo"
# Note the PIDs, then:
docker exec parol6_dev bash -c "kill -9 <PID1> <PID2>"

# Restart
ros2 launch parol6 gazebo.launch.py
```

**Fix Option B** - Use Ignition instead (may avoid some network issues):
```bash
ros2 launch parol6 ignition.launch.py
```

**Fix Option C** - Skip Gazebo entirely:
```bash
# Just use RViz for visualization
ros2 launch parol6_moveit_config demo.launch.py
```

---

### Issue 3: No Gazebo Window Appears

**Check X11 forwarding:**
```bash
# On host machine
xhost +local:docker

# Inside container, verify DISPLAY
echo $DISPLAY  # Should show :0 or :1
```

**Check if Gazebo processes are running:**
```bash
docker exec parol6_dev bash -c "ps aux | grep -E 'gzserver|gzclient'"
```

If processes exist but no window:
- Check if window is minimized/hidden
- Try `wmctrl -l` to list windows
- Gazebo might have opened on different workspace

---

### Issue 4: Controllers Not Loading

**Symptoms:**
```
[spawner]: waiting for service /controller_manager/list_controllers
```

**Fix:**
```bash
# Manually spawn controllers after Gazebo loads
ros2 run controller_manager spawner joint_state_broadcaster
ros2 run controller_manager spawner parol6_arm_controller
```

---

## 🔍 Verification Checklist

After launching Gazebo, verify everything works:

```bash
# 1. Check active nodes
ros2 node list
# Should include: /gazebo, /robot_state_publisher, /controller_manager

# 2. Check topics
ros2 topic list
# Should include: /joint_states, /robot_description, /clock

# 3. Check controllers
ros2 control list_controllers
# Should show: joint_state_broadcaster, parol6_arm_controller as 'active'

# 4. Test robot movement
ros2 topic pub /parol6_arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "..." 
# (Advanced - use MoveIt instead for easier control)
```

---

## 📊 Integration with Thesis Validation

Per `parol6_vision/docs/GAZEBO_VALIDATION_PLAN.md`:

### Recommended Workflow:

1. **Capture Kinect Snapshot** (teammate with camera)
2. **Replay Bag + Gazebo** (deterministic testing)
3. **Run Vision Pipeline** (detect lines, generate paths)
4. **Execute in Gazebo** (visual validation)
5. **Record Metrics** (bag record output for analysis)

**Commands:**
```bash
# Terminal 1: Bag replay
ros2 bag play kinect_snapshot_20260124 --loop

# Terminal 2: Gazebo
ros2 launch parol6 gazebo.launch.py

# Terminal 3: Vision
ros2 launch parol6_vision camera_setup.launch.py

# Terminal 4: Record validation data
ros2 bag record /joint_states /tf /detected_lines_3d /welding_path
```

---

## 🎓 For Teammates

### First Time Setup

1. **Ensure Docker container is running:**
   ```bash
   docker ps | grep parol6_dev
   ```

2. **Source workspace (ALWAYS required):**
   ```bash
   docker exec -it parol6_dev bash
   cd /workspace && source install/setup.bash
   ```

3. **Launch Gazebo:**
   ```bash
   ros2 launch parol6 gazebo.launch.py
   ```

4. **Wait 10-15 seconds** for initialization

5. **Optional: Add MoveIt in new terminal** (for planning/execution)

### Daily Usage

**Quick launch (one-liner):**
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6 gazebo.launch.py"
```

---

## 📚 Related Documentation

- **Vision Pipeline:** `parol6_vision/docs/README.md`
- **Gazebo Validation Plan:** `parol6_vision/docs/GAZEBO_VALIDATION_PLAN.md`
- **MoveIt Setup:** `docs/TEAMMATE_COMPLETE_GUIDE.md`
- **Camera Setup:** `parol6_vision/docs/RVIZ_SETUP_GUIDE.md`
- **ROS Bag Workflow:** `parol6_vision/docs/rosbag/KINECT_SNAPSHOT_GUIDE.md`

---

## ❓ FAQ

**Q: Can I run Gazebo and Camera RViz at the same time?**  
A: Yes! They're independent. Gazebo shows robot simulation, Camera RViz shows camera feed visualization.

**Q: Do I need Gazebo to test the vision pipeline?**  
A: No. You can test vision detection with just `camera_setup.launch.py`. Gazebo is optional for visualization.

**Q: Gazebo vs. Real Robot - which to use?**  
A: Test in Gazebo first (safe), then switch to real robot. See `docs/REAL_ROBOT_INTEGRATION.md` for mode switching.

**Q: How do I stop Gazebo?**  
A: Press `Ctrl+C` in the terminal where it's running.

**Q: Gazebo crashed, how to clean up?**  
A:
```bash
docker exec parol6_dev bash -c "ps aux | grep gzserver | awk '{print \$2}' | xargs kill -9"
```

---

**Last Updated:** 2026-01-24  
**Maintainer:** PAROL6 Vision Team  
**Status:** Production Ready
