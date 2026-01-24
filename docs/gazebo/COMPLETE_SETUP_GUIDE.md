# PAROL6 Complete Setup & Visualization Guide

**For Teammates Using Antigravity**

This guide explains the complete PAROL6 system architecture and all visualization options (Gazebo, MoveIt RViz, Vision RViz).

---

## 🎯 System Overview

The PAROL6 project has **three independent visualization tools** that can run together:

```
┌─────────────────────────────────────────────────────────────┐
│                    PAROL6 ROS 2 System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Ignition   │  │ MoveIt RViz  │  │ Vision RViz  │     │
│  │    Gazebo    │  │              │  │              │     │
│  │              │  │              │  │              │     │
│  │ Robot        │  │ Motion       │  │ Camera +     │     │
│  │ Simulation   │  │ Planning     │  │ Line Detect  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│                  ┌────────▼────────┐                       │
│                  │  ROS 2 Core     │                       │
│                  │  Topics/Services│                       │
│                  └────────┬────────┘                       │
│                           │                                │
│                  ┌────────▼────────┐                       │
│                  │ Hardware Driver │                       │
│                  │ (ESP32/Motors)  │                       │
│                  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### The Three Visualization Tools

| Tool | Purpose | Launch Command | When to Use |
|------|---------|----------------|-------------|
| **Ignition Gazebo** | 3D physics simulation | `ros2 launch parol6 ignition.launch.py` | Testing motions safely, thesis videos |
| **MoveIt RViz** | Motion planning interface | `ros2 launch parol6_moveit_config demo.launch.py` | Planning trajectories, executing motions |
| **Vision RViz** | Camera + detection visualization | `ros2 launch parol6_vision camera_setup.launch.py` | Testing vision pipeline, line detection |

**Important:** All three can run **simultaneously** - they're independent!

---

## 🚀 Getting Started

### Prerequisites

1. **Docker container running:**
   ```bash
   docker ps | grep parol6_dev
   ```
   If not running, start it:
   ```bash
   cd ~/Desktop/PAROL6_URDF
   ./start_container.sh
   ```

2. **X11 forwarding enabled:**
   ```bash
   xhost +local:docker
   ```

---

## 📦 Option 1: Ignition Gazebo (Robot Simulation)

### What It Does
- Displays 3D robot model in physics simulation
- Shows robot movement in real-time
- Perfect for testing before using real hardware

### How to Launch

**Terminal 1:**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**Expected:** Ignition Gazebo window opens showing robot in empty world

**Wait 10-15 seconds** for full initialization

### Verify It's Working

```bash
# New terminal
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash

# Check if controllers loaded
ros2 control list_controllers
```

**Expected output:**
```
joint_state_broadcaster[...] active
parol6_arm_controller[...] active
```

### Common Issues

**Gazebo hangs?**
- Kill processes: `docker exec parol6_dev bash -c "pkill -9 gzserver gzclient"`
- Relaunch

**No window appears?**
- Check: `echo $DISPLAY` (should show `:0` or `:1`)
- Run: `xhost +local:docker`

---

## 🎨 Option 2: MoveIt + RViz (Motion Planning)

### What It Does
- Interactive motion planning interface
- Drag target poses, robot automatically plans path
- Execute planned motions (to Gazebo OR real robot)
- Shows collision objects, planning scene

### How to Launch

**Standalone (no Gazebo):**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**With Gazebo (for visual execution):**

Terminal 1 - Gazebo:
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

Terminal 2 - MoveIt (wait 10 seconds after Gazebo):
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

### What You'll See

**RViz window with:**
- Robot model (orange/white)
- Interactive markers (colored spheres/arrows)
- Planning request display
- Trajectory visualization

**How to use:**
1. Drag the interactive marker (colored sphere at end effector)
2. Click "Plan" button in Motion Planning panel
3. Blue trajectory appears if planning succeeds
4. Click "Execute" to run motion (in Gazebo if running, or real robot if connected)

### Common Uses

**Scenario A: Plan motions for thesis documentation**
- Plan → Screenshot → Include in thesis
- No Gazebo needed

**Scenario B: Test motions in simulation**
- Launch Gazebo first
- Plan → Execute → Watch in Gazebo

**Scenario C: Control real robot**
- Connect ESP32 (per `REAL_ROBOT_INTEGRATION.md`)
- Plan → Execute → Real motors move

---

## 📷 Option 3: Vision RViz (Camera + Line Detection)

### What It Does
- Shows Kinect camera feed (RGB, Depth, Point Cloud)
- Visualizes detected red lines in 3D
- Displays generated welding paths
- Shows TF frames (camera, robot, world)

### How to Launch

**With Live Camera:**
```bash
# Terminal 1: Start camera node
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

**With ROS Bag (Frozen Data for Testing):**
```bash
# Terminal 1: Play bag
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_* --loop

# Terminal 2: Launch vision RViz
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

### What You'll See

**RViz window with:**
- Camera feed overlay (top-left image view)
- Point cloud (3D colored points)
- Detected lines (red markers in 3D)
- TF tree (coordinate frames)
- Robot model

**Common Uses:**

**Scenario A: Test camera is working**
```bash
ros2 launch parol6_vision camera_setup.launch.py
```
→ See live RGB and depth feed

**Scenario B: Test line detection algorithm**
```bash
# Play bag with known test pattern
ros2 bag play test_data/kinect_snapshot_*
ros2 launch parol6_vision camera_setup.launch.py
```
→ Verify red lines are detected correctly

**Scenario C: Full vision-guided welding demo**
```bash
# Terminal 1: Bag replay
ros2 bag play test_data/kinect_snapshot_* --loop

# Terminal 2: Gazebo
ros2 launch parol6 ignition.launch.py

# Terminal 3: Vision
ros2 launch parol6_vision camera_setup.launch.py

# Terminal 4: MoveIt (optional, for execution)
ros2 launch parol6_moveit_config demo.launch.py
```
→ See complete pipeline: Camera → Detection → Path → Execution

---

## 🔧 Combining All Three

### Full System Demo

This shows the complete vision-guided welding pipeline:

```bash
# Terminal 1: Data source (ROS bag with frozen camera data)
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_20260124_* --loop

# Terminal 2: Simulation (robot visualization)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py

# Terminal 3: Vision (camera + line detection)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py

# Terminal 4: Planning (motion planning interface)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**You'll have 4 windows:**
1. Ignition Gazebo showing robot
2. Vision RViz showing camera + detected lines
3. MoveIt RViz showing motion planning
4. Terminal showing bag playback

**Workflow:** Camera detects line → Generates path → Plan motion → Execute in Gazebo

---

## 📚 Which Guide to Follow?

### For Initial Setup
→ `docs/TEAMMATE_COMPLETE_GUIDE.md`

**Covers:**
- Docker installation
- Building workspace
- ESP32 setup (if using real robot)
- First launch verification

### For Gazebo Simulation
→ `docs/gazebo/GAZEBO_GUIDE.md` (this folder)

**Covers:**
- Ignition vs Standard Gazebo
- Troubleshooting simulation issues
- Integration with thesis validation

### For Vision Pipeline
→ `parol6_vision/docs/README.md`

**Covers:**
- Camera calibration
- Line detection testing
- ROS bag capture/replay
- Vision RViz configuration

### For Real Robot
→ `docs/REAL_ROBOT_INTEGRATION.md`

**Covers:**
- ESP32 flashing
- Serial communication
- Motor control
- Mode switching (sim vs. real)

---

## 🎓 Typical Workflows

### Workflow 1: New Teammate First Launch

**Goal:** Verify everything is installed correctly

```bash
# Step 1: Test MoveIt (easiest, no external dependencies)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**Expected:** RViz opens, robot model visible, can drag interactive markers

**If this works:** Basic setup is correct!

```bash
# Step 2: Test Gazebo (simulation)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**Expected:** Gazebo window opens, robot appears

**If this works:** Graphics and simulation work!

```bash
# Step 3: Test Vision (if camera available)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

**Expected:** RViz opens with camera feed

**If all 3 work:** System is fully operational!

---

### Workflow 2: Thesis Validation (Your Use Case)

**Goal:** Test vision-guided path execution with frozen sensor data

Per `parol6_vision/docs/GAZEBO_VALIDATION_PLAN.md`:

1. **Capture snapshot** (if not done):
   ```bash
   # On machine with camera
   cd ~/Desktop/PAROL6_URDF/parol6_vision/scripts
   ./record_kinect_snapshot.sh
   ```

2. **Transfer to your machine** (if captured by teammate)

3. **Run validation**:
   ```bash
   # Terminal 1: Bag replay
   ros2 bag play test_data/kinect_snapshot_* --loop
   
   # Terminal 2: Vision RViz
   docker exec -it parol6_dev bash
   cd /workspace && source install/setup.bash
   ros2 launch parol6_vision camera_setup.launch.py
   
   # Terminal 3: Gazebo (for execution visualization)
   docker exec -it parol6_dev bash
   cd /workspace && source install/setup.bash
   ros2 launch parol6 ignition.launch.py
   
   # Terminal 4: Record results
   ros2 bag record /detected_lines_3d /welding_path /joint_states
   ```

4. **Collect metrics** from recorded bag for thesis

---

### Workflow 3: Real Robot Testing

**Goal:** Control physical PAROL6 robot

**Important:** Do NOT run Gazebo and real robot simultaneously!

```bash
# Step 1: Ensure Gazebo is NOT running
docker exec parol6_dev bash -c "pkill -9 gzserver gzclient"

# Step 2: Connect ESP32 via USB

# Step 3: Launch real robot driver
cd ~/Desktop/PAROL6_URDF
./start_real_robot.sh

# Step 4: (In separate terminal) Launch MoveIt
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py

# Step 5: Plan and execute (now goes to real motors!)
```

**Safety:** Always test in Gazebo first before real robot!

---

## 🔍 Verification Checklist

Use this to verify your setup is working:

### Docker & Workspace
- [ ] `docker ps | grep parol6_dev` shows running container
- [ ] `docker exec parol6_dev bash -c "ls /workspace/install"` shows packages
- [ ] `xhost +local:docker` runs without error

### MoveIt RViz
- [ ] `ros2 launch parol6_moveit_config demo.launch.py` opens RViz
- [ ] Robot model is visible (orange/white)
- [ ] Can drag interactive marker
- [ ] Planning succeeds (blue trajectory appears)

### Gazebo
- [ ] `ros2 launch parol6 ignition.launch.py` opens Gazebo window
- [ ] Robot model appears in simulation
- [ ] `ros2 control list_controllers` shows 2 active controllers
- [ ] `ros2 topic list | grep joint_states` shows topic exists

### Vision RViz
- [ ] `ros2 launch parol6_vision camera_setup.launch.py` opens RViz
- [ ] (If camera connected) Camera feed visible
- [ ] (If bag playing) Point cloud visible
- [ ] TF frames shown correctly

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Package 'parol6' not found"

**Cause:** Workspace not sourced

**Fix:**
```bash
cd /workspace && source install/setup.bash
```

**Permanent fix (optional):**
```bash
docker exec -it parol6_dev bash
echo "source /workspace/install/setup.bash" >> ~/.bashrc
```

---

### Issue 2: No RViz/Gazebo Window

**Cause:** X11 forwarding issue

**Fix:**
```bash
# On host machine
xhost +local:docker

# Inside container
echo $DISPLAY  # Should show :0 or :1
```

**If still not working:**
```bash
# Try restarting container
docker restart parol6_dev
```

---

### Issue 3: Gazebo Hangs on Launch

**Symptoms:** Terminal shows "waiting for service /controller_manager"

**Fix:**
```bash
# Kill stuck processes
docker exec parol6_dev bash -c "ps aux | grep gazebo | awk '{print \$2}' | xargs kill -9"

# Relaunch
ros2 launch parol6 ignition.launch.py
```

---

### Issue 4: Camera/Bag Conflict

**Symptoms:** Topics not updating or "failed to play bag"

**Cause:** Live camera and bag replay both publishing

**Fix:** See `parol6_vision/docs/rosbag/CAMERA_VS_BAG_CONFLICT.md`

**Quick solution:**
```bash
# Stop camera node first
pkill -9 kinect

# Then play bag
ros2 bag play ...
```

---

### Issue 5: Controllers Not Loading

**Symptoms:** MoveIt can't execute, "No controller" error

**Fix:**
```bash
# Check controller status
ros2 control list_controllers

# If not active, manually spawn
ros2 run controller_manager spawner joint_state_broadcaster
ros2 run controller_manager spawner parol6_arm_controller
```

---

## 📖 Documentation Map

```
PAROL6_URDF/
├── docs/
│   ├── TEAMMATE_COMPLETE_GUIDE.md ← Start here
│   ├── REAL_ROBOT_INTEGRATION.md ← For hardware control
│   ├── ROS_SYSTEM_ARCHITECTURE.md ← System design
│   └── gazebo/
│       ├── COMPLETE_SETUP_GUIDE.md ← YOU ARE HERE
│       ├── GAZEBO_GUIDE.md ← Detailed Gazebo reference
│       └── QUICK_START.md ← Quick commands only
│
└── parol6_vision/docs/
    ├── README.md ← Vision pipeline overview
    ├── RVIZ_SETUP_GUIDE.md ← Vision RViz details
    ├── GAZEBO_VALIDATION_PLAN.md ← Thesis methodology
    └── rosbag/
        ├── KINECT_SNAPSHOT_GUIDE.md ← How to capture data
        └── TEAMMATE_CAPTURE_SNAPSHOT.md ← Teammate instructions
```

---

## 🎯 Quick Reference Commands

### Source Workspace (ALWAYS REQUIRED)
```bash
cd /workspace && source install/setup.bash
```

### Launch Ignition Gazebo
```bash
ros2 launch parol6 ignition.launch.py
```

### Launch MoveIt RViz
```bash
ros2 launch parol6_moveit_config demo.launch.py
```

### Launch Vision RViz
```bash
ros2 launch parol6_vision camera_setup.launch.py
```

### Check System Status
```bash
ros2 node list                    # Active ROS nodes
ros2 topic list                   # Published topics
ros2 control list_controllers     # Controller status
```

### Kill Everything
```bash
 pkill -9 gzserver gzclient rviz2
```

---

## 💡 Tips for Antigravity Users

### Using Antigravity Assistant

**Good prompts:**
- "Launch MoveIt and plan a trajectory to position [x, y, z]"
- "Show me active ROS topics related to the camera"
- "Help me debug why Gazebo won't start"
- "Create a ROS bag recording of joint states"

**What Antigravity can help with:**
- Writing Python nodes for custom behaviors
- Modifying launch files
- Analyzing ROS bag data
- Debugging ROS communication issues
- Creating documentation

**What to handle manually:**
- Initial Docker setup (follow TEAMMATE_COMPLETE_GUIDE.md)
- Physical robot connections (ESP32, camera USB)
- X11 forwarding configuration

### Terminal Organization Tip

**Use 4 persistent terminals:**
1. **Term 1:** Host machine (for Docker commands, file editing)
2. **Term 2:** Inside Docker - for launching Gazebo
3. **Term 3:** Inside Docker - for launching RViz windows
4. **Term 4:** Inside Docker - for quick commands (topic list, node info, etc.)

Use `tmux` or `screen` to keep sessions persistent!

---

## ✅ Success Criteria

After following this guide, you should be able to:

- [ ] Launch any of the 3 visualization tools independently
- [ ] Run all 3 simultaneously for full system demo
- [ ] Switch between simulation and real robot
- [ ] Understand which tool to use for which task
- [  ] Troubleshoot common issues without help
- [ ] Navigate documentation to find specific guides


---

## 🔧 Recent Fixes and Troubleshooting (2026-01-24)

This section documents common issues encountered with the latest Docker image update and their solutions.

### Issue 1: Robot Doesn't Render in Ignition

**Symptoms:** Robot appears in entity tree but not in 3D viewport  
**Cause:** `IGN_GAZEBO_RESOURCE_PATH` not set correctly

**Solution:** Already fixed in [`ignition.launch.py`](file:///home/a7med/Desktop/PAROL6_URDF/PAROL6/launch/ignition.launch.py)

---

### Issue 2: parol6_vision Package Not Found

**Solution:**
```bash
colcon build --symlink-install --packages-skip esp32_feedback
```

---

### Issue 3: MoveIt Segmentation Fault with Ignition

**Cause:** Controller manager conflict

**Workaround:** Use MoveIt demo standalone (without Ignition) for planning

---

### Issue 4: PATH_TOLERANCE_VIOLATED Errors

**Symptoms:** Second motion fails with path tolerance error  
**Solution:** Already fixed in [`ros2_controllers.yaml`](file:///home/a7med/Desktop/PAROL6_URDF/parol6_moveit_config/config/ros2_controllers.yaml)

Path tolerances now set to:
- `trajectory: 0.1` rad (during execution)
- `goal: 0.05` rad (at final position)

---

## 🆘 Getting Help


**Check Documentation:**
1. This guide (complete setup)
2. `GAZEBO_GUIDE.md` (detailed Gazebo reference)
3. `parol6_vision/docs/README.md` (vision pipeline)
4. `REAL_ROBOT_INTEGRATION.md` (hardware control)

**Still stuck?**
- Check `docs/TROUBLESHOOTING.md`
- Use Antigravity: "@analyze why Gazebo won't launch"
- Ask your teammate who originally set up the system

---

**Last Updated:** 2026-01-24  
**Maintainer:** PAROL6 Vision Team  
**Status:** Production Ready for Thesis Work
