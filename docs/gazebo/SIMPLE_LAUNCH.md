# ⚡ Simple Launch Reference (No ESP32)

Ultra-simple reference for daily development.

Choose one or run all in separate terminals.

---

## 🟦 Vision RViz

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

**Used for:**
- Camera visualization
- Vision algorithms
- Bag replay testing

---

## 🟩 MoveIt RViz

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py use_sim_time:=false
```

**Used for:**
- Motion planning
- Kinematics validation
- Collision checking

---

## 🟨 Gazebo

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
export IGN_GAZEBO_RESOURCE_PATH=/workspace/install/parol6/share:$IGN_GAZEBO_RESOURCE_PATH
ros2 launch parol6 ignition.launch.py
```

**Used for:**
- Simulation visualization
- Digital twin validation

---

## ⚠️ Important

- **ESP32 must NOT be connected**
- **No hardware drivers should be running**
- Each command runs independently
- Open each in a separate terminal
- Keep it simple

---

## ✅ Usage Rules

**Rule A — These are independent tools**

Users may run:
- One only
- Two simultaneously  
- All three simultaneously

In separate terminals. No interdependency.

**Rule B — Hardware must be OFF for this mode**

This guide applies only when:
- ESP32 is NOT connected
- No real motors
- No ros2_control hardware active

This prevents accidental misuse later.

---

**Last Updated:** 2026-01-24  
**Status:** Production Ready

---

## 🟦 Vision Only

**Purpose:** Image processing, YOLO testing, dataset replay, debugging perception

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

**What you'll see:**
- RViz window with camera feed
- Point cloud visualization
- TF frames
- Detected red lines (if test pattern visible)

**Use when:**
- Testing vision algorithms
- Replaying ROS bag data
- Debugging camera calibration
- Developing line detection

---

## 🟩 MoveIt Only

**Purpose:** Planning validation, kinematics, trajectory visualization

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py use_sim_time:=false
```

**What you'll see:**
- RViz window with robot model
- Interactive markers at end effector
- Motion planning interface
- Trajectory previews

**Use when:**
- Planning motions
- Testing inverse kinematics
- Validating workspace reach
- Creating trajectory demos for thesis

---

## 🟨 Gazebo Only

**Purpose:** Robot visualization, path following verification, collision debugging

```bash
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash
export IGN_GAZEBO_RESOURCE_PATH=/workspace/install/parol6/share:$IGN_GAZEBO_RESOURCE_PATH
ros2 launch parol6 ignition.launch.py
```

**What you'll see:**
- Ignition Gazebo window with 3D robot
- Physics simulation
- Robot in empty world

**Use when:**
- Visualizing planned motions before hardware
- Testing path following
- Debugging collisions
- Recording thesis videos

---

## 🧠 Key Rule

**You can run all three in separate terminals if needed.**

They remain completely independent:
- Vision RViz → Terminal 1
- MoveIt RViz → Terminal 2  
- Gazebo → Terminal 3

No combined scripts.  
No automation complexity.  
No fragile orchestration.

---

## 📦 Sensor Data Strategy

### Use ROS Bags (Not Live Camera)

**Why:**
- Deterministic behavior
- Easy sharing between teammates
- No hardware dependency
- Reproducible results for thesis

**Record snapshot:**
```bash
cd parol6_vision/scripts
./record_kinect_snapshot.sh
```

**Replay snapshot:**
```bash
ros2 bag play test_data/kinect_snapshot_20260124_* --loop
```

⚠️ **Never run camera and bag at the same time.**  
See: `parol6_vision/docs/rosbag/CAMERA_VS_BAG_CONFLICT.md`

---

## ✅ Verification Checklist

After following this guide, verify:

### Vision
- [ ] `ros2 launch parol6_vision camera_setup.launch.py` opens RViz
- [ ] Camera feed visible (if bag playing)
- [ ] Point cloud displays
- [ ] TF frames shown

### MoveIt
- [ ] `ros2 launch parol6_moveit_config demo.launch.py use_sim_time:=false` opens RViz
- [ ] Robot model visible
- [ ] Can drag interactive marker
- [ ] Planning succeeds (blue trajectory appears)

### Gazebo
- [ ] `ros2 launch parol6 ignition.launch.py` opens Gazebo window
- [ ] Robot appears in 3D world
- [ ] Controllers active: `ros2 control list_controllers`

---

## 🎓 For Thesis Work

### Digital Twin Strategy

We build the digital twin progressively:

**Stage 1 — Kinematic Twin** (Now)
- URDF correct
- Joint limits verified
- TCP frame validated
- Motion visually correct in Gazebo

**Stage 2 — Control Twin** (After ros2_control)
- Controllers mirror hardware interfaces
- Timing behavior matches hardware
- Joint feedback realistic

**Stage 3 — Process Twin** (Later)
- Welding path visualization
- Tool orientation validation
- Collision envelope checking

⚠️ We do NOT simulate welding physics, heat, or material behavior.

**Digital twin purpose:**
- Motion correctness
- Safety validation
- Planning verification
- Visualization for thesis

---

## 📚 Related Documentation

- **Vision details:** `parol6_vision/docs/RVIZ_SETUP_GUIDE.md`
- **ROS bag workflow:** `parol6_vision/docs/rosbag/KINECT_SNAPSHOT_GUIDE.md`
- **Camera conflict:** `parol6_vision/docs/rosbag/CAMERA_VS_BAG_CONFLICT.md`
- **Gazebo details:** `docs/gazebo/GAZEBO_GUIDE.md`
- **Full system:** `docs/gazebo/COMPLETE_SETUP_GUIDE.md`

---

**Philosophy:** Keep it simple. Each tool serves one purpose. Run what you need.

**Last Updated:** 2026-01-24  
**Status:** Production Ready
