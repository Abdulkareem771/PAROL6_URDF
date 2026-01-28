# Updated Gazebo Launch Guide (2026-01-24)

## 🎯 Current Situation

You have **3 Gazebo launch options** available:

1. **`gazebo.launch.py`** - Standard Gazebo (recommended)
2. **`ignition.launch.py`** - Ignition Gazebo (what `start_ignition.sh` uses)
3. **`gazebo_classic.launch.py`** - Legacy Gazebo Classic

**Old Scripts:**
- ❌ `start_ignition.sh` - Creates a NEW container (conflicts with running `parol6_dev`)
- ❌ `add_moveit.sh` - Uses old launch file `Movit_RViz_launch.py`

## ✅ Recommended NEW Workflow

### Option A: Standard Gazebo (Easiest)

```bash
# Terminal 1: Gazebo
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 gazebo.launch.py
```

Wait 10 seconds for Gazebo to fully load, then:

```bash
# Terminal 2: MoveIt + RViz
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**What this gives you:**
- Full robot simulation in Gazebo
- Motion planning with MoveIt
- Interactive control via RViz
- Controllers pre-loaded

---

### Option B: Ignition Gazebo (Advanced)

```bash
# Terminal 1: Ignition
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

Then in Terminal 2:
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

---

### Option C: Vision Pipeline + Gazebo (Validation Plan)

For testing the full vision → planning → execution pipeline:

```bash
# Terminal 1: ROS bag replay (when you have snapshot)
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_* --loop

# Terminal 2: Gazebo
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 gazebo.launch.py

# Terminal 3: Vision pipeline
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py

# Terminal 4: MoveIt (optional)
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

---

## 🔧 Why NOT Use Old Scripts?

### Problem with `start_ignition.sh`:
```bash
docker run -d --rm --name parol6_dev ...  # ❌ Tries to create NEW container
```

**Issue:** You already have `parol6_dev` running! This script would try to create a duplicate.

### Problem with `add_moveit.sh`:
```bash
ros2 launch parol6 Movit_RViz_launch.py  # ❌ Old launch file
```

**Issue:** Should use `parol6_moveit_config demo.launch.py` instead.

---

## 🚀 Should We Update the Scripts?

### Option 1: Create NEW simplified scripts

**`launch_gazebo.sh`:**
```bash
#!/bin/bash
exec docker exec -it parol6_dev bash -c "
  cd /workspace && 
  source install/setup.bash && 
  ros2 launch parol6 gazebo.launch.py
"
```

**`launch_moveit.sh`:**
```bash
#!/bin/bash
echo "Wait 10 seconds after Gazebo starts..."
sleep 10
exec docker exec -it parol6_dev bash -c "
  cd /workspace && 
  source install/setup.bash && 
  ros2 launch parol6_moveit_config demo.launch.py
"
```

### Option 2: Keep manual commands (recommended for now)

Since you're developing and testing, manual commands give you more control and understanding.

---

## 📝 Quick Reference Card

### To launch Gazebo:
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6 gazebo.launch.py"
```

### To launch MoveIt (after Gazebo):
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6_moveit_config demo.launch.py"
```

### To check if it's working:
```bash
# List active nodes
ros2 node list

# Check controllers
ros2 control list_controllers

# Monitor joint states
ros2 topic echo /joint_states
```

---

## ✅ Recommended Action Now

**Step 1:** Test basic Gazebo launch
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 gazebo.launch.py
```

**Expected:** Gazebo window opens, robot appears

**Step 2:** If successful, add MoveIt in new terminal
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

**Expected:** RViz opens, you can plan and execute motions

---

## 🎓 For Thesis Validation

Once your teammate captures the Kinect snapshot:
1. Use Option C workflow (Vision Pipeline + Gazebo)
2. Follow `parol6_vision/docs/GAZEBO_VALIDATION_PLAN.md`
3. Collect metrics for thesis

---

**Next Steps:**
- [ ] Test Gazebo launch (Option A, Terminal 1)
- [ ] Test MoveIt integration (Option A, Terminal 2)
- [ ] Verify robot can execute planned motions
- [ ] (Later) Integrate with vision pipeline when snapshot is ready
