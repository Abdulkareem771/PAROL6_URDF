# PAROL6 - Super Simple Usage Guide

## 🎯 **You Only Need These Commands!**

### **Start Gazebo (Terminal 1):**
```bash
### **Start Gazebo (Terminal 1):**
```bash
./start_ignition.sh
```
This automatically:
- ✅ Starts Docker container
- ✅ Builds the workspace
- ✅ Launches Ignition Gazebo with the robot
- ✅ Loads all controllers

**Wait for Gazebo window to appear!**

### **Add MoveIt + RViz (Terminal 2):**
```bash
./add_moveit.sh
```
This adds:
- ✅ MoveIt motion planning
- ✅ RViz visualization
- ✅ Interactive markers

### **Check Status:**
```bash
./status.sh
```
Shows what's running and system health.

### **Stop Everything:**
```bash
./stop.sh
```
Cleanly shuts down all components.

---

## 📋 **Complete Workflow**

### First Time Setup (One Time Only):
```bash
cd /home/kareem/Desktop/PAROL6_URDF
./start_ignition.sh
```

Wait ~30 seconds, then you'll see:
- Gazebo window with robot
- RViz window with MoveIt

### Daily Usage:
```bash
# Start your work session
./start_ignition.sh

# ... do your work ...

# End your work session
./stop.sh
```

---

## 🎮 **Using the Robot**

Once `./start_ignition.sh` finishes:

1. **In RViz window:**
   - Find "MotionPlanning" panel (left side)
   - Select planning group: `parol6_arm`
   - Drag the interactive marker (orange/blue sphere)
   - Click "Plan" button
   - Click "Execute" button

2. **Watch the robot move in Gazebo!**

---

## 🔧 **Additional Commands**

### Enter Container Shell:
```bash
docker exec -it parol6_dev bash
# Then inside:
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
# Now run any ROS commands
```

### Quick Status Check:
```bash
./status.sh
```

### View Logs:
```bash
docker logs parol6_dev
```

### Send Manual Command:
```bash
docker exec parol6_dev bash -c "
  source /opt/ros/humble/setup.bash && \
  source /workspace/install/setup.bash && \
  ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
    trajectory_msgs/msg/JointTrajectory \
    '{joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6], 
      points: [{positions: [0.5, -0.5, 0.5, 0.0, 0.0, 0.0], 
                time_from_start: {sec: 3}}]}'
"
```

---

## 🆘 **Troubleshooting**

### Problem: "Container already running"
```bash
./stop.sh
./start_ignition.sh
```

### Problem: "No module named 'moveit_configs_utils'"
```bash
./fix_current_container.sh
```

### Problem: "Windows don't appear"
```bash
# On host machine:
xhost +local:docker
./start_ignition.sh
```

### Problem: "Build fails"
```bash
# Check the log:
cat /tmp/parol6_build.log
```

### Problem: "Something is broken"
```bash
# Full reset:
./stop.sh
docker system prune -f
./start_ignition.sh
```

---

## 📁 **File Organization**

### Scripts (Run These):
- **`start.sh`** ← Start everything (use this!)
- **`stop.sh`** ← Stop everything
- **`status.sh`** ← Check what's running
- `launch.sh` ← Old interactive menu (still works)
- `test_setup.sh` ← Run tests

### Documentation (Read These):
- **`SIMPLE_USAGE.md`** ← This file (start here!)
- `docs/QUICKSTART.md` ← Quick start guide
- `docs/DOCUMENTATION.md` ← Complete technical docs
- `docs/CONTAINER_ARCHITECTURE.md` ← How Docker works
- `docs/INDEX.md` ← Documentation index

### Helper Scripts:
- `docs/QUICKREF.sh` ← Command reference
- `docs/CONTAINER_DIAGRAM.sh` ← Visual diagram
- `docs/launch.sh` ← Old interactive menu
- `docs/test_setup.sh` ← Run tests

---

## ✅ **Daily Checklist**

**Starting Work:**
- [ ] Run `./start.sh`
- [ ] Wait ~30 seconds
- [ ] See Gazebo and RViz windows
- [ ] Start developing!

**Ending Work:**
- [ ] Save your work
- [ ] Run `./stop.sh`
- [ ] Done!

---

## 🎓 **For New Team Members**

**Setup (5 minutes):**
1. Install Docker
2. Get this folder
3. Run `./start.sh`
4. Done!

**No need to install:**
- ❌ ROS 2
- ❌ MoveIt
- ❌ Gazebo
- ❌ Any dependencies

**Just Docker!** 🐳

---

## 💡 **Pro Tips**

1. **Keep it simple:** Just use `./start.sh` and `./stop.sh`
2. **Edit files normally:** Use VS Code, vim, whatever you like
3. **Files sync automatically:** Changes are instant
4. **Multiple terminals:** Use `docker exec -it parol6_dev bash`
5. **Check status anytime:** Run `./status.sh`

---

## 🚀 **Quick Reference Card**

```bash
# Start robot system
./start.sh

# Check if running
./status.sh

# Stop robot system
./stop.sh

# Enter container
docker exec -it parol6_dev bash

# View logs
docker logs parol6_dev

# Full reset
./stop.sh && docker system prune -f && ./start.sh
```

---

**That's it! You're ready to use the PAROL6 robot.** 🤖

**Questions?** Check `DOCUMENTATION.md` or ask a team member.
