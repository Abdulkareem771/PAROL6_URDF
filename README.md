# PAROL6 Robot - Quick Reference

**One-page guide to get you started fast!**

> **✨ Now using `parol6-ultimate:latest` image with Gazebo, Ignition, and MoveIt pre-installed!**

---

## 🚀 **Quick Start (2 Steps)**

### Terminal 1: Start Ignition Gazebo
```bash
./start_ignition.sh
```
Wait for Ignition Gazebo window to appear with the robot.

### Terminal 2: Add MoveIt + RViz
```bash
./add_moveit.sh
```

**That's it!** Now you have Ignition Gazebo + MoveIt + RViz running.

---

## 🎮 **Using the Robot**

### In RViz:
1. Find **"MotionPlanning"** panel (left side)
2. Select planning group: **`parol6_arm`**
3. **Drag** the interactive marker (orange/blue sphere)
4. Click **"Plan"** → then **"Execute"**
5. Watch the robot move in Gazebo!

---

## 🔧 **Common Commands**

### Enter Container:
```bash
docker exec -it parol6_dev bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
```

### Check Controllers:
```bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 control list_controllers"
```

### Send Manual Movement:
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

## 🤖 **Robot Info**

| Property | Value |
|----------|-------|
| **Joints** | joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6 (6-DOF) |
| **Planning Group** | `parol6_arm` |
| **Named Poses** | `home`, `ready` |

### Joint Limits:
- **joint_L1** (base): -1.7 to 1.7 rad
- **joint_L2** (shoulder): -0.98 to 1.0 rad  
- **joint_L3** (elbow): -2.0 to 1.3 rad
- **joint_L4** (wrist pitch): -2.0 to 2.0 rad
- **joint_L5** (wrist roll): -2.1 to 2.1 rad
- **joint_L6** (end effector): continuous

---

## 🆘 **Troubleshooting**

### Problem: Container already running
```bash
./stop.sh
./start_ignition.sh
```

### Problem: Windows don't appear
```bash
xhost +local:docker
./start_ignition.sh
```

### Problem: "No module named 'moveit_configs_utils'"
This means dependencies are missing because the container was reset.
```bash
./fix_current_container.sh
```
Then try `./add_moveit.sh` again.

### Problem: Need to rebuild
```bash
docker exec parol6_dev bash -c "source /opt/ros/humble/setup.bash && cd /workspace && colcon build --symlink-install"
```

---

## 📚 **More Documentation**

- **`SIMPLE_USAGE.md`** - Detailed usage guide
- **`docs/`** - Complete documentation folder
  - `QUICKSTART.md` - Quick start guide
  - `DOCUMENTATION.md` - Complete technical reference
  - `CONTAINER_ARCHITECTURE.md` - Docker workflow explained
  - `ARCHITECTURE.md` - System architecture
  - `INDEX.md` - Documentation index

---

## 💡 **Important Notes**

- ✅ **Edit files** on your host machine (VS Code, vim, etc.)
- ✅ **Run ROS commands** inside the Docker container
- ✅ **Files sync automatically** - no copying needed
- ✅ **GUI apps** display on your screen

**This is the correct workflow!** See `docs/CONTAINER_ARCHITECTURE.md` for details.

---

## 🎯 **Daily Workflow**

```bash
# Morning
./start.sh

# ... work on your robot code ...
# Edit files with your favorite editor
# Test in Gazebo and RViz

# Evening
./stop.sh
```

---

## 📞 **Need Help?**

1. Check `SIMPLE_USAGE.md`
2. Run `./status.sh` to see what's running
3. Check `docs/` folder for detailed guides
4. Ask a team member

---

**Quick tip:** Bookmark this file - it has everything you need for daily use!
