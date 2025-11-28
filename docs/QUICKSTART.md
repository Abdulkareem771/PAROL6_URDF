# PAROL6 Robot - Quick Start Guide

**For:** New team members  
**Time to get running:** 5 minutes  
**Last updated:** 2025-11-27

---

## ⚠️ IMPORTANT: This Project Runs Inside Docker

**All ROS 2, MoveIt, and Gazebo commands run inside the Docker container.**

- ✅ **Edit files** on your host machine (with VS Code, vim, etc.)
- ✅ **Run commands** inside the Docker container
- ✅ **View GUI** on your host screen (RViz, Gazebo)
- ✅ **Files sync automatically** between host and container

**This is intentional and the recommended way!** See `CONTAINER_ARCHITECTURE.md` for details.

---

## 🚀 What is This?

PAROL6 is a 6-axis robotic arm with full ROS 2 + MoveIt integration. This guide gets you up and running in 5 minutes.

---

## ⚡ Super Quick Start (3 Steps)

### Step 1: Start the Container

```bash
cd /home/kareem/Desktop/PAROL6_URDF

docker run -it --rm \
  --name parol6_dev \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/home/kareem/Desktop/PAROL6_URDF:/workspace" \
  parol6-robot:latest
```

### Step 2: Build & Source (Inside Container)

```bash
source /opt/ros/humble/setup.bash
cd /workspace
colcon build --symlink-install
source install/setup.bash
```

### Step 3: Launch!
```bash
./start_ignition.sh
```

Then in a new terminal:
```bash
./add_moveit.sh
```

---

## 🎮 Using the Robot

### In Gazebo:
- Robot spawns in the world
- Controllers automatically load
- Use RViz or command line to control

### In RViz (MoveIt):
1. Look for **"MotionPlanning"** panel on the left
2. Select planning group: **`parol6_arm`**
3. **Drag the interactive marker** (orange/blue sphere at end effector)
4. Click **"Plan"** button
5. Click **"Execute"** to move the robot

### Quick Commands:

```bash
# Check if controllers are running
ros2 control list_controllers

# Expected output:
# parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
# joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active

# Monitor joint positions
ros2 topic echo /joint_states

# Send a test movement
ros2 topic pub --once /parol6_arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6], 
    points: [{positions: [0.0, -0.5, 0.5, 0.0, 0.0, 0.0], 
              time_from_start: {sec: 2}}]}"
```

---

## 📋 Common Tasks

### Task: Run Tests
```bash
cd /home/kareem/Desktop/PAROL6_URDF
./test_setup.sh
```

### Task: Use Interactive Menu
```bash
cd /home/kareem/Desktop/PAROL6_URDF
./launch.sh
```

### Task: Enter Running Container
```bash
docker exec -it parol6_dev bash
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
```

### Task: Stop Everything
```bash
# Press Ctrl+C in the terminal
# Or from outside:
docker stop parol6_dev
```

---

## 🤖 Robot Info

| Property | Value |
|----------|-------|
| **Joints** | joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6 (6-DOF) |
| **Planning Group** | `parol6_arm` |
| **End Effector** | joint_L6 |
| **Named Poses** | `home` (all zeros), `ready` (pre-configured) |

### Joint Limits Quick Reference:
- **joint_L1** (base): -1.7 to 1.7 rad
- **joint_L2** (shoulder): -0.98 to 1.0 rad
- **joint_L3** (elbow): -2.0 to 1.3 rad
- **joint_L4** (wrist pitch): -2.0 to 2.0 rad
- **joint_L5** (wrist roll): -2.1 to 2.1 rad
- **joint_L6** (end effector): continuous

---

## 🐍 Python Example

```python
#!/usr/bin/env python3
import rclpy
from moveit.planning import MoveItPy

# Initialize
rclpy.init()
moveit = MoveItPy(node_name="parol6_demo")
arm = moveit.get_planning_component("parol6_arm")

# Move to home position
arm.set_goal_state(configuration_name="home")
plan = arm.plan()
if plan:
    moveit.execute(plan.trajectory, controllers=[])
    print("✓ Moved to home!")

rclpy.shutdown()
```

Save as `test_move.py` and run:
```bash
python3 test_move.py
```

---

## 🆘 Troubleshooting

### Problem: "Package not found"
```bash
source /workspace/install/setup.bash
```

### Problem: "Controllers not loading"
```bash
# Check controller manager is running
ros2 control list_controllers

# If empty, restart the launch file
```

### Problem: "Planning fails"
- Make sure goal is reachable (not too far)
- Try different planner in RViz
- Increase planning time

### Problem: "Gazebo crashes"
```bash
export QT_X11_NO_MITSHM=1
# Then restart Gazebo
```

### Problem: "No display / X11 error"
```bash
# On host machine:
xhost +local:docker
```

---

## 📚 More Information

- **Complete Documentation:** `DOCUMENTATION.md`
- **System Architecture:** `ARCHITECTURE.md`
- **Setup Details:** `SETUP_COMPLETE.md`
- **Command Reference:** Run `./QUICKREF.sh`
- **Navigation Guide:** `INDEX.md`

---

## 🎯 Next Steps After Quick Start

1. **Experiment in RViz:**
   - Try moving to different positions
   - Use named states ("home", "ready")
   - Test different motion planners

2. **Learn the API:**
   - Run example: `python3 parol6_moveit_config/scripts/example_controller.py`
   - Read: `DOCUMENTATION.md` Section 7

3. **Understand the System:**
   - Read: `ARCHITECTURE.md`
   - Explore configuration files in `parol6_moveit_config/config/`

4. **Develop Your Application:**
   - Write Python/C++ control scripts
   - Add custom named poses
   - Integrate with your workflow

---

## 💡 Pro Tips

- **Use the interactive launcher:** `./launch.sh` - easiest way to start
- **Run tests first:** `./test_setup.sh` - verify everything works
- **Keep this terminal open:** Don't close the Docker container terminal
- **Use multiple terminals:** `docker exec -it parol6_dev bash` for new sessions
- **Check logs:** `ros2 run rqt_console rqt_console` for debugging

---

## 📞 Getting Help

1. **Check the docs:** Start with `INDEX.md`
2. **Run diagnostics:** `./test_setup.sh`
3. **View reference:** `./QUICKREF.sh`
4. **Ask the team:** Share this guide with colleagues!

---

## ✅ Checklist for New Users

- [ ] Docker container starts successfully
- [ ] Workspace builds without errors
- [ ] Can launch Gazebo simulation
- [ ] Can see robot in RViz
- [ ] Can plan and execute motions
- [ ] Controllers are active
- [ ] Understand basic commands

**Once you complete this checklist, you're ready to develop!** 🎉

---

**Questions?** Check `DOCUMENTATION.md` or ask a team member who's already set up.

**Happy robot programming!** 🤖
