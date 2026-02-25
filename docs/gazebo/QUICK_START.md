# Gazebo Quick Reference

## 🚀 Launch Commands

### Ignition Gazebo (Recommended)
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6 ignition.launch.py"
```

### Standard Gazebo (Alternative)
```bash
docker exec -it parol6_dev bash -c "cd /workspace && source install/setup.bash && ros2 launch parol6 gazebo.launch.py"
```

---

## 🎯 Common Workflows

### Workflow 1: Gazebo + MoveIt (Motion Planning)

**Terminal 1:**
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**Terminal 2:** (wait 10 seconds)
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_moveit_config demo.launch.py
```

---

### Workflow 2: Vision Testing (with ROS Bag)

**Terminal 1:** Bag replay
```bash
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_* --loop
```

**Terminal 2:** Gazebo
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6 ignition.launch.py
```

**Terminal 3:** Camera visualization
```bash
docker exec -it parol6_dev bash
cd /workspace && source install/setup.bash
ros2 launch parol6_vision camera_setup.launch.py
```

---

## 🔧 Quick Fixes

### Kill Frozen Gazebo
```bash
docker exec parol6_dev bash -c "ps aux | grep gzserver | awk '{print \$2}' | xargs kill -9"
```

### Check if Running
```bash
docker exec parol6_dev bash -c "ps aux | grep -E 'gzserver|gzclient'"
```

### Verify Controllers
```bash
ros2 control list_controllers
```

---

## 📚 Full Documentation

See `docs/gazebo/GAZEBO_GUIDE.md` for complete guide.
