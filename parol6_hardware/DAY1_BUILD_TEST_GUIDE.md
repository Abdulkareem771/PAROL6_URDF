# Day 1 SIL Build and Test Guide

**Complete step-by-step instructions for building and validating the ros2_control hardware interface.**

---

## 🔧 **Critical Fixes Applied**

All issues from code review have been fixed:

✅ **Fixed 1:** XML schema typo in `package.xml` removed  
✅ **Fixed 2:** Serial dependency corrected (`find_package(serial)` instead of pkg-config)  
✅ **Fixed 3:** Plugin XML installation added to CMakeLists.txt  
✅ **Fixed 4:** Logger initialized directly in header for safety  
✅ **Fixed 5:** Real PAROL6 URDF included (not fake simplified model)  
✅ **Fixed 6:** All file naming consistency verified

---

## 📋 **Prerequisites**

### 1. Docker Container Running

```bash
# From host machine
cd /home/kareem/Desktop/PAROL6_URDF
./start_container.sh

# Or if already running
docker exec -it parol6_dev bash
```

### 2. Verify ROS 2 Environment

Inside container:
```bash
# Should show ROS 2 Humble
ros2 --version

# Should be in workspace
pwd  # /workspace
```

### 3. Install Serial Package (if not already installed)

```bash
sudo apt-get update
sudo apt-get install -y ros-humble-serial
```

---

## 🏗️ **Build Process**

### Step 1: Clean Previous Builds (if any)

```bash
cd /workspace
rm -rf build/ install/ log/
```

### Step 2: Build Package

```bash
colcon build --packages-select parol6_hardware --symlink-install
```

**Expected output:**
```
Starting >>> parol6_hardware
Finished <<< parol6_hardware [XXs]

Summary: 1 package finished [XXs]
```

**If build fails, check:**
- Package dependencies: `rosdep install --from-paths src --ignore-src -r -y`
- Serial library: `dpkg -l | grep ros-humble-serial`
- Compiler errors in the output

### Step 3: Source the Workspace

```bash
source install/setup.bash
```

### Step 4: Verify Plugin Registration

```bash
# Check if plugin XML is installed
ls install/parol6_hardware/share/parol6_hardware/parol6_hardware_plugin.xml

# Should see: install/parol6_hardware/share/parol6_hardware/parol6_hardware_plugin.xml
```

---

## 🚀 **Launch and Validate**

### Terminal 1: Launch Hardware Interface

```bash
cd /workspace
source install/setup.bash
ros2 launch parol6_hardware real_robot.launch.py
```

**Expected output:**
```
[INFO] [controller_manager]: Loading controller 'joint_state_broadcaster'
[INFO] [controller_manager]: Configuring controller 'joint_state_broadcaster'
[INFO] [controller_manager]: Activating controller 'joint_state_broadcaster'
[INFO] [PAROL6System]: 🚀 Day 1: SIL Validation - Initializing PAROL6 Hardware Interface
[INFO] [PAROL6System]:   ✓ Joint: joint_L1
[INFO] [PAROL6System]:   ✓ Joint: joint_L2
... (6 joints total)
[INFO] [PAROL6System]: ✅ on_init() complete - 6 joints configured
[INFO] [controller_manager]: Loading controller 'parol6_arm_controller'
```

**If you see these logs → SUCCESS! Continue to validation.**

**If launch fails, common issues:**

| Error | Solution |
|-------|----------|
| `Plugin not found` | Check plugin XML installation |
| `Serial port error` | Normal for Day 1 - we don't open serial yet |
| `URDF not found` | Check `parol6` package is built |
| `Controller timeout` | Check update_rate in YAML |

---

### Terminal 2: Validate Controllers

```bash
# New terminal in container
docker exec -it parol6_dev bash
cd /workspace
source install/setup.bash

# List controllers
ros2 control list_controllers
```

**Expected output:**
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
parol6_arm_controller[joint_trajectory_controller/JointTrajectoryController] active
```

✅ **Both controllers should show "active"**

---

### Terminal 3: Monitor Joint States

```bash
# Check publication rate
ros2 topic hz /joint_states
```

**Expected output:**
```
average rate: 25.xxx
        min: 24.xxx ms
        max: 26.xxx ms
```

✅ **Should be ~25 Hz (±1 Hz is OK)**

```bash
# Check content (Ctrl+C to stop)
ros2 topic echo /joint_states
```

**Expected output:**
```
header:
  stamp:
    sec: ...
    nanosec: ...
  frame_id: ''
name:
- joint_L1
- joint_L2
- joint_L3
- joint_L4
- joint_L5
- joint_L6
position:
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
velocity:
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
```

✅ **All positions and velocities should be 0.0 (Day 1 SIL)**

---

### Terminal 4: Send Test Trajectory

```bash
ros2 action send_goal /parol6_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "{
    trajectory: {
      joint_names: [joint_L1, joint_L2, joint_L3, joint_L4, joint_L5, joint_L6],
      points: [
        {positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 0}},
        {positions: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}
      ]
    }
  }"
```

**Expected output:**
```
Waiting for an action server to become available...
Sending goal...
Goal accepted :)
Result:
    error_code: 0
Goal finished with status: SUCCEEDED
```

✅ **Goal should SUCCEED (even though motors don't move yet)**

---

## ✅ **Day 1 Success Criteria Checklist**

Run through this checklist:

- [ ] **Build:** Package compiles with no errors
- [ ] **Plugin:** Loads successfully (check controller_manager logs)
- [ ] **Controllers:** Both activate (check `list_controllers`)  
- [ ] **Topic:** `/joint_states` publishes at 25Hz
- [ ] **Data:** Positions/velocities are 0.0 (expected for SIL)
- [ ] **Trajectory:** Test action succeeds
- [ ] **Stability:** Runs for 5 minutes without crashes
- [ ] **Shutdown:** Clean exit with Ctrl+C (no segfaults)

**If ALL boxes checked → Day 1 COMPLETE! ✅**

---

## 📊 **Collect Evidence for Documentation**

After successful validation, capture this data:

### 1. Controller List
```bash
ros2 control list_controllers > day1_controllers.txt
```

### 2. Topic Rate
```bash
timeout 30 ros2 topic hz /joint_states > day1_topic_rate.txt
```

### 3. Hardware Interfaces
```bash
ros2 control list_hardware_interfaces > day1_hw_interfaces.txt
```

### 4. Logs
```bash
# From Terminal 1 (ctrl+C first to stop)
# Save the console output showing:
# - "[PAROL6System]: ✅ on_init() complete"
# - "[controller_manager]: ...active"
```

**Save these files** - they prove Day 1 SIL validation passed.

---

## 🐛 **Troubleshooting**

### Issue: Build fails with "serial not found"

```bash
# Install dependencies
sudo apt-get install ros-humble-serial

# Or use rosdep
rosdep install --from-paths src --ignore-src -r -y
```

### Issue: Plugin not discovered

```bash
# Verify plugin XML exists
cat install/parol6_hardware/share/parol6_hardware/parol6_hardware_plugin.xml

# Check pluginlib can find it
ros2 pkg prefix parol6_hardware
```

### Issue: URDF not found (parol6 package)

```bash
# Build parol6 package if needed
colcon build --packages-select parol6

# Or check if already installed
ros2 pkg prefix parol6
```

### Issue: Controllers don't activate

```bash
# Check controller manager status
ros2 control list_hardware_components

# Manually load and activate
ros2 control set_controller_state parol6_arm_controller inactive
ros2 control set_controller_state parol6_arm_controller active
```

### Issue: Segfault on exit

This is usually due to:
- Logger used before initialization (FIXED in latest code)
- Plugin unloading issue (check destructor)

**Solution:** Check that logger is initialized in header:
```cpp
rclcpp::Logger logger_{rclcpp::get_logger("PAROL6System")};
```

---

## 📈 **Performance Validation**

### Check Controller Timing

```bash
# Monitor controller updates
ros2 topic echo --once /controller_manager/state
```

**Look for:** `update_rate` should show ~25 Hz

### Check for Jitter

```bash
# Run for 60 seconds and analyze
timeout 60 ros2 topic hz /joint_states --window 100 > timing.txt
cat timing.txt
```

**Acceptable:**
- Average: 24.5 - 25.5 Hz
- Min: > 20 Hz
- Max: < 30 Hz
- Std dev: < 2 Hz

**Unacceptable (indicates blocking):**
- Large gaps (> 100ms)
- Frequent deadline misses
- Erratic timing

---

## 🎯 **Next Steps After Day 1 Success**

**Day 1 Complete →** Move to Day 2: Serial Communication

**Day 2 Preview:**
1. Add serial port opening in `on_configure()`
2. Implement `write()` to send commands to ESP32
3. Test with ESP32 connected (no motors)
4. Validate no controller blocking

**Before Day 2, you need:**
- [ ] ESP32 flashed with compatible firmware
- [ ] Serial cable connected (`/dev/ttyUSB0` or `/dev/ttyACM0`)
- [ ] Serial permissions set (`sudo usermod -a -G dialout $USER`)

---

## 📚 **Documentation to Create Next**

After Day 1 validation:

1. **HARDWARE_INTERFACE_GUIDE.md** - Detailed C++ class documentation
2. **Day 1 Validation Report** - Evidence of successful SIL
3. **Team Onboarding** - Instructions for colleagues

---

**Day 1 Build & Test Guide Complete!**  
**Follow this guide step-by-step for successful validation.** ✅
