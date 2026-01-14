# Common Issues & Troubleshooting

**Solutions to problems encountered during development and testing**

This guide covers real issues that came up during development, and how to fix them.

---

## 🚨 Critical Issues (Must Fix First)

### Issue 1: Driver Message Format Mismatch

**Symptoms:**
```
ESP32 logs show: "Invalid message format"
No commands being processed
```

**Cause:** Driver sending wrong format (missing sequence number)

**Fix:**
The driver MUST be rebuilt after any code changes:

```bash
docker exec -it parol6_dev bash
cd /workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select parol6_driver
source install/setup.bash
exit
```

**Verify fix:**
```bash
# ESP32 should show:
I (1234) BENCHMARK: SEQ:0 J:[0.000,0.000,0.000,0.000,0.000,0.000]
```

**Root cause:** Code changes in `parol6_driver/parol6_driver/real_robot_driver.py` require rebuilding the package.

---

### Issue 2: Container Name Confusion

**Symptoms:**
```
Error: No such container: parol6_dev
# or
Error: No such container: parol6_real
```

**Cause:** Different scripts use different container names

**Fix:**

Check which containers exist:
```bash
docker ps -a | grep parol6
```

**If you see `parol6_real` but scripts expect `parol6_dev`:**
```bash
# Stop old container
docker stop parol6_real
docker rm parol6_real

# Restart with correct name
./start_real_robot.sh  # Now uses parol6_dev
```

**Permanent fix:** All scripts now use `parol6_dev` consistently.

---

### Issue 3: RViz Opens But Robot Not Visible

**Symptoms:**
- RViz window opens
- Only grid visible, no robot model
- Empty scene

**Possible Causes & Fixes:**

**A. Wrong RViz config path**

Check launch file is using correct path:
```python
# In parol6_driver/launch/real_robot_viz.launch.py
rviz_config_file = os.path.join(
    get_package_share_directory("parol6_moveit_config"),
    "rviz",  # ← Must be "rviz" not "config"
    "moveit.rviz"
)
```

**B. Camera view too far**

In RViz:
1. Views panel → Reset View
2. Or set Distance: 3-4 meters
3. Click "Zero" button

**C. Robot Alpha set to 0**

In RViz left panel:
1. Displays → MotionPlanning → expand
2. Find "Robot Alpha"
3. Set to 1.0

**D. Fixed Frame wrong**

Top of RViz left panel:
- Fixed Frame should be `world` or `base_link`
- Try switching between them

---

### Issue 4: Interactive Markers Not Showing

**Symptoms:**
- Robot visible in RViz
- Can't find orange sphere with 3 arrows
- Can't drag end-effector

**Debug: Check if they're running**
```bash
docker exec -it parol6_dev bash
ros2 topic list | grep marker
```

Should show:
```
/rviz_.../robot_interaction_interactive_marker_topic/update
/rviz_.../robot_interaction_interactive_marker_topic/feedback
```

**If topics exist but markers invisible:**

**Fix 1: Reset camera**
- Bottom panel → MotionPlanning → Click "Reset" button
- Markers might be out of view

**Fix 2: Increase marker size**
- Left panel → MotionPlanning → Planning Request
- Set "Interactive Marker Size" to 0.3 or higher

**Fix 3: Enable Query Goal State**
- Left panel → MotionPlanning → Planning Request
- Enable: "Query Goal State" ✓
- Disable: "Query Start State"
- Check: "Allow External Comm." ✓

**If topics don't exist:**

RViz config file has markers disabled. The config path fix (Issue 3A) should solve this.

---

## ⚙️ Build & Environment Issues

### Issue 5: Python Environment Conflicts

**Symptoms:**
```
ModuleNotFoundError: No module named 'catkin_pkg'
# or
ModuleNotFoundError: No module named 'ament_package'
```

**Cause:** ESP-IDF's Python environment conflicts with ROS

**Fix (One-time):**
```bash
docker exec -it parol6_dev bash
cd /workspace
./fix_python_env.sh
```

This adds `export PYTHON_EXECUTABLE=/usr/bin/python3` to `~/.bashrc`

**Verify:**
```bash
echo $PYTHON_EXECUTABLE
# Should output: /usr/bin/python3
```

**Manual fix (if script fails):**
```bash
export PYTHON_EXECUTABLE=/usr/bin/python3
colcon build --symlink-install
```

---

### Issue 6: ESP32 Serial Port Permission Denied

**Symptoms:**
```
Permission denied: '/dev/ttyUSB0'
```

**Quick fix:**
```bash
sudo chmod 666 /dev/ttyUSB0
```

**Permanent fix:**
```bash
sudo usermod -a -G dialout $USER
# Logout and login again
```

---

### Issue 7: Multiple Programs Using Serial Port

**Symptoms:**
```
Serial port busy
100% packet loss in tests
```

**Cause:** ESP32 monitor or another program is using the port

**Fix:**

1. **Close ESP32 monitor:**
   - Press `Ctrl + ]` in monitor terminal

2. **Find what's using the port:**
   ```bash
   sudo lsof /dev/ttyUSB0
   ```

3. **Kill the process:**
   ```bash
   sudo kill <PID>
   ```

4. **Reset ESP32:**
   - Press RESET button on board

---

## 🔧 ROS-Specific Issues

### Issue 8: MoveIt Segmentation Fault on Shutdown

**Symptoms:**
```
Segmentation fault (Address not mapped to object)
[ERROR] [move_group-4]: process has died [pid XXX, exit code -11]
```

**When:** Only when pressing Ctrl+C to stop

**Is this a problem?** **NO!** 

This is a known bug in ROS 2 Humble + MoveIt that happens **only on shutdown**. It's harmless and doesn't affect operation.

**Action:** Ignore this error. System works perfectly during execution.

---

### Issue 9: RViz Crashes on Startup

**Symptoms:**
```
[ERROR] [rviz2-5]: process has died [pid XXX, exit code -11]
```

**Possible causes:**

**A. X11 display not configured**
```bash
xhost +local:docker
# or
xhost +local:root
```

**B. Wrong RViz config path** (See Issue 3A)

**C. Conflicting ROS workspace**
```bash
# Clean rebuild
cd /workspace
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

### Issue 10: "No such file or directory" for Launch Files

**Symptoms:**
```
Package 'parol6_moveit_config' not found
# or
Launch file 'unified_bringup.launch.py' not found
```

**Cause:** Workspace not built or sourced

**Fix:**
```bash
cd /workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Now launch
ros2 launch parol6_driver real_robot_viz.launch.py
```

---

## 📡 Communication Issues

### Issue 11: ESP32 Shows "Invalid message format"

**Symptoms:**
ESP32 logs show:
```
W (1234) BENCHMARK: Invalid message format received
```

**Cause:** Driver sending wrong  format or sequence number missing

**Check what's being sent:**

Look at PC logs:
```bash
cat /workspace/logs/driver_commands_*.csv
```

**Expected command format:**
```
<0,0.5000,0.3000,-0.2000,0.1000,0.4000,-0.1000>
```

**If missing sequence number (first number):** Rebuild driver (See Issue 1)

---

### Issue 12: High Latency or Packet Loss

**Symptoms:**
```
Avg Latency: >100ms
Loss Rate: >5%
```

**Causes & Fixes:**

**A. USB cable issue**
- Try different USB cable
- Use USB 2.0 port (not 3.0)

**B. Baud rate mismatch**

Check both match:
```c
// ESP32: main/benchmark_main.c
.baud_rate = 115200,
```

```python
# PC: parol6_driver/parol6_driver/real_robot_driver.py
self.ser = serial.Serial('/dev/ttyUSB0', 115200)
```

**C. Serial buffer overflow**
```c
// ESP32: increase buffer
#define BUF_SIZE 2048  // Increase if needed
```

**D. CPU overload**
```bash
# Check CPU usage
top
# If Docker using >90%, reduce other processes
```

---

## 🗂️ Log & File Issues

### Issue 13: Empty CSV Log Files

**Symptoms:**
```
driver_commands_*.csv exists but only 177 bytes (header only)
```

**Cause:** Test was too short or no trajectory executed

**Expected size:**
- Quick test (1-2 seconds): 1-10 KB
- Full trajectory: 50-500 KB

**Fix:** Execute longer trajectory in RViz

**Note:** Analysis script filters out empty logs automatically

---

### Issue 14: Log Timestamps Show Wrong Time

**Symptoms:**
```
Log shows 22:56:57 but current time is 02:07:14
```

**Is this a problem?** **NO!**

**Explanation:**
- You're in timezone UTC+3
- Logs are correct
- Time difference is normal

**Example:**
- File: `2026-01-12T22:56:57` (10:56 PM yesterday)
- Now: `2026-01-13T02:07:14` (2:07 AM today)
- Difference: ~3h 11m (correct!)

---

## 🔍 Data Analysis Issues

### Issue 15: matplotlib vs pandas Version Conflict

**Symptoms:**
```
ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported
```

**Cause:** Pandas changed indexing behavior

**Fix:** Already fixed in `quick_log_analysis.py` using `.values`:
```python
# Use this:
ax.plot(x_axis, np.degrees(df[col].values), ...)

# Not this:
ax.plot(x_axis, np.degrees(df[col]), ...)  # Old way
```

---

## 🐳 Docker Issues

### Issue 16: Container Exists But Won't Start

**Symptoms:**
```
Docker: container already exists
Can't create new container
```

**Fix:**
```bash
# Stop and remove old container
docker stop parol6_dev
docker rm parol6_dev

# Restart
./start_real_robot.sh
```

---

### Issue 17: "out of memory" When Building

**Symptoms:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Cause:** Docker container has insufficient RAM

**Fix:**

Increase Docker memory:
```bash
# Docker Desktop: Settings → Resources → Memory
# Set to at least 4GB (8GB recommended)
```

Or build with fewer parallel jobs:
```bash
colcon build --symlink-install --parallel-workers 1
```

---

## 📊 Success Verification Checklist

After following guides, verify each item:

**Environment:**
- [ ] Docker container `parol6_dev` running
- [ ] Python environment fixed (`echo $PYTHON_EXECUTABLE` shows `/usr/bin/python3`)
- [ ] Workspace built without errors
- [ ] ESP32 connected (`ls /dev/ttyUSB0` shows device)

**ESP32:**
- [ ] Firmware flashed successfully
- [ ] Monitor shows "READY: ESP32_BENCHMARK_V2"
- [ ] Standalone test shows 0% packet loss
- [ ] Latency < 50ms

**ROS:**
- [ ] `start_real_robot.sh` launches without errors
- [ ] RViz opens and shows robot model
- [ ] Interactive markers visible (orange sphere)
- [ ] Driver logs show "Connected to Microcontroller"

**Full Pipeline:**
- [ ] Can drag interactive marker in RViz
- [ ] "Plan" generates trajectory visualization
- [ ] "Execute" sends commands to ESP32
- [ ] ESP32 monitor shows "SEQ:X J:[...]" messages
- [ ] CSV logs created in `/workspace/logs/`
- [ ] Analysis script runs without errors

**If all checked:** ✅ **System is working correctly!**

**If any unchecked:** See corresponding issue above or ask team.

---

## 🆘 Getting Help

**Before asking:**
1. Check this troubleshooting guide
2. Check the specific guide you're following
3. Try searching error message online

**When asking for help:**
- Specify which guide/step you're on
- Copy exact error message
- Show command you ran
- Include relevant logs

**Where to ask:**
- Team chat: Quick questions
- GitHub issues: Bug reports
- Documentation: Add FAQ if it helps others

---

**Last Updated:** January 2026  
**Based on:** Real issues encountered during development

**Found a new issue?** Document it here to help the next person!
