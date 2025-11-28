# 🎯 PAROL6 Dual Simulator Setup

## ✅ **What's Been Created**

You now have **flexible simulator support** - choose what works best!

---

## 📁 **File Structure**

### **URDF Files (3 variants):**
```
PAROL6/urdf/
├── PAROL6.urdf           # Original (currently Ignition)
├── PAROL6_gazebo.urdf    # Gazebo Classic variant
└── PAROL6_ignition.urdf  # Ignition Gazebo variant
```

### **Launch Files:**
```
PAROL6/launch/
├── gazebo.launch.py          # Original Gazebo Classic
├── gazebo_classic.launch.py  # New Gazebo Classic (uses PAROL6_gazebo.urdf)
└── ignition.launch.py        # Ignition Gazebo (uses PAROL6_ignition.urdf)
```

### **Launcher Scripts:**
```
./start_gazebo_manual.sh   # Interactive shell - run Gazebo manually
./start_ign_simple.sh      # Ignition Gazebo launcher
./start.sh                 # Original Gazebo launcher
```

---

## 🚀 **How to Use**

### **Option 1: Manual Gazebo

```bash
./start_gazebo_manual.sh
```

This will:
1. Start container
2. Build workspace
3. **Open interactive shell**
4. You manually run:
   ```bash
   source /opt/ros/humble/setup.bash
   source /workspace/install/setup.bash
   ros2 launch parol6 gazebo_classic.launch.py
   ```

**Why this works:** Has problems

---

### **Option 2: Ignition Gazebo**(Recommended)

```bash
./start_ign_simple.sh
```
then in a seperate terminal run: 
./add_moveit.sh

**Status:** Working successfully
---

## 🔧 **Key Differences**

### **Gazebo Classic URDF (`PAROL6_gazebo.urdf`):**
```xml
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  ...
</ros2_control>

<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>package://parol6/config/ros2_controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### **Ignition URDF (`PAROL6_ignition.urdf`):**
```xml
<ros2_control name="IgnitionSystem" type="system">
  <hardware>
    <plugin>ign_ros2_control/IgnitionSystem</plugin>
  </hardware>
  ...
</ros2_control>

<gazebo>
  <plugin filename="ign_ros2_control-system" name="ign_ros2_control::IgnitionROS2ControlPlugin">
    <parameters>package://parol6/config/ros2_controllers.yaml</parameters>
    <robot_param>robot_description</robot_param>
    <robot_param_node>robot_state_publisher</robot_param_node>
  </plugin>
</gazebo>
```

---

## 📋 **Next Steps**

### **Immediate: Test Manual Gazebo**(Not Recommended)

1. Run:
   ```bash
   ./start_gazebo_manual.sh
   ```

2. Inside container, run:
   ```bash
   source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch parol6 gazebo_classic.launch.py
   ```

3. **If Gazebo opens successfully**, open a NEW terminal and run:
   ```bash
   ./add_moveit.sh
   ```

4. **Test motion planning in RViz!**

---

### **If Manual Works:**

We can then automate it by updating `start.sh` to use `gazebo_classic.launch.py`.

---

### **If You Want to Fix Ignition:**

The issue is `ign_ros2_control` plugin not loading. Possible fixes:
1. Check if `ros-humble-ign-ros2-control` is installed
2. Verify plugin path
3. Check Ignition logs for plugin errors

---

## 🎯 **Recommended Path Forward**

1. ✅ **Try ignition Gazebo** (most likely to work)
2. ✅ **If it works, use it!**
3. ⏭️ **Optionally debug classic later**

The goal is to **get you working ASAP**, not to make everything perfect.

---

## 💡 **Why This Approach?**

- **Flexibility:** Switch simulators without editing files
- **Pragmatic:** Use what works (manual Gazebo)
- **Future-proof:** Can fix Ignition later
- **No more editing:** Just run different scripts

---

**Try `./start_gazebo_manual.sh` now!** 🚀
