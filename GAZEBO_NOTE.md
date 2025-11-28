# PAROL6 - Important Note About Simulators

## ⚠️ Your System Has Ignition Gazebo (Not Gazebo Classic)

Your Docker container has **Ignition Gazebo 6 (Fortress)** installed, not Gazebo Classic.

### What This Means:

**Ignition Gazebo** (now called **Gazebo**) is the newer simulator:
- ✅ Better performance
- ✅ Modern architecture  
- ✅ Better physics
- ❌ Different API than Gazebo Classic
- ❌ Requires different URDF plugins

**Gazebo Classic** (old version):
- ❌ Not installed in your container
- ❌ Launch files won't work

---

## 🚀 Current Status:

The PAROL6 URDF was originally configured for **Gazebo Classic** with `gazebo_ros2_control`, but your container only has **Ignition Gazebo**.

### Options:

#### Option 1: Use Ignition Gazebo (Recommended)
- ✅ Already installed
- ✅ Modern and supported
- ❌ Requires URDF modifications
- ❌ Uses `ign_ros2_control` instead

#### Option 2: Install Gazebo Classic
- ✅ URDF already configured
- ✅ Launch files ready
- ❌ Need to install in container
- ❌ Older technology

---

## 📝 What Needs to Change for Ignition:

### 1. URDF Changes:
Replace `gazebo_ros2_control` plugin with `ign_ros2_control`:

```xml
<!-- OLD (Gazebo Classic) -->
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
</ros2_control>

<!-- NEW (Ignition Gazebo) -->
<ros2_control name="IgnitionSystem" type="system">
  <hardware>
    <plugin>ign_ros2_control/IgnitionSystem</plugin>
  </hardware>
</ros2_control>
```

### 2. Launch File:
Use `ros_ign_gazebo` instead of `gazebo_ros`:
- Created: `ignition.launch.py` (new file)
- Uses: `ros_ign_gazebo` and `ros_ign_bridge`

---

## 🔧 Quick Fix (Install Gazebo Classic):

If you want to use the existing setup without changes:

```bash
# Inside container
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

Then the original `gazebo.launch.py` will work.

---

## 🎯 Recommendation:

**For now: Install Gazebo Classic** (easier, everything already configured)

**Future: Migrate to Ignition** (better long-term, but requires URDF updates)

---

## 📚 More Info:

- Ignition Gazebo: https://gazebosim.org/
- ROS 2 + Ignition: https://github.com/gazebosim/ros_gz
- Migration Guide: https://gazebosim.org/docs/fortress/migrating_gazebo_classic

---

**Current files:**
- `gazebo.launch.py` - For Gazebo Classic (won't work without install)
- `ignition.launch.py` - For Ignition Gazebo (NEW, needs URDF changes)
