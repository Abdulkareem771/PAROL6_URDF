# Camera vs Bag Replay - Important Conflict Information

## ⚠️ Critical: Camera and Bag Cannot Run Simultaneously

### The Problem
If you have:
- ✅ Physical camera connected AND `kinect2_bridge` running
- ✅ Bag replay running (`ros2 bag play --loop`)

**Both publish to the SAME topics:**
```
/kinect2/qhd/image_color_rect
/kinect2/qhd/image_depth_rect
/kinect2/qhd/camera_info
/tf
/tf_static
```

### What Happens (Bad!)
- 🔴 **Topic collision** - Two sources publishing to same topic
- 🔴 **Timestamp conflicts** - Messages with different timestamps
- 🔴 **TF tree warnings** - Conflicting transform publishers
- 🔴 **Unpredictable behavior** - Subscribers receive mixed data
- 🔴 **Vision pipeline errors** - Cannot determine which source is correct

---

## ✅ Correct Usage

### Mode 1: Live Camera Development
```bash
# Terminal 1: Launch camera
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# Terminal 2: Launch vision pipeline
ros2 launch parol6_vision camera_setup.launch.py
```

**When to use:** Testing with real hardware, calibration, live capture

---

### Mode 2: Bag Replay Development (No Camera)
```bash
# Terminal 1: Replay bag (camera NOT connected/running)
unset ROS_DOMAIN_ID
ros2 bag play test_data/kinect_snapshot_YYYYMMDD_HHMMSS --loop

# Terminal 2: Launch vision pipeline
ros2 launch parol6_vision camera_setup.launch.py
```

**When to use:** Teammates without camera, deterministic testing, debugging

---

## 🔍 How to Detect Conflict

### Check what's publishing:
```bash
ros2 topic info /kinect2/qhd/image_color_rect
```

**Good output (single publisher):**
```
Publisher count: 1
```

**Bad output (conflict!):**
```
Publisher count: 2
  Node name: /kinect2_bridge
  Node name: /rosbag2_player
```

---

## 🛡️ Prevention Checklist

Before launching vision pipeline, verify:

**For live camera mode:**
- [ ] Camera physically connected
- [ ] `kinect2_bridge` is running
- [ ] **NO bag replay running**
- [ ] Topics show 1 publisher (kinect2_bridge)

**For bag replay mode:**
- [ ] Camera **NOT** connected OR kinect2_bridge **NOT** running
- [ ] Bag replay is running (`ros2 bag play --loop`)
- [ ] Topics show 1 publisher (rosbag2_player)

---

## 🚨 What To Do If Conflict Occurs

**Symptoms:**
- RViz shows flickering images
- Vision pipeline produces inconsistent results
- TF warnings in terminal
- Timestamp errors

**Fix:**
```bash
# 1. Stop everything
Ctrl+C in all terminals

# 2. Choose ONE mode:

# Option A: Use live camera
ros2 launch kinect2_bridge kinect2_bridge.launch.py

# Option B: Use bag replay
ros2 bag play test_data/kinect_snapshot_* --loop

# 3. Launch vision pipeline
ros2 launch parol6_vision camera_setup.launch.py
```

---

## 💡 Future Enhancement (Not Yet Implemented)

We could add a launch argument to automatically handle this:

```bash
# Future possibility:
ros2 launch parol6_vision camera_setup.launch.py use_bag_replay:=true bag_path:=...
```

This would:
- Automatically start bag replay
- Skip kinect2_bridge launch
- Ensure no conflicts

**For now:** Manually ensure only ONE source is active.

---

## 📊 Quick Reference Table

| Scenario | Camera Connected? | kinect2_bridge Running? | Bag Replay Running? | Result |
|----------|-------------------|-------------------------|---------------------|--------|
| Live testing | ✅ Yes | ✅ Yes | ❌ No | ✅ **Good** |
| Bag replay | ❌ No | ❌ No | ✅ Yes | ✅ **Good** |
| **CONFLICT** | ✅ Yes | ✅ Yes | ✅ Yes | 🔴 **BAD** |
| Idle | ✅ Yes | ❌ No | ❌ No | ⚠️ OK (nothing running) |

---

## 🎓 For Your Thesis

When documenting experiments, clearly state which mode was used:

**Live camera experiments:**
> "Experiments were performed using live Kinect v2 sensor data streaming at 30 Hz..."

**Bag replay experiments:**
> "Experiments used frozen ROS bag dataset `kinect_snapshot_20260124_024153` for deterministic, reproducible testing..."

This prevents ambiguity in methodology.
