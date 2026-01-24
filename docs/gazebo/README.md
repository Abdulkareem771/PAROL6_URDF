# Gazebo Documentation

This directory contains all Gazebo simulation documentation for the PAROL6 project.

---

## 📚 Documentation Files

### [SIMPLE_LAUNCH.md](./SIMPLE_LAUNCH.md) ⭐⭐⭐
**RECOMMENDED START - Simple independent launches**

**Philosophy:** No combined scripts. No automation complexity.

Covers:
- Vision RViz (camera + detection)
- MoveIt RViz (motion planning)  
- Gazebo (simulation)
- Each runs independently in separate terminals
- ROS bag strategy (frozen datasets)
- Digital twin progressive building

**For:** Everyone - this is the cleanest approach

---

### [COMPLETE_SETUP_GUIDE.md](./COMPLETE_SETUP_GUIDE.md)
**Complete system overview for teammates**

Covers:
- System architecture (Gazebo + MoveIt RViz + Vision RViz)
- How all three visualization tools work together
- Step-by-step launch instructions for each tool
- Typical workflows (first launch, thesis validation, real robot)
- Complete troubleshooting section
- Tips for using Antigravity assistant

**For:** New teammates, complete system understanding

---

### [GAZEBO_GUIDE.md](./GAZEBO_GUIDE.md)
**Detailed Gazebo reference guide**

Covers:
- Gazebo vs. Ignition comparison
- Usage scenarios (standalone, with MoveIt, with vision pipeline)
- Troubleshooting common issues
- Integration with thesis validation plan
- FAQ and verification checklists

**For:** Both new and experienced users

---

### [QUICK_START.md](./QUICK_START.md)
**Quick reference card**

Provides:
- One-line launch commands
- Common workflows
- Quick troubleshooting fixes

**For:** Daily usage, quick lookup

---

## 🎯 When to Use Gazebo

### ✅ Use Gazebo For:
- **Visual Validation:** See robot movement in 3D before real hardware
- **Thesis Documentation:** Record videos/screenshots of simulated execution
- **Path Visualization:** See welding paths executed on virtual robot
- **Safety Testing:** Test potentially dangerous motions in simulation first

### ❌ Don't Need Gazebo For:
- **Vision Pipeline Development:** Use `camera_setup.launch.py` + RViz
- **Motion Planning:** Use `demo.launch.py` (MoveIt + RViz)
- **Real Robot Control:** Use `start_real_robot.sh`

---

## 🚀 Quick Navigation

**Want the simplest approach?** ⭐  
→ **[SIMPLE_LAUNCH.md](./SIMPLE_LAUNCH.md)** - 3 independent launches, no scripts

**First time with the PAROL6 system?**  
→ [COMPLETE_SETUP_GUIDE.md](./COMPLETE_SETUP_GUIDE.md)  
  (Explains entire system: Gazebo + MoveIt + Vision)

**First time using Gazebo specifically?**  
→ See [GAZEBO_GUIDE.md](./GAZEBO_GUIDE.md) § Quick Start

**Just need launch commands?**  
→ See [QUICK_START.md](./QUICK_START.md)

**Gazebo hanging/not working?**  
→ Check [GAZEBO_GUIDE.md](./GAZEBO_GUIDE.md) § Troubleshooting

**Working on thesis validation?**  
→ See [GAZEBO_GUIDE.md](./GAZEBO_GUIDE.md) § Integration with Thesis Validation

---

## 🔗 Related Documentation

- **Vision Pipeline:** `parol6_vision/docs/README.md`
- **ROS Bags:** `parol6_vision/docs/rosbag/`
- **MoveIt Setup:** `docs/TEAMMATE_COMPLETE_GUIDE.md`
- **Real Robot:** `docs/REAL_ROBOT_INTEGRATION.md`
- **System Architecture:** `docs/ROS_SYSTEM_ARCHITECTURE.md`

---

## 📝 Available Launch Files

Located in `PAROL6/launch/`:

| File | Description | Use When |
|------|-------------|----------|
| `ignition.launch.py` | Ignition Gazebo (Modern) | **Default choice** - Best compatibility |
| `gazebo.launch.py` | Standard Gazebo (Classic) | Legacy/compatibility only |
| `gazebo_classic.launch.py` | Legacy Gazebo | Old systems only |

**Recommendation:** Use `ignition.launch.py` for best performance and compatibility.

---

**Last Updated:** 2026-01-24  
**Maintainer:** PAROL6 Vision Team
