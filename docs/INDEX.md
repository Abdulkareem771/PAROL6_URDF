# PAROL6 Documentation Index

Welcome to the PAROL6 Robot documentation! This index will help you find the information you need.

## 📚 Documentation Files

### 1. **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete Technical Documentation
**Best for:** Comprehensive reference, API details, advanced topics

**Contents:**
- System requirements and installation
- Architecture overview
- Configuration file details
- Complete usage guide
- Programming interfaces (Python/C++)
- Troubleshooting guide
- Advanced topics
- API reference

**Start here if:** You want complete technical details

---

### 2. **[README.md](README.md)** - Quick Start Guide
**Best for:** Getting started quickly

**Contents:**
- Quick start instructions
- Docker setup
- Basic usage examples
- Common commands
- Testing procedures
- Next steps

**Start here if:** You want to get running fast

---

### 3. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Setup Completion Summary
**Best for:** Understanding what was configured

**Contents:**
- List of completed tasks
- File structure overview
- Configuration summary
- Test results
- What you can do now
- Next development steps

**Start here if:** You want to see what's been set up

---

### 4. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System Architecture
**Best for:** Understanding how everything works together

**Contents:**
- Component diagrams
- Data flow visualization
- ROS 2 topics/services/actions
- Launch file hierarchy
- Development workflow
- Quick reference commands

**Start here if:** You want to understand the system design

---

### 5. **[QUICKREF.sh](QUICKREF.sh)** - Quick Reference Card
**Best for:** Command-line reference

**Contents:**
- Workspace setup commands
- Launch commands
- Controller commands
- Monitoring commands
- Testing commands
- Docker commands
- Troubleshooting tips

**Start here if:** You need quick command reference

---

### 6. **[CONTAINER_ARCHITECTURE.md](CONTAINER_ARCHITECTURE.md)** - Docker Workflow Explained
**Best for:** Understanding why everything runs in Docker

**Contents:**
- Why Docker is used
- How host/container interaction works
- File synchronization explained
- Multi-terminal workflow
- Best practices
- Alternative approaches (and why not to use them)

**Start here if:** You're confused about Docker or want to run on host

---

## 🚀 Quick Navigation

### I want to...

#### **Get started immediately**
→ Run `./launch.sh` or see [README.md](README.md)

#### **Understand the system**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md)

#### **Learn the API**
→ See [DOCUMENTATION.md](DOCUMENTATION.md) Section 7

#### **Troubleshoot an issue**
→ See [DOCUMENTATION.md](DOCUMENTATION.md) Section 8

#### **See what's configured**
→ Read [SETUP_COMPLETE.md](SETUP_COMPLETE.md)

#### **Find a specific command**
→ Run `./QUICKREF.sh`

---

## 📖 Reading Path by Experience Level

### Beginner (New to ROS 2 / MoveIt)

1. **[README.md](README.md)** - Get familiar with basics
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand components
3. **[DOCUMENTATION.md](DOCUMENTATION.md)** Sections 1-6 - Learn usage
4. Practice with `./launch.sh`
5. **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 7 - Try programming

### Intermediate (Familiar with ROS 2)

1. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - See what's available
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand integration
3. **[DOCUMENTATION.md](DOCUMENTATION.md)** Sections 5-7 - Configuration & API
4. Start programming with examples
5. **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 9 - Advanced topics

### Advanced (Experienced with MoveIt)

1. **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Quick overview
2. Review configuration files directly
3. **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 9 - Advanced topics
4. **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 10 - API reference
5. Customize for your needs

---

## 🎯 Topic-Specific Guides

### Motion Planning
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 5.3 - OMPL configuration
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 9.2 - Tuning planners
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Planning pipeline

### Controllers
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 5.4 - Controller config
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 6.3 - Command line control
- **[README.md](README.md)** - Controller setup

### Kinematics
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 5.2 - IK solver
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 1 - Joint specifications
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Kinematics in pipeline

### Simulation
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 6.1 - Launch options
- **[README.md](README.md)** - Gazebo setup
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 8 - Troubleshooting

### Programming
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 7 - Complete API guide
- `parol6_moveit_config/scripts/example_controller.py` - Example code
- **[DOCUMENTATION.md](DOCUMENTATION.md)** Section 9 - Advanced programming

---

## 🛠️ Utility Scripts

### Interactive Tools
- **`launch.sh`** - Interactive launcher menu
- **`test_setup.sh`** - Automated testing
- **`QUICKREF.sh`** - Display reference card

### Example Code
- **`parol6_moveit_config/scripts/example_controller.py`** - Python API example

---

## 📁 Configuration Files Reference

### MoveIt Configuration
Located in `parol6_moveit_config/config/`:

- **`parol6.srdf`** - Planning groups, collision matrix
  - Documented in [DOCUMENTATION.md](DOCUMENTATION.md) Section 5.1

- **`kinematics.yaml`** - IK solver settings
  - Documented in [DOCUMENTATION.md](DOCUMENTATION.md) Section 5.2

- **`ompl_planning.yaml`** - Motion planners
  - Documented in [DOCUMENTATION.md](DOCUMENTATION.md) Section 5.3

- **`moveit_controllers.yaml`** - Controller manager
  - Documented in [DOCUMENTATION.md](DOCUMENTATION.md) Section 5.4

- **`joint_limits.yaml`** - Velocity/acceleration limits
  - Documented in [DOCUMENTATION.md](DOCUMENTATION.md) Section 5.5

### Robot Description
Located in `PAROL6/`:

- **`urdf/PAROL6.urdf`** - Complete robot description
  - Specifications in [DOCUMENTATION.md](DOCUMENTATION.md) Section 1

- **`config/ros2_controllers.yaml`** - Gazebo controllers
  - Documented in [README.md](README.md)

---

## 🔍 Search by Keyword

| Looking for... | See... |
|----------------|--------|
| Installation | [README.md](README.md) Section 2, [DOCUMENTATION.md](DOCUMENTATION.md) Section 3 |
| Docker | [README.md](README.md), [DOCUMENTATION.md](DOCUMENTATION.md) Section 3 |
| Launch files | [DOCUMENTATION.md](DOCUMENTATION.md) Section 6.1, [ARCHITECTURE.md](ARCHITECTURE.md) |
| RViz | [DOCUMENTATION.md](DOCUMENTATION.md) Section 6.2 |
| Python API | [DOCUMENTATION.md](DOCUMENTATION.md) Section 7.1 |
| C++ API | [DOCUMENTATION.md](DOCUMENTATION.md) Section 7.3 |
| Troubleshooting | [DOCUMENTATION.md](DOCUMENTATION.md) Section 8 |
| Joint limits | [DOCUMENTATION.md](DOCUMENTATION.md) Section 1, 5.5 |
| Collision checking | [DOCUMENTATION.md](DOCUMENTATION.md) Section 5.1, 9.3 |
| Named states | [DOCUMENTATION.md](DOCUMENTATION.md) Section 9.1 |
| Real hardware | [DOCUMENTATION.md](DOCUMENTATION.md) Section 9.5 |
| Performance | [DOCUMENTATION.md](DOCUMENTATION.md) Section 8.3 |
| Commands | [QUICKREF.sh](QUICKREF.sh) |

---

## 📞 Getting Help

### Check Documentation
1. Search this index for your topic
2. Read the relevant section
3. Try the examples

### Run Tests
```bash
./test_setup.sh
```

### View Quick Reference
```bash
./QUICKREF.sh
```

### Check Logs
```bash
# Inside container
ros2 run rqt_console rqt_console
```

### Community Resources
- ROS Discourse: https://discourse.ros.org/
- MoveIt Discord: https://discord.gg/moveit

---

## 🎓 Learning Path

### Week 1: Basics
- [ ] Read [README.md](README.md)
- [ ] Run `./launch.sh` and try MoveIt demo
- [ ] Practice moving robot in RViz
- [ ] Read [ARCHITECTURE.md](ARCHITECTURE.md)

### Week 2: Configuration
- [ ] Read [DOCUMENTATION.md](DOCUMENTATION.md) Sections 1-6
- [ ] Understand configuration files
- [ ] Try Gazebo simulation
- [ ] Experiment with different planners

### Week 3: Programming
- [ ] Read [DOCUMENTATION.md](DOCUMENTATION.md) Section 7
- [ ] Run `example_controller.py`
- [ ] Write simple motion script
- [ ] Try pose-based goals

### Week 4: Advanced
- [ ] Read [DOCUMENTATION.md](DOCUMENTATION.md) Section 9
- [ ] Add custom named states
- [ ] Tune motion planners
- [ ] Add collision objects

---

## ✅ Quick Checklist

Before starting development:
- [ ] Docker container running (`docker ps`)
- [ ] Workspace built (`colcon build`)
- [ ] Environment sourced (`source install/setup.bash`)
- [ ] Tests passing (`./test_setup.sh`)
- [ ] Can launch demo (`ros2 launch parol6_moveit_config demo.launch.py`)

---

## 📊 Documentation Statistics

- **Total Documentation Files:** 5 main files
- **Total Pages:** ~50 equivalent pages
- **Code Examples:** 20+ examples
- **Configuration Files:** 6 MoveIt configs
- **Launch Files:** 4 launch files
- **Helper Scripts:** 3 utility scripts

---

**Last Updated:** 2025-11-27  
**Version:** 1.0  
**Maintained by:** AntiGravity AI
