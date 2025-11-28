# PAROL6 Project - Container Architecture

## 🐳 Important: This Project Runs Inside Docker

**This is intentional and the recommended way to work with this project.**

---

## Why Docker?

### ✅ Advantages

1. **Consistent Environment**
   - Everyone has identical ROS 2 Humble setup
   - No "works on my machine" problems
   - Locked dependency versions

2. **Clean Host System**
   - No ROS 2 installation needed on your computer
   - No conflicting packages
   - Easy to remove (just delete container)

3. **Easy Onboarding**
   - New team members: Install Docker → Done
   - No complex ROS 2 installation steps
   - Works on Ubuntu, Windows (WSL2), macOS

4. **Portability**
   - Same setup on laptop, workstation, CI/CD
   - Easy to share with collaborators
   - Version controlled via Dockerfile

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR HOST MACHINE                     │
│                                                          │
│  /home/kareem/Desktop/PAROL6_URDF/  ← Your files here   │
│  (You edit files here with your favorite editor)        │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         DOCKER CONTAINER                        │    │
│  │                                                 │    │
│  │  /workspace/  ← Mounted from host              │    │
│  │  (Same files, automatically synced)            │    │
│  │                                                 │    │
│  │  ROS 2 Humble ✓                                │    │
│  │  MoveIt 2 ✓                                    │    │
│  │  Gazebo ✓                                      │    │
│  │  All dependencies ✓                            │    │
│  │                                                 │    │
│  │  GUI apps → Display on host via X11           │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Your Display ← Shows RViz, Gazebo                      │
└─────────────────────────────────────────────────────────┘
```

---

## What Runs Where?

### On Your Host Machine (Outside Container):
- ✅ **Code editing** - Use VS Code, vim, etc.
- ✅ **File management** - Browse, copy, organize files
- ✅ **Git operations** - Commit, push, pull
- ✅ **Documentation** - Read markdown files
- ✅ **Display** - See RViz and Gazebo windows

### Inside Docker Container:
- ✅ **ROS 2 commands** - `ros2 launch`, `ros2 topic`, etc.
- ✅ **Building** - `colcon build`
- ✅ **Running** - Gazebo, RViz, MoveIt
- ✅ **Testing** - All ROS 2 tests
- ✅ **Python/C++ execution** - Run your robot code

---

## Typical Workflow

### 1. Edit Files on Host
```bash
# On your host machine
cd /home/kareem/Desktop/PAROL6_URDF
code .  # Or use any editor

# Edit files:
# - PAROL6/urdf/PAROL6.urdf
# - parol6_moveit_config/config/*.yaml
# - Your custom scripts
```

### 2. Run Commands in Container
```bash
# In container terminal
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Build (if you changed code)
colcon build --symlink-install

# Launch
ros2 launch parol6 gazebo.launch.py
```

### 3. View Results on Host
- RViz window appears on your screen
- Gazebo window appears on your screen
- All GUI apps display normally

---

## Working with Multiple Terminals

### Terminal 1: Main Container
```bash
# Start container (interactive)
docker run -it --rm \
  --name parol6_dev \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/home/kareem/Desktop/PAROL6_URDF:/workspace" \
  parol6-robot:latest

# Inside container
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
ros2 launch parol6 gazebo.launch.py
```

### Terminal 2: Additional Commands
```bash
# From host, enter running container
docker exec -it parol6_dev bash

# Inside this new shell
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Run additional commands
ros2 topic list
ros2 control list_controllers
```

### Terminal 3: Code Editing (Host)
```bash
# On host machine
cd /home/kareem/Desktop/PAROL6_URDF
code .
# Edit files - changes are immediately visible in container
```

---

## File Synchronization

**Important:** Files are **automatically synchronized** between host and container!

```bash
# Edit on host:
nano /home/kareem/Desktop/PAROL6_URDF/PAROL6/urdf/PAROL6.urdf

# Immediately available in container:
# /workspace/PAROL6/urdf/PAROL6.urdf

# No need to copy or sync manually!
```

---

## Common Misconceptions

### ❌ "I need to install ROS 2 on my host"
**No!** ROS 2 only needs to be in the container. Your host just needs Docker.

### ❌ "I can't edit files easily"
**Wrong!** Edit files on your host with any editor. Changes sync automatically.

### ❌ "Docker is slow"
**Not true!** The volume mount is fast. GUI forwarding via X11 is native speed.

### ❌ "I need to rebuild the container when I change code"
**No!** Only rebuild when you change the Dockerfile. Code changes just need `colcon build`.

---

## Alternative: Install ROS 2 on Host (Not Recommended)

If you **really** want to run on host (not recommended):

### Why Not Recommended:
- ❌ Complex installation process
- ❌ Can conflict with other software
- ❌ Different setup for each team member
- ❌ Harder to maintain
- ❌ Version drift over time

### If You Insist:
```bash
# Install ROS 2 Humble on Ubuntu 22.04
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-moveit
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-ros2-controllers
sudo apt install ros-humble-gazebo-ros2-control

# Then build workspace
cd /home/kareem/Desktop/PAROL6_URDF
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

**But seriously, just use Docker. It's easier.**

---

## Best Practices

### ✅ DO:
- Keep container running while working
- Use `docker exec` for additional terminals
- Edit files on host
- Run ROS commands in container
- Use `./launch.sh` helper script

### ❌ DON'T:
- Try to run ROS commands on host
- Install ROS 2 on host (unless you have a good reason)
- Copy files between host and container (they're already synced!)
- Rebuild container for code changes (just `colcon build`)

---

## Quick Reference

### Start Container
```bash
cd /home/kareem/Desktop/PAROL6_URDF
./launch.sh  # Interactive menu
# Or manually:
docker run -it --rm --name parol6_dev \
  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/home/kareem/Desktop/PAROL6_URDF:/workspace" \
  parol6-robot:latest
```

### Enter Running Container
```bash
docker exec -it parol6_dev bash
```

### Check Container Status
```bash
docker ps  # List running containers
docker ps -a  # List all containers
```

### Stop Container
```bash
docker stop parol6_dev
# Or Ctrl+C in the container terminal
```

---

## Team Collaboration

### For New Team Members:

1. **Install Docker** (one-time setup)
   ```bash
   # Ubuntu
   sudo apt install docker.io
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Clone/Copy Project**
   ```bash
   # Get the PAROL6_URDF directory
   # (via git, USB, network share, etc.)
   ```

3. **Start Working**
   ```bash
   cd PAROL6_URDF
   ./launch.sh
   ```

**That's it!** No ROS 2 installation needed.

---

## Summary

**The container-based approach is the correct and recommended way to use this project.**

- ✅ Edit files on host
- ✅ Run ROS commands in container  
- ✅ View GUI on host
- ✅ Files automatically synced
- ✅ Clean, consistent, portable

**Don't fight the container - embrace it!** 🐳

---

**Questions?** This is standard practice for ROS 2 development. Many professional teams use this approach.
