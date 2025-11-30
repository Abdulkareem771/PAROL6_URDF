# MoveIt Servo + Xbox Controller Setup Guide

## 🎯 What This Does

This setup integrates **MoveIt Servo** with your PAROL6 robot, allowing you to control it using an Xbox controller with:
- **Collision avoidance**
- **Singularity handling**
- **Smooth, professional-grade motion**
- **Real-time teleoperation**

## 🚀 Installation (ONE-TIME SETUP)

### Step 1: Rebuild Docker Image with MoveIt Servo

This will permanently add MoveIt Servo and Xbox controller support to your Docker image:

```bash
cd /home/kareem/Desktop/PAROL6_URDF
./rebuild_image.sh
```

**This will take 5-10 minutes**. The packages will be permanently installed in the image, so:
- ✅ You never have to download them again
- ✅ Your colleagues can use the same image
- ✅ Everything persists across container restarts

### Step 2: Save and Share the Image (Optional)

After rebuilding, you can save the image to share with colleagues:

```bash
# Save image to a file
docker save parol6-ultimate:latest | gzip > parol6-ultimate-with-servo.tar.gz

# Share the file with colleagues, who can then load it:
docker load < parol6-ultimate-with-servo.tar.gz
```

Alternatively, push to Docker Hub:

```bash
docker tag parol6-ultimate:latest yourdockerhub/parol6-ultimate:latest
docker push yourdockerhub/parol6-ultimate:latest
```

## 🎮 Usage

### Step 1: Start the Simulation

```bash
./start_ignition.sh
```

Wait for Ignition Gazebo to fully load and the robot to spawn.

### Step 2: Launch Xbox Controller with MoveIt Servo

Open a **new terminal** and run:

```bash
cd /home/kareem/Desktop/PAROL6_URDF
./start_xbox_servo.sh
```

### Step 3: Control the Robot

**Xbox Controller Mapping:**

| Control | Function |
|---------|----------|
| **Left Stick** | X/Y linear motion (forward/back, left/right) |
| **Right Stick (Vertical)** | Z linear motion (up/down) |
| **Right Stick (Horizontal)** | Yaw rotation |
| **D-Pad (Vertical)** | Pitch rotation |
| **L2/R2 Triggers** | Roll rotation |

## 🏗️ Architecture

```
Xbox Controller
    ↓
joy_node (reads controller input)
    ↓
xbox_to_servo.py (converts to Twist messages)
    ↓
MoveIt Servo (collision-aware motion planning)
    ↓
ros2_control (sends commands to robot)
    ↓
Robot Moves!
```

## 📦 What Was Added

### Docker Image Changes
- `ros-humble-moveit-servo` - MoveIt Servo package
- `ros-humble-joy` - Joystick/controller input support
- `ros-humble-joint-state-publisher` - Better robot state handling

### New Files
- `parol6_moveit_config/config/parol6_servo.yaml` - Servo configuration
- `parol6_moveit_config/launch/servo.launch.py` - Servo launcher
- `parol6_moveit_config/launch/servo_with_joy.launch.py` - Complete system launcher
- `parol6_moveit_config/scripts/xbox_to_servo.py` - Xbox to Servo bridge
- `start_xbox_servo.sh` - Convenient start script

### Updated Files
- `Dockerfile` - Added MoveIt Servo and Joy packages
- `parol6_moveit_config/package.xml` - Added dependencies
- `parol6_moveit_config/CMakeLists.txt` - Install scripts directory

## 🐛 Troubleshooting

### "Package 'moveit_servo' not found"
You need to rebuild the Docker image:
```bash
./rebuild_image.sh
```

### Xbox controller not detected
1. Check if controller is connected: `ls /dev/input/js*`
2. Test with: `jstest /dev/input/js0`
3. Make sure `--device /dev/input` is mounted in docker run command

### Robot doesn't move
1. Check that Ignition Gazebo is fully loaded
2. Verify controllers are loaded: `ros2 control list_controllers`
3. Check servo is receiving commands: `ros2 topic echo /servo_node/status`

### "Permission denied" for serial/USB
The container runs with `--privileged` and `-v /dev:/dev`, so this shouldn't happen. If it does:
```bash
sudo chmod 666 /dev/input/js0
```

## 🎓 Understanding the Code

### xbox_to_servo.py
This bridge node:
1. Subscribes to `/joy` topic (Xbox controller data)
2. Maps joystick axes to Cartesian velocities
3. Publishes `TwistStamped` messages to `/servo_node/delta_twist_cmds`
4. Uses scaling factors (0.1 for linear, 0.2 for angular) for smooth motion

### MoveIt Servo Configuration
See `parol6_moveit_config/config/parol6_servo.yaml` for:
- Update frequency (50 Hz)
- Velocity limits
- Collision checking settings
- Singularity thresholds

## 📝 Next Steps

- [ ] Tune velocity scaling in `xbox_to_servo.py` for better response
- [ ] Add button mappings for gripper control
- [ ] Configure collision objects in the scene
- [ ] Add visual feedback in RViz for servo status

## 🏭 Production Ready

This setup follows industrial best practices:
- ✅ All dependencies in Docker image (reproducible)
- ✅ Proper ROS 2 package structure
- ✅ Collision avoidance enabled
- ✅ Configurable parameters via YAML
- ✅ Clean separation of concerns (joy → bridge → servo)

---

**Questions?** Check the MoveIt Servo documentation:
https://moveit.picknik.ai/humble/doc/examples/realtime_servo/realtime_servo_tutorial.html
