# PAROL6 Robot Setup Guide for Team Members

This guide will help you set up the PAROL6 robot simulation on your machine.

---

## 📋 Prerequisites

1. **Ubuntu 22.04 LTS** (or WSL2 with Ubuntu 22.04)
2. **Docker** installed
3. **Git** installed

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Required Software

```bash
# Update system
sudo apt update

# Install Docker
sudo apt install docker.io -y
sudo usermod -aG docker $USER
newgrp docker

# Install Git (if not already installed)
sudo apt install git -y

# Install X11 dependencies for GUI
sudo apt install x11-xserver-utils -y
```

### Step 2: Clone the Repository

```bash
# Clone the project
cd ~/Desktop
git clone https://github.com/YOUR_USERNAME/PAROL6_URDF.git
cd PAROL6_URDF
```

### Step 3: Load the Docker Image

You should have received the `parol6-ultimate-with-servo.tar.gz` file. Follow these steps:

```bash
# Navigate to where you downloaded the image
cd ~/Downloads  # or wherever you saved the file

# Load the Docker image
docker load < parol6-ultimate-with-servo.tar.gz

# Verify the image is loaded
docker images | grep parol6
```

**Expected output:**
```
parol6-ultimate    latest    ...    ...    8.69GB
```

**⚠️ IMPORTANT:** If the image name is different (not `parol6-ultimate:latest`), you need to retag it:

```bash
# Check what name it has
docker images

# If it's something else, retag it (replace SOURCE_NAME with actual name)
docker tag SOURCE_NAME:latest parol6-ultimate:latest

# Verify
docker images | grep parol6-ultimate
```

---

## ✅ Test the Setup

### Start the Simulation

```bash
cd ~/Desktop/PAROL6_URDF
./start_ignition.sh
```

You should see:
1. Terminal output showing the build process
2. Ignition Gazebo window with the robot

### Add MoveIt (Optional)

Open a **new terminal** and run:

```bash
cd ~/Desktop/PAROL6_URDF
./add_moveit.sh
```

You should see RViz open with the robot and motion planning interface.

---

## 🐛 Troubleshooting

### Problem: "X11 connection rejected"
```bash
xhost +local:docker
./start_ignition.sh
```

### Problem: "Permission denied" on scripts
```bash
chmod +x *.sh
./start_ignition.sh
```

### Problem: Docker image size too large
The image is ~8.5GB. Make sure you have at least 15GB free space.

### Problem: "Cannot connect to Docker daemon"
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

---

## 📂 Project Structure

```
PAROL6_URDF/
├── start_ignition.sh      ← Start simulation (run this first)
├── add_moveit.sh          ← Add motion planning (run after ignition)
├── README.md              ← Main project documentation
├── docs/                  ← Additional guides
├── scripts/               ← Helper scripts
├── PAROL6/                ← Robot description files
└── parol6_moveit_config/  ← MoveIt configuration
```

---

## 🎮 Available Branches

```bash
# Switch to different control modes
git checkout main              # Clean base version
git checkout xbox-controller   # Xbox controller support
git checkout joystick-control  # Generic joystick
git checkout mobile-ros        # Mobile app control
```

---

## 🆘 Need Help?

1. Check the `docs/` folder for detailed guides
2. Run `./scripts/utils/status.sh` to see what's running
3. Ask Kareem or check the project README

---

## ⚙️ Environment Details

- **Python:** 3.10.12
- **OS:** Ubuntu 22.04.5 LTS
- **ROS:** ROS 2 Humble
- **Docker Image:** parol6-ultimate:latest

---

## 🎯 Next Steps

After setup is complete:
1. Explore the simulation in Ignition Gazebo
2. Try planning motions in RViz with MoveIt
3. Check different branches for advanced features
4. Read `docs/PROJECT_EXPLANATION.md` for architecture details
