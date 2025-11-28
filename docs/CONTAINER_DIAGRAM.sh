#!/bin/bash
# Visual explanation of container architecture

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════╗
║              PAROL6 CONTAINER ARCHITECTURE                           ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR HOST COMPUTER                            │
│  (Ubuntu, Windows WSL2, macOS)                                      │
│                                                                      │
│  📁 /home/kareem/Desktop/PAROL6_URDF/                               │
│     ├── PAROL6/                    ← Edit here with VS Code        │
│     ├── parol6_moveit_config/      ← Files sync automatically      │
│     ├── *.md (documentation)                                        │
│     └── *.sh (helper scripts)                                       │
│                                                                      │
│  💻 Your Editor (VS Code, vim, etc.)                                │
│  🖥️  Your Display (shows RViz, Gazebo)                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    DOCKER CONTAINER                             │ │
│  │  Name: parol6_dev                                              │ │
│  │                                                                 │ │
│  │  📁 /workspace/  ← SAME FILES (mounted from host)              │ │
│  │     ├── PAROL6/                                                │ │
│  │     ├── parol6_moveit_config/                                  │ │
│  │     ├── build/                                                 │ │
│  │     └── install/                                               │ │
│  │                                                                 │ │
│  │  🔧 Installed Software:                                         │ │
│  │     ├── ROS 2 Humble                                           │ │
│  │     ├── MoveIt 2                                               │ │
│  │     ├── Gazebo Classic                                         │ │
│  │     ├── ros2_control                                           │ │
│  │     └── All dependencies                                       │ │
│  │                                                                 │ │
│  │  ▶️  Run commands here:                                         │ │
│  │     • ros2 launch parol6 gazebo.launch.py                      │ │
│  │     • colcon build                                             │ │
│  │     • python3 your_script.py                                   │ │
│  │                                                                 │ │
│  │  🖼️  GUI → X11 forwarding → Your display                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════

WORKFLOW:

1️⃣  EDIT FILES (on host)
   cd /home/kareem/Desktop/PAROL6_URDF
   code .
   # Edit PAROL6/urdf/PAROL6.urdf
   # Changes are INSTANTLY visible in container!

2️⃣  RUN COMMANDS (in container)
   docker exec -it parol6_dev bash
   source /opt/ros/humble/setup.bash
   source /workspace/install/setup.bash
   ros2 launch parol6 gazebo.launch.py

3️⃣  VIEW RESULTS (on host)
   # Gazebo window appears on your screen
   # RViz window appears on your screen
   # All GUI apps display normally

═══════════════════════════════════════════════════════════════════════

FILE SYNCHRONIZATION:

Host:      /home/kareem/Desktop/PAROL6_URDF/PAROL6/urdf/PAROL6.urdf
           ↕️  (Automatically synced - no copying needed!)
Container: /workspace/PAROL6/urdf/PAROL6.urdf

Edit on host → Immediately available in container
Build in container → Outputs visible on host

═══════════════════════════════════════════════════════════════════════

MULTIPLE TERMINALS:

Terminal 1 (Container - Main):
  $ docker run -it --rm --name parol6_dev ... parol6-robot:latest
  $ source /opt/ros/humble/setup.bash
  $ source /workspace/install/setup.bash
  $ ros2 launch parol6 gazebo.launch.py

Terminal 2 (Container - Additional):
  $ docker exec -it parol6_dev bash
  $ source /opt/ros/humble/setup.bash
  $ source /workspace/install/setup.bash
  $ ros2 topic list

Terminal 3 (Host - Editing):
  $ cd /home/kareem/Desktop/PAROL6_URDF
  $ code .
  # Edit files

═══════════════════════════════════════════════════════════════════════

WHY DOCKER?

✅ Consistent environment for all team members
✅ No ROS 2 installation needed on host
✅ Clean host system (no package conflicts)
✅ Easy onboarding (just install Docker)
✅ Portable (works on any OS with Docker)
✅ Version controlled (Dockerfile)

❌ DON'T try to install ROS 2 on host
❌ DON'T try to run ros2 commands on host
❌ DON'T copy files between host and container

✅ DO edit files on host
✅ DO run ROS commands in container
✅ DO use multiple terminals with docker exec

═══════════════════════════════════════════════════════════════════════

QUICK COMMANDS:

Start container:
  ./launch.sh

Enter running container:
  docker exec -it parol6_dev bash

Check if container is running:
  docker ps

Stop container:
  docker stop parol6_dev
  # Or Ctrl+C in container terminal

═══════════════════════════════════════════════════════════════════════

For more details, see: CONTAINER_ARCHITECTURE.md

EOF
