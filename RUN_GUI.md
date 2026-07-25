# Running the GUIs (PAROL6 & WeldVision)

This guide provides instructions on how to start the different graphical user interfaces within the PAROL6 development environment.

---

## 1. WeldVision Pipeline GUI (PySide6)
This is the main vision pipeline interface for object detection, mask editing, manual path annotation, and sending trajectories to MoveIt.

### Option A: From inside the Docker container (Recommended)
1. Enter the running container:
   ```bash
   docker exec -it parol6_dev bash
   ```
2. Source the environment and launch the GUI:
   ```bash
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   python3 parol6_vision/scripts/vision_pipeline_gui.py
   ```

### Option B: Using the Toolkit Launcher
To open the multi-tool selection screen:
1. Enter the running container:
   ```bash
   docker exec -it parol6_dev bash
   ```
2. Run the launcher script:
   ```bash
   python3 vision_work/launcher.py
   ```
3. Click on the **🔭 Vision Pipeline Launcher (ROS 2)** button.

---

## 2. Firmware Configurator GUI
Use this to configure the boards, setup parameters, and test motor configuration.

### From the Host Machine:
Simply run the GUI startup script on your host terminal:
```bash
./start_container_gui.sh
```
This script ensures X11 auth forwarding is active and launches the firmware configuration interface directly.

---

## Troubleshooting X11/GUI Issues

### error: `QXcbConnection: Could not connect to display`
If you cannot open any GUI from inside the container, verify X11 permissions on your **host machine** (outside the container):

1. **Allow local connections to the X server** (run on host):
   ```bash
   xhost +local:root
   ```
2. **Refresh X11 authorization tokens** (run on host):
   Optionally restart the container manager to re-inject the target token:
   ```bash
   ./start_container.sh
   ```
3. **Verify DISPLAY environment inside the container**:
   ```bash
   echo $DISPLAY
   ```
   It should match your host display variable (typically `:0` or `:1`).
