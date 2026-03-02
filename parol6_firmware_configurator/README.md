# PAROL6 Firmware Configurator & Launcher

A unified desktop application to configure the PAROL6 robot hardware, tune control loops, and launch the ROS 2 software stack. 

## Overview
This tool eliminates the need to manually edit C++ headers (`config.h`), memorize bash launcher arguments, or open multiple terminal windows.

### Features
* **Configuration Generation:** Tune PID gains, set velocity limits, select transport modes (USB CDC/UART), and map hardware pins via the GUI.
* **Firmware Flashing:** Generates the C++ `config.h` and natively cross-compiles and flashes the Teensy 4.1.
* **Live Telemetry & Jog:** Real-time PyQtGraph oscilloscope showing commanded vs actual positions and velocities. Includes a manual slider-based JOG tab for testing individual joints safely before integrating ROS.
* **Unified ROS Launcher:** Start the robot in `sim` (Gazebo), `real` (Hardware), or `fake` (RViz only) mode directly from the GUI. The container logs are streamed directly into the application.

## Setup
The GUI requires Python 3.10+ and PyQt6. It is run directly on the host machine (not inside the Docker container), while it interfaces with the container behind the scenes to launch ROS.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the configurator
python3 main.py
```

## Developer Usage Guide

### 1. Hardware Tuning (The "config.h" Pipeline)
When you modify a setting in the GUI (e.g., changing J1's $K_p$ from 5.0 to 1.5 in the **Joints** tab, or changing the Control Rate in the **Comms** tab):
1. Go to the **Flash** tab.
2. Click **Generate config.h**. This writes your changes to `parol6_firmware/generated/config.h`. (You can view the generated C++ code in the preview window).
3. Click **Compile & Flash** (or Flash Only if it's already built). The firmware's `main.cpp` will automatically pick up the new gains and limits from the generated header.

### 2. Testing Motors (Jog & Oscilloscope)
*Always verify new limits safely before putting them into ROS.*
1. Go to the **Serial** tab and ensure you are connected to the Teensy (e.g. `/dev/ttyACM0`).
2. Go to the **Jog** tab. 
3. Move the sliders. Notice how the robot physically moves the targeted joint using the velocity stream commanded by the GUI.
4. Go to the **Oscilloscope** tab to verify that the Actual Position (read from the MT6816 encoders) tracks the Commanded Position without jitter. If you see oscillation, lower your $K_p$ in the config and re-flash.

### 3. Launching ROS 2 Stack
Once the hardware is tuned, you can start the full MoveIt stack:
1. Ensure the Docker container is running (`./start_container.sh` in the project root).
2. Go to the **ROS2 Launch** tab in the GUI.
3. Select your target:
   * **Real Hardware:** Runs `real_robot_driver.py` to bridge MoveIt to the Teensy.
   * **Simulation:** Boots Gazebo classic and spawns the URDF model.
   * **Fake:** Boots RViz with the `fake_components` controller (no hardware/physics).
4. Click **Launch**. RViz/Gazebo will open natively on your screen, and all backend logs will stream into the GUI's terminal window.

### 4. Logging Faults
If the hardware supervisor triggers an E-Stop (e.g., velocity limit exceeded, encoder read error), it is immediately logged in the **Faults** tab. You can export these logs to CSV for post-mortem debugging.
