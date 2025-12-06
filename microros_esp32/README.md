# Micro-ROS ESP32 LED Control

This project implements a micro-ROS client on ESP32 that controls an LED via ROS 2 topics.

## Features

- **Subscriber**: Listens on `/parol6/esp32/led/command` for "ON", "OFF", "BLINK".
- **Publisher**: Publishes acknowledgment on `/parol6/esp32/led/status`.
- **LED Control**: Controls GPIO2 (Built-in LED) with non-blocking blinking.
- **Transport**: UART/Serial over `/dev/ttyUSB0` at 115200 baud.
- **ROS 2 Integration**: Uses standardized topic naming for easy integration with PAROL6 robot system.
- **Container-Based**: Designed to work seamlessly inside Docker container with existing ROS 2 setup.

## Prerequisites

- **Docker** installed and running
- **Docker container** `parol6_dev` (from the main project)
- **ESP-IDF v5.1+** (installed inside the container)
- **Micro-ROS Agent** (installed inside the container via `install_esp_tools.sh`)
- **ESP32 board** connected via USB

## Important: Working Inside Docker Container

⚠️ **All ESP-IDF commands must be run inside the Docker container**, not on your host machine. The ESP-IDF toolchain is installed inside the container at `/opt/esp-idf`.

## Installation Steps

### Step 1: Start the Docker Container

First, ensure the Docker container is running:

```bash
# From the project root directory
./start_ignition.sh
```

Or if the container is already running, you can enter it:

```bash
docker exec -it parol6_dev bash
```

### Step 2: Install ESP-IDF and Dependencies (First Time Only)

If ESP-IDF is not yet installed in the container, run the installation script:

```bash
# Inside the container
bash /workspace/scripts/setup/install_esp_tools.sh
```

This script will:
- Install all required dependencies (git, python3, cmake, etc.)
- Clone and install ESP-IDF v5.1 to `/opt/esp-idf`
- Build the Micro-ROS Agent workspace
- Set up environment variables in `~/.bashrc`

**Note**: The installation may take 15-30 minutes depending on your internet connection.

After installation, reload your shell environment:

```bash
source ~/.bashrc
```

### Step 3: Verify Micro-ROS Component

The micro-ROS component should already be cloned in `components/micro_ros_espidf_component/`. Verify it exists:

```bash
# Inside the container
cd /workspace/microros_esp32
ls -la components/micro_ros_espidf_component/
```

If it's missing, clone it:

```bash
cd /workspace/microros_esp32
mkdir -p components
git clone -b humble https://github.com/micro-ROS/micro_ros_espidf_component.git components/micro_ros_espidf_component
```

### Step 4: Install Python Dependencies

Install required Python packages for building micro-ROS:

```bash
# Inside the container
. /opt/esp-idf/export.sh
pip3 install catkin_pkg lark-parser colcon-common-extensions
```

**Important**: Install these packages **after** sourcing ESP-IDF, as they need to be in the ESP-IDF Python environment.

## Build and Flash

### Quick Build (Using Helper Script)

From the host machine, use the provided build script:

```bash
# From project root
./build_microros_esp32.sh [esp32|esp32s2|esp32s3|esp32c3]
```

This script automatically:
- Checks if the container is running
- Sources ESP-IDF
- Installs dependencies
- Builds the project

### Manual Build Steps

If you prefer to build manually:

1. **Enter the container**:
   ```bash
   docker exec -it parol6_dev bash
   ```

2. **Source ESP-IDF and navigate to project**:
   ```bash
   . /opt/esp-idf/export.sh
   cd /workspace/microros_esp32
   ```

3. **Set target board**:
   ```bash
   idf.py set-target esp32  # or esp32s2, esp32s3, esp32c3, esp32c6
   ```

4. **Configure (optional but recommended)**:
   ```bash
   idf.py menuconfig
   ```
   
   Navigate to:
   - `Micro-ROS Settings` → `Transport Settings` → Select `UART`
   - `Micro-ROS Settings` → `UART Configuration` → Set UART port (usually 0)
   - `Micro-ROS Settings` → `WiFi Configuration` → Disable if using UART

5. **Build**:
   ```bash
   idf.py build
   ```

   **First build will take 10-20 minutes** as it compiles the entire micro-ROS library.

6. **Flash to ESP32**:
   ```bash
   idf.py -p /dev/ttyUSB0 flash
   ```
   
   Replace `/dev/ttyUSB0` with your actual serial port. To find it:
   ```bash
   ls -la /dev/ttyUSB*  # or /dev/ttyACM*
   ```

7. **Monitor serial output**:
   ```bash
   idf.py monitor
   ```
   
   Press `Ctrl+]` to exit monitor.

## Usage

### 1. Start the Micro-ROS Agent

**Recommended: Run the agent inside the Docker container** (where it's already installed):

```bash
# Inside the container (or via docker exec)
docker exec -it parol6_dev bash -c "
    source /opt/ros/humble/setup.bash && \
    source /microros_ws/install/setup.bash && \
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
"
```

**Why run it in the container?**
- ✅ Agent is already installed there (from `install_esp_tools.sh`)
- ✅ Consistent environment with the rest of the ROS 2 system
- ✅ Container has device access (`-v /dev:/dev --privileged`)
- ✅ Easier to integrate with other ROS 2 nodes in the container
- ✅ No need to install ROS 2 on host machine

**Alternative: Run on host machine** (if you prefer):
```bash
# On host machine (requires ROS 2 installed)
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
```

### 2. Send Commands

**If agent is in container**, use another container terminal:

```bash
docker exec -it parol6_dev bash -c "
    source /opt/ros/humble/setup.bash && \
    ros2 topic pub --once /parol6/esp32/led/command std_msgs/msg/String \"data: 'ON'\"
"
```

**If agent is on host**, use a regular terminal:

```bash
# Turn LED ON
ros2 topic pub --once /parol6/esp32/led/command std_msgs/msg/String "data: 'ON'"

# Make LED BLINK
ros2 topic pub --once /parol6/esp32/led/command std_msgs/msg/String "data: 'BLINK'"

# Turn LED OFF
ros2 topic pub --once /parol6/esp32/led/command std_msgs/msg/String "data: 'OFF'"
```

### 3. Listen for Status

```bash
# Inside container
docker exec -it parol6_dev bash -c "
    source /opt/ros/humble/setup.bash && \
    ros2 topic echo /parol6/esp32/led/status
"

# Or on host (if agent is on host)
ros2 topic echo /parol6/esp32/led/status
```

## Topic Naming Convention

### Current Topics

The project uses the following topic names following ROS 2 best practices:

- **Command Topic**: `/parol6/esp32/led/command` (subscribed by ESP32)
- **Status Topic**: `/parol6/esp32/led/status` (published by ESP32)

### Why This Naming?

1. **Robot Namespace** (`/parol6/`): Identifies this as part of the PAROL6 robot system
2. **Component** (`/esp32/`): Identifies the hardware component (ESP32)
3. **Function** (`/led/`): Identifies the specific function (LED control)
4. **Direction** (`/command` or `/status`): Clear indication of data flow

### Benefits for Future Deployment

- ✅ **Scalable**: Easy to add more ESP32s (`/parol6/esp32_2/...`, `/parol6/esp32_gripper/...`)
- ✅ **Organized**: Clear hierarchy makes it easy to find topics
- ✅ **Integration**: Matches existing PAROL6 topic structure (e.g., `/parol6_arm_controller/...`)
- ✅ **ROS 2 Standard**: Follows ROS 2 naming conventions

### Customizing Topic Names

To change topic names, edit `main/main.c`:

```c
// Change these lines in micro_ros_task():
RCCHECK(rclc_publisher_init_default(
    &publisher, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/your/custom/topic/name"));  // ← Change here

RCCHECK(rclc_subscription_init_default(
    &subscriber, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/your/custom/topic/name"));  // ← Change here
```

Then rebuild and flash.

## Troubleshooting

### Build Error: "rcl/rcl.h: No such file or directory"

**Problem**: The micro-ROS component hasn't been built yet.

**Solution**: 
1. Make sure you've run `idf.py set-target esp32` first
2. The first build will compile the micro-ROS library - be patient (10-20 minutes)
3. If it still fails, try:
   ```bash
   idf.py clean-microros
   idf.py build
   ```

### Build Error: "idf.py: command not found"

**Problem**: ESP-IDF environment not sourced.

**Solution**: 
```bash
. /opt/esp-idf/export.sh
```

### Container Not Running

**Problem**: `docker exec` fails with "container not found".

**Solution**: 
```bash
./start_ignition.sh
```

### Permission Denied on /dev/ttyUSB0

**Problem**: User doesn't have permission to access serial port.

**Solution** (on host machine):
```bash
sudo usermod -aG dialout $USER
# Log out and log back in, or:
newgrp dialout
```

### Menuconfig Fails with Terminal Error

**Problem**: Terminal type not compatible with menuconfig.

**Solution**: 
```bash
export TERM=xterm-256color
idf.py menuconfig
```

Or use the non-interactive configuration method by editing `sdkconfig` directly.

### Micro-ROS Agent Can't Connect

**Problem**: Agent shows "Waiting for agent..." but never connects.

**Solutions**:
1. Check serial port: `ls -la /dev/ttyUSB*` (both in container and on host)
2. Verify baud rate matches (115200)
3. Check that ESP32 is powered and connected
4. Try resetting the ESP32 (press reset button)
5. Verify transport is set to UART in menuconfig
6. **If running agent in container**: Make sure container has device access:
   ```bash
   docker exec parol6_dev ls -la /dev/ttyUSB*
   ```
   If not found, restart container with `-v /dev:/dev --privileged` flags
7. **Check agent is running**: In another terminal, verify the agent process:
   ```bash
   # If in container
   docker exec parol6_dev ps aux | grep micro_ros_agent
   
   # If on host
   ps aux | grep micro_ros_agent
   ```

### Agent Not Found in Container

**Problem**: `ros2 run micro_ros_agent` fails with "package not found".

**Solution**: The agent needs to be built first. Run the installation script:
```bash
docker exec -it parol6_dev bash /workspace/scripts/setup/install_esp_tools.sh
```

This will build the micro-ROS agent workspace at `/microros_ws/install/setup.bash`.

### Build Takes Too Long

**Problem**: First build is very slow.

**Explanation**: This is normal! The first build compiles the entire micro-ROS library (thousands of files). Subsequent builds are much faster (usually < 1 minute).

**Tip**: Use `idf.py build -j4` to use 4 parallel jobs (adjust based on your CPU cores).

## Project Structure

```
microros_esp32/
├── components/
│   └── micro_ros_espidf_component/  # Micro-ROS component (required)
├── main/
│   ├── main.c                      # Main application code
│   └── CMakeLists.txt
├── CMakeLists.txt                  # Project CMake file
├── sdkconfig.defaults              # Default ESP-IDF configuration
└── README.md                       # This file
```

## Tips for Colleagues

1. **Always work inside the container**: ESP-IDF is installed at `/opt/esp-idf` inside the container, not on your host.

2. **Run agent in container**: The micro-ROS agent is already installed in the container - use it there for consistency and easier integration.

3. **First build is slow**: Be patient on the first build - it compiles the entire micro-ROS library (10-20 minutes).

4. **Keep container running**: Don't stop the container between builds - it's faster to keep it running.

5. **Use the helper script**: `./build_microros_esp32.sh` automates most steps.

6. **Check serial port**: ESP32 might appear as `/dev/ttyUSB0`, `/dev/ttyUSB1`, or `/dev/ttyACM0` depending on your system. Check both in container and on host.

7. **Monitor output**: Always check `idf.py monitor` output for connection status and errors.

8. **Topic discovery**: Use `ros2 topic list` to see all available topics. The ESP32 topics will appear as `/parol6/esp32/led/command` and `/parol6/esp32/led/status`.

9. **Clean builds**: If something goes wrong, try:
   ```bash
   idf.py fullclean
   idf.py build
   ```

10. **Configuration**: Use `idf.py menuconfig` to change settings. Changes are saved to `sdkconfig`.

11. **Topic naming**: Follow the `/parol6/component/function/direction` pattern for consistency with the rest of the robot system.

12. **Multiple ESP32s**: If you add more ESP32s, use distinct names like `/parol6/esp32_gripper/...` or `/parol6/esp32_sensors/...`.

## Quick Reference: Running Agent in Container

### Start Agent (Background)
```bash
docker exec -d parol6_dev bash -c "
    source /opt/ros/humble/setup.bash && \
    source /microros_ws/install/setup.bash && \
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
"
```

### Start Agent (Foreground - for debugging)
```bash
docker exec -it parol6_dev bash -c "
    source /opt/ros/humble/setup.bash && \
    source /microros_ws/install/setup.bash && \
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200 -v6
"
```

### Check if Agent is Running
```bash
docker exec parol6_dev ps aux | grep micro_ros_agent
```

### Stop Agent
```bash
docker exec parol6_dev pkill -f micro_ros_agent
```

### List All Topics (Verify ESP32 is Connected)
```bash
docker exec -it parol6_dev bash -c "
    source /opt/ros/humble/setup.bash && \
    ros2 topic list
"
```

You should see `/parol6/esp32/led/command` and `/parol6/esp32/led/status` in the list.

## Additional Resources

- [ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/)
- [Micro-ROS Documentation](https://micro.ros.org/)
- [Micro-ROS ESP-IDF Component](https://github.com/micro-ROS/micro_ros_espidf_component)
- [ROS 2 Topic Naming Conventions](https://design.ros2.org/articles/topic_and_service_names.html)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review build logs in `/workspace/microros_esp32/build/log/`
3. Check ESP32 monitor output for runtime errors
4. Verify all prerequisites are installed
5. Ensure the micro-ROS agent is running and can access the serial port
