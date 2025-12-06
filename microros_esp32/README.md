# Micro-ROS ESP32 LED Control

This project implements a micro-ROS client on ESP32 that controls an LED via ROS 2 topics.

## Features

- **Subscriber**: Listens on `esp/led_cmd` for "ON", "OFF", "BLINK".
- **Publisher**: Publishes acknowledgment on `esp/led_status`.
- **LED Control**: Controls GPIO2 (Built-in LED) with non-blocking blinking.
- **Transport**: UART/Serial over `/dev/ttyUSB0` at 115200 baud.

## Prerequisites

- **Docker** installed and running
- **Docker container** `parol6_dev` (from the main project)
- **ESP-IDF v5.1+** (installed inside the container)
- **Micro-ROS Agent** installed on the host/agent machine
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

On your **host machine** (not in the container), start the micro-ROS agent:

```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
```

**Note**: The agent must run on the host machine to access the USB serial port. Make sure ROS 2 is installed on your host.

### 2. Send Commands

In another terminal (on host machine):

```bash
# Turn LED ON
ros2 topic pub --once /esp/led_cmd std_msgs/msg/String "data: 'ON'"

# Make LED BLINK
ros2 topic pub --once /esp/led_cmd std_msgs/msg/String "data: 'BLINK'"

# Turn LED OFF
ros2 topic pub --once /esp/led_cmd std_msgs/msg/String "data: 'OFF'"
```

### 3. Listen for Status

```bash
ros2 topic echo /esp/led_status
```

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
1. Check serial port: `ls -la /dev/ttyUSB*`
2. Verify baud rate matches (115200)
3. Check that ESP32 is powered and connected
4. Try resetting the ESP32 (press reset button)
5. Verify transport is set to UART in menuconfig

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

2. **First build is slow**: Be patient on the first build - it compiles the entire micro-ROS library.

3. **Keep container running**: Don't stop the container between builds - it's faster to keep it running.

4. **Use the helper script**: `./build_microros_esp32.sh` automates most steps.

5. **Check serial port**: ESP32 might appear as `/dev/ttyUSB0`, `/dev/ttyUSB1`, or `/dev/ttyACM0` depending on your system.

6. **Monitor output**: Always check `idf.py monitor` output for connection status and errors.

7. **Clean builds**: If something goes wrong, try:
   ```bash
   idf.py fullclean
   idf.py build
   ```

8. **Configuration**: Use `idf.py menuconfig` to change settings. Changes are saved to `sdkconfig`.

## Additional Resources

- [ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/)
- [Micro-ROS Documentation](https://micro.ros.org/)
- [Micro-ROS ESP-IDF Component](https://github.com/micro-ROS/micro_ros_espidf_component)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review build logs in `/workspace/microros_esp32/build/log/`
3. Check ESP32 monitor output for runtime errors
4. Verify all prerequisites are installed
