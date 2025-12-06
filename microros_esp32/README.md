# Micro-ROS ESP32 LED Control

This project implements a micro-ROS client on ESP32 that controls an LED via ROS 2 topics.

## Features

- **Subscriber**: Listens on `esp/led_cmd` for "ON", "OFF", "BLINK".
- **Publisher**: Publishes acknowledgment on `esp/led_status`.
- **LED Control**: Controls GPIO2 (Built-in LED) with non-blocking blinking.
- **Transport**: UART/Serial over `/dev/ttyUSB0` at 115200 baud.

## Prerequisites

- ESP-IDF v5.x
- Micro-ROS Agent installed on the host/agent machine.

## Setup

1.  **Clone the micro-ROS component**:
    You must clone the `micro_ros_espidf_component` into the `components` directory.
    ```bash
    mkdir -p components
    git clone -b humble https://github.com/micro-ROS/micro_ros_espidf_component.git components/micro_ros_espidf_component
    # Or use the branch matching your ROS 2 distribution
    ```

2.  **Configuration**:
    The project is configured to use the micro-ROS component.
    If you need to change the serial port or other settings, run:
    ```bash
    idf.py menuconfig
    ```
    Navigate to `Micro-ROS Settings` -> `WiFi Configuration` (disable) / `Transport Settings` (select UART).
    *Note: By default, the component might default to UART or WiFi depending on the version. Ensure UART is selected and configured for UART0 or the appropriate UART.*

## Build and Flash

1.  **Setup Environment**:
    ```bash
    source ~/esp-idf/export.sh
    ```

2.  **Build**:
    ```bash
    cd /workspace/microros_esp32
    idf.py build
    ```

3.  **Flash**:
    ```bash
    idf.py -p /dev/ttyUSB0 flash
    ```

4.  **Monitor**:
    ```bash
    idf.py monitor
    ```

## Usage

1.  **Start the Micro-ROS Agent**:
    On your host machine (or wherever the agent is running):
    ```bash
    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
    ```

2.  **Send Commands**:
    ```bash
    ros2 topic pub --once /esp/led_cmd std_msgs/msg/String "data: 'ON'"
    ros2 topic pub --once /esp/led_cmd std_msgs/msg/String "data: 'BLINK'"
    ros2 topic pub --once /esp/led_cmd std_msgs/msg/String "data: 'OFF'"
    ```

3.  **Listen for Status**:
    ```bash
    ros2 topic echo /esp/led_status
    ```
