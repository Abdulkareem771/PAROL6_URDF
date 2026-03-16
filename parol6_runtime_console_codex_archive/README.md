# PAROL6 Runtime Console

A separate GUI for day-to-day firmware bringup and runtime operations.

It does not generate code and does not modify the existing firmware configurator. It is intended for:

- Serial monitor
- Oscilloscope / telemetry plotting
- PlatformIO build and flash
- ROS2 command launching

## Project Registry

Projects are defined in [`project_registry.json`](/home/kareem/Desktop/PAROL6_URDF/parol6_runtime_console/project_registry.json).

Each project can declare:

- Default serial port and baud
- PlatformIO project directory and environments
- A list of ROS2 actions

That keeps the app modular for future Teensy, ESP, and STM32 targets without rewiring the GUI.

## Run

```bash
python3 parol6_runtime_console/main.py
```

## Notes

- The `Flash` tab assumes PlatformIO is available in the environment where the GUI runs.
- The `ROS2` tab runs commands exactly as listed in the project registry.
- The oscilloscope consumes `<ACK,...>` packets compatible with the existing PAROL6 telemetry format.
