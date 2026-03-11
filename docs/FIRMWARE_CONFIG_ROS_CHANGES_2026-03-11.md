# Firmware / Config / ROS Changes

Date: 2026-03-11

## Firmware

- Prevented the 1 kHz ISR from overwriting motor outputs while homing is active.
- Switched supervisor runaway checks to use joint-space velocity instead of motor-space velocity.
- Applied configured `HOME_OFFSETS_RAD` during homing zeroing so the post-home pose is no longer hardcoded to `0.0`.
- Reset integral state on enable and per-axis homing zero to avoid stale windup across mode transitions.
- Guarded homing setup against zero or invalid homing speeds.
- Throttled homing sequencer ticking to once per millisecond instead of as fast as `loop()` runs.
- Enabled real UART selection by mapping `TRANSPORT_MODE == 0` to `Serial1`.
- Made velocity deadband honor `FEATURE_VEL_DEADBAND` instead of applying unconditionally.
- Added stale/out-of-order command rejection in the Teensy command drain path. Rejected frames now emit `STALE_CMD`.
- Defensive zero-initialisation of `RosCommand` fields before parsing (prevents uninitialized reads on malformed packets).
- **[Hotfix]** Extended telemetry `<ACK>` packet to include a trailing `state_byte` so the GUI explicitly knows if the Teensy is in `NOMINAL`(1), `HOMING`(2), or `FAULT`(3) state.
- **[Hotfix]** Added strict anti-windup: `integral_error` is immediately zeroed across all axes whenever the supervisor transitions into *any* fault state (e.g. limit switch strike or watchdog softly disabling motors).

## ROS Hardware Interface

- Added explicit `allow_spoofing` control.
- Real hardware launch now defaults to `allow_spoofing=false`, so missing or invalid serial feedback fails loudly instead of silently echoing command as state.
- Read-path parse and bounds failures now return errors when spoofing is disabled.
- Write-path now errors if the serial link is absent and spoofing is disabled.

## MoveIt Launch

- Unified fake-hardware launch so `move_group`, RViz, and `ros2_control_node` use the same plugin-swapped robot description.

## Configurator / GUI

- Added `core/config_validator.py` with validation rules for unsupported or unsafe config combinations.
- Added a `Configuration Validation` panel in the Flash tab.
- Generate now blocks on validation errors instead of producing a misleading `config.h`.
- Added a transport capability note in the Comms tab.
- GUI serial connect now refuses Ethernet mode cleanly because the firmware does not implement it yet.
- Serial telemetry parsing now exposes `lim_state` when present.
- Corrected the default J4 mechanical switch polarity to `FALLING` so the stock profile matches the GUI’s recommended NC wiring.
- Added hard validation for NPN/PNP/mechanical pull and polarity combinations.
- **[Hotfix]** Serial monitor parser updated to extract the new `state_byte` from telemetry limits.
- **[Hotfix]** `code_generator.py` now automatically generates `parol6_moveit_config/config/joint_limits.yaml` mirroring `MAX_VEL_RAD_S` to ensure MoveIt and Teensy are perfectly synced.
- **[Hotfix]** Added a "✅ Validate Only" button to the Flash tab to run `config_validator.py` checks without triggering generation or flashing.

## Launcher / Docs

- Fixed `scripts/launchers/launch_moveit_real_hw.sh` so it now really starts `parol6_hardware real_robot.launch.py` first, waits for controllers, and then starts MoveIt/RViz.
- Added `PAROL6_SERIAL_PORT` and `PAROL6_BAUD_RATE` overrides to the real-hardware launcher.
- Updated launcher documentation to match the actual grouped command protocol and feedback format with `lim_state`.
- **[Hotfix]** Wrote `docs/POST_REAL_ROBOT_GUIDE.md` detailing dynamic PID tuning, Alpha-Beta filtering, homing offset calibration, and safety checks.

## Key Files Touched

- `parol6_firmware/src/main.cpp`
- `parol6_firmware/src/safety/Supervisor.h`
- `parol6_firmware/src/transport/SerialTransport.h`
- `parol6_firmware/README.md`
- `scripts/launchers/launch_moveit_real_hw.sh`
- `scripts/launchers/LAUNCH_METHODS.md`
- `parol6_hardware/src/parol6_system.cpp`
- `parol6_hardware/include/parol6_hardware/parol6_system.hpp`
- `parol6_hardware/urdf/parol6.ros2_control.xacro`
- `parol6_hardware/launch/real_robot.launch.py`
- `parol6_moveit_config/launch/demo.launch.py`
- `parol6_firmware_configurator/core/code_generator.py`
- `parol6_firmware_configurator/core/config_validator.py`
- `parol6_firmware_configurator/core/serial_monitor.py`
- `parol6_firmware_configurator/core/config_model.py`
- `parol6_firmware_configurator/main.py`
- `parol6_firmware_configurator/tabs/comms_tab.py`
- `parol6_firmware_configurator/tabs/flash_tab.py`
- `docs/POST_REAL_ROBOT_GUIDE.md`
