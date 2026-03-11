# PAROL6 Firmware Changelog

All notable changes to the firmware and tight integrations will be documented in this file.

## [Unreleased] - 2026-03-11

### Added
- **Telemetry State Byte**: Extended telemetry `<ACK>` packet to include a trailing `state_byte` (`NOMINAL`=1, `HOMING`=2, `FAULT/ESTOP`=3).
- **Anti-Windup Reset**: Integral error (`integral_error`) is immediately zeroed across all axes whenever the supervisor transitions into any fault state (e.g., limit switch strike or watchdog timeout).
- **GUI Validations**: Introduced a heavy-duty validator in the Configurator that stops `config.h` generation if unsafe configurations are chosen.
- **`joint_limits.yaml` Auto-generation**: The configurator now automatically outputs MoveIt-compatible limits directly from the firmware `MAX_VEL_RAD_S` setting to guarantee kinematic sync.
- **Xacro Dynamic Inversion**: The configurator now patches the GUI's `ROS_DIR_INVERT` directly into `parol6.ros2_control.xacro`, completely eliminating missing directionality bugs.

### Changed
- **Default Command Rate**: Changed default `ros_command_rate_hz` to `100` Hz to perfectly match the `ros2_controllers.yaml` update rate, fixing interpolator time skew/drift.
- **Baud Rate Macro**: Firmware now uses `UART_BAUD_RATE` emitted by the configurator instead of a hardcoded `115200`, supporting parameter overrides securely.
- **ROS Hardware Spoofing**: Tightened `allow_spoofing` controls. Read-path bounds and token counts (Now strictly 14-18 tokens to accept `state_byte`) will accurately throw `ERROR` to the controller manager if real serial hardware dies.
- **URDF Consistency**: The `parol6.urdf.xacro` dummy SIL model now simply includes the true `PAROL6.urdf` CAD definition instead of defining its own mismatched geometric primitives.

### Fixed
- Fixed an `AttributeError` crashing the limits generator when looking for an acceleration property.
- Fixed the `allow_spoofing` launch argument not actually piercing the xacro macro.
