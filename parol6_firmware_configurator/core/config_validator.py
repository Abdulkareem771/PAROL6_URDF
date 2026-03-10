"""
config_validator.py — Lightweight validation rules for firmware configs.
Keeps GUI warnings/errors aligned with what the firmware actually supports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config_model import RobotConfig


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_robot_config(cfg: RobotConfig) -> ValidationReport:
    report = ValidationReport()

    if cfg.comms.transport == "ETHERNET":
        report.errors.append(
            "Ethernet transport is not implemented in the Teensy firmware yet. "
            "Use USB_CDC_HS or UART_115200."
        )

    homing_order = cfg.homing.order
    expected_axes = list(range(len(cfg.joints)))
    if sorted(homing_order) != expected_axes:
        report.errors.append(
            f"Homing order must contain each joint index exactly once: {expected_axes}."
        )

    seen_pins: dict[str, tuple[int, str]] = {}
    for index, joint in enumerate(cfg.joints):
        axis_name = joint.name or f"J{index + 1}"
        pin_fields = {
            "STEP": joint.step_pin,
            "DIR": joint.dir_pin,
            "ENC": joint.encoder_pin,
        }

        if joint.limit.enabled:
            pin_fields["LIMIT"] = joint.limit.pin

        for role, pin in pin_fields.items():
            key = f"pin:{pin}"
            owner = f"{axis_name} {role}"
            if key in seen_pins:
                report.errors.append(
                    f"Pin {pin} is assigned to both {seen_pins[key][1]} and {owner}."
                )
            else:
                seen_pins[key] = (pin, owner)

        if joint.max_vel_rad_s <= 0.0:
            report.errors.append(f"{axis_name} max velocity must be > 0.")

        if joint.homing_speed_steps_s <= 0:
            report.errors.append(f"{axis_name} homing speed must be > 0 steps/s.")

        if joint.limit.enabled and joint.limit.switch_type == "NONE":
            report.errors.append(f"{axis_name} enables a limit switch but type is NONE.")

        if joint.limit.switch_type == "MECHANICAL" and joint.limit.polarity == "RISING":
            report.warnings.append(
                f"{axis_name} uses MECHANICAL + RISING. Most NC mechanical switches use FALLING."
            )

    if cfg.comms.feedback_rate_hz <= 0 or cfg.comms.ros_command_rate_hz <= 0:
        report.errors.append("ROS command rate and feedback rate must both be > 0.")

    if cfg.comms.command_timeout_ms <= (1000 / max(cfg.comms.ros_command_rate_hz, 1)):
        report.warnings.append(
            "Command timeout is shorter than one ROS command period; transient packet jitter may estop the robot."
        )

    if any(not joint.enabled for joint in cfg.joints):
        report.warnings.append(
            "Disabled joints remain part of NUM_AXES and the ROS protocol. "
            "Keep host-side controllers aware of any disabled axes."
        )

    return report
