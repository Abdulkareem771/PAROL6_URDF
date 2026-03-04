"""
config_model.py — Pure data model for PAROL6 firmware configuration.
All settings serialise to/from JSON. No Qt imports here.
"""
from __future__ import annotations
import json
import copy
from dataclasses import dataclass, field, asdict
from typing import List


# ---------------------------------------------------------------------------
# Limit switch
# ---------------------------------------------------------------------------
@dataclass
class LimitSwitchConfig:
    enabled: bool = False
    pin: int = 0
    switch_type: str = "NONE"          # NONE | MECHANICAL | INDUCTIVE_NPN | INDUCTIVE_PNP
    polarity: str = "FALLING"          # FALLING | RISING  (auto-suggested by type)
    pull: str = "INPUT_PULLUP"         # INPUT_PULLUP | INPUT | INPUT_PULLDOWN


# ---------------------------------------------------------------------------
# Single joint
# ---------------------------------------------------------------------------
@dataclass
class JointConfig:
    name: str = "J1"
    enabled: bool = True

    # Pins
    step_pin: int = 2
    dir_pin: int = 30
    encoder_pin: int = 10

    # Mechanics (from legacy STM32 firmware — authoritative)
    gear_ratio: float = 6.4
    microsteps: int = 32
    steps_per_rev: int = 200
    dir_invert: bool = False        # Teensy DIR pin inversion (physical motor direction)
    ros_dir_invert: bool = False    # ROS kinematic sign inversion (read/write in parol6_system.cpp)

    # Safety
    max_vel_rad_s: float = 3.0
    max_current_ma: int = 2000

    # Homing
    homing_speed_steps_s: int = 500
    homed_position_steps: int = 13500  # steps from limit to zero
    standby_position_steps: int = 10240

    # Control gains
    kp: float = 5.0
    ki: float = 0.0
    max_integral: float = 5.0
    alpha: float = 0.1    # AlphaBeta filter position correction factor
    beta: float = 0.005   # AlphaBeta filter velocity correction factor
    home_offset_rad: float = 0.0 # Absolute physical offset in radians

    # Limit switch
    limit: LimitSwitchConfig = field(default_factory=LimitSwitchConfig)

    @property
    def steps_per_rad(self) -> float:
        import math
        return (self.steps_per_rev * self.microsteps * self.gear_ratio) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Feature flags — each maps to a #define in generated config.h
# ---------------------------------------------------------------------------
@dataclass
class FeatureFlags:
    lock_interpolator: bool = True         # True = 1000/comms.ros_command_rate_hz
    alphabeta_filter: bool = True
    velocity_feedforward: bool = True
    watchdog: bool = True
    safety_supervisor: bool = True
    anti_glitch_filter: bool = True
    velocity_deadband: bool = True
    encoder_test_mode: bool = False        # Disables control loop, just reads encoders
    sine_test_mode: bool = False           # Automatically sweeps joints in a sine wave
    hardware_pwm_step_dir: bool = True     # True = FlexPWM, False = basic software bitbang
    fixed_step_freq_hz: int = 0           # 0=off; >0 = all STEP pins at this fixed Hz
    velocity_deadband_rad_s: float = 0.02


# ---------------------------------------------------------------------------
# Communications & timing
# ---------------------------------------------------------------------------
@dataclass
class CommsConfig:
    transport: str = "USB_CDC_HS"          # UART_115200 | USB_CDC_HS | ETHERNET
    ros_command_rate_hz: int = 25
    feedback_rate_hz: int = 10
    control_loop_rate_hz: int = 1000       # 500 | 1000
    command_timeout_ms: int = 200

    # Serial monitor (GUI-side, not flashed)
    serial_port: str = ""
    baud_rate: int = 115200

    # Ethernet (only active when transport=ETHERNET)
    ethernet_ip: str = "192.168.1.177"
    ethernet_port: int = 8888
    ethernet_gateway: str = "192.168.1.1"
    ethernet_subnet: str = "255.255.255.0"


# ---------------------------------------------------------------------------
# Homing sequence
# ---------------------------------------------------------------------------
@dataclass
class HomingConfig:
    # Homing order: list of 0-based joint indices.
    # Default: J4,J5,J6 first (wrist), then J1,J2,J3 (base) — mechanical coupling.
    order: List[int] = field(default_factory=lambda: [3, 4, 5, 0, 1, 2])


# ---------------------------------------------------------------------------
# Top-level robot configuration
# ---------------------------------------------------------------------------
@dataclass
class RobotConfig:
    name: str = "default"
    description: str = ""
    phase: int = 0   # testing phase this config maps to

    joints: List[JointConfig] = field(default_factory=lambda: [
        JointConfig("J1", step_pin=2,  dir_pin=30, encoder_pin=10, gear_ratio=6.4,
                    microsteps=32, dir_invert=True,  ros_dir_invert=True,
                    max_vel_rad_s=3.0, max_current_ma=2000,
                    kp=5.0, homed_position_steps=13500, standby_position_steps=10240,
                    limit=LimitSwitchConfig(enabled=False, pin=20, switch_type="INDUCTIVE_NPN", polarity="FALLING")),

        JointConfig("J2", step_pin=6,  dir_pin=31, encoder_pin=11, gear_ratio=20.0,
                    microsteps=32, dir_invert=False, ros_dir_invert=False,
                    max_vel_rad_s=3.0, max_current_ma=2000,
                    kp=5.0, homed_position_steps=19588, standby_position_steps=-32000,
                    limit=LimitSwitchConfig(enabled=False, pin=21, switch_type="INDUCTIVE_NPN", polarity="RISING")),

        JointConfig("J3", step_pin=7,  dir_pin=32, encoder_pin=12, gear_ratio=18.0952381,
                    microsteps=32, dir_invert=True,  ros_dir_invert=True,
                    max_vel_rad_s=6.0, max_current_ma=1900,
                    kp=2.0, homed_position_steps=23020, standby_position_steps=57905,
                    limit=LimitSwitchConfig(enabled=False, pin=22, switch_type="INDUCTIVE_NPN", polarity="RISING")),

        JointConfig("J4", step_pin=8,  dir_pin=33, encoder_pin=14, gear_ratio=4.0,
                    microsteps=32, dir_invert=False, ros_dir_invert=False,
                    max_vel_rad_s=6.0, max_current_ma=1700,
                    kp=2.0, homed_position_steps=-10200, standby_position_steps=0,
                    limit=LimitSwitchConfig(enabled=False, pin=23, switch_type="MECHANICAL", polarity="RISING")),

        JointConfig("J5", step_pin=4,  dir_pin=34, encoder_pin=15, gear_ratio=4.0,
                    microsteps=32, dir_invert=False, ros_dir_invert=False,
                    max_vel_rad_s=6.0, max_current_ma=1700,
                    kp=2.0, homed_position_steps=8900,  standby_position_steps=0,
                    limit=LimitSwitchConfig(enabled=False, pin=24, switch_type="INDUCTIVE_NPN", polarity="RISING")),

        JointConfig("J6", step_pin=5,  dir_pin=35, encoder_pin=18, gear_ratio=10.0,
                    microsteps=32, dir_invert=True,  ros_dir_invert=True,
                    max_vel_rad_s=6.0, max_current_ma=965,
                    kp=2.0, homed_position_steps=15900, standby_position_steps=32000,
                    limit=LimitSwitchConfig(enabled=False, pin=25, switch_type="INDUCTIVE_NPN", polarity="FALLING")),
    ])

    features: FeatureFlags = field(default_factory=FeatureFlags)
    comms: CommsConfig = field(default_factory=CommsConfig)
    homing: HomingConfig = field(default_factory=HomingConfig)

    # -----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def from_dict(d: dict) -> "RobotConfig":
        cfg = RobotConfig.__new__(RobotConfig)
        cfg.name        = d.get("name", "default")
        cfg.description = d.get("description", "")
        cfg.phase       = d.get("phase", 0)

        cfg.joints = []
        for jd in d.get("joints", []):
            lim_d = jd.pop("limit", {})
            j = JointConfig(**{k: v for k, v in jd.items() if k in JointConfig.__dataclass_fields__})
            j.limit = LimitSwitchConfig(**{k: v for k, v in lim_d.items()
                                           if k in LimitSwitchConfig.__dataclass_fields__})
            cfg.joints.append(j)

        fd = d.get("features", {})
        cfg.features = FeatureFlags(**{k: v for k, v in fd.items()
                                       if k in FeatureFlags.__dataclass_fields__})

        cd = d.get("comms", {})
        cfg.comms = CommsConfig(**{k: v for k, v in cd.items()
                                   if k in CommsConfig.__dataclass_fields__})

        hd = d.get("homing", {})
        cfg.homing = HomingConfig(**{k: v for k, v in hd.items()
                                     if k in HomingConfig.__dataclass_fields__})
        return cfg

    @staticmethod
    def from_json(s: str) -> "RobotConfig":
        return RobotConfig.from_dict(json.loads(s))

    @staticmethod
    def load(path: str) -> "RobotConfig":
        with open(path) as f:
            return RobotConfig.from_json(f.read())

    def deep_copy(self) -> "RobotConfig":
        return RobotConfig.from_dict(copy.deepcopy(self.to_dict()))
