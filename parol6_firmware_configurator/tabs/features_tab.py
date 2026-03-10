"""
features_tab.py — Feature flag toggles. Each toggle writes a #define in config.h.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QFrame
)
from PyQt6.QtCore import pyqtSignal
from core.config_model import FeatureFlags


class FeatureRow(QWidget):
    """Label + toggle + description row."""
    changed = pyqtSignal()

    def __init__(self, label: str, tooltip: str, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)

        self.check = QCheckBox(label)
        self.check.setToolTip(tooltip)
        self.check.stateChanged.connect(self.changed)

        desc = QLabel(tooltip)
        desc.setStyleSheet("color: #6c7086; font-size: 11px;")
        desc.setWordWrap(True)

        row.addWidget(self.check)
        row.addWidget(desc, stretch=1)

    @property
    def value(self) -> bool:
        return self.check.isChecked()

    @value.setter
    def value(self, v: bool) -> None:
        self.check.blockSignals(True)
        self.check.setChecked(v)
        self.check.blockSignals(False)


class FeaturesTab(QWidget):
    """Displays all feature flags as toggle rows; emits changed when any flag changes."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("⚙️  Feature Flags")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)

        sub = QLabel("Each flag maps to a #define in the generated config.h. "
                     "Change flags then click 'Generate & Flash' in the Flash tab.")
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#a6adc8; font-size:12px;")
        root.addWidget(sub)

        hint = QLabel(
            "💡 <b>New to these flags?</b> Open the <b>📖 Docs</b> tab → "
            "<i>⚙️ Features Tab Guide</i> for a plain-English explanation of every toggle."
        )
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "background:#2a2a3e; border:1px solid #45475a; border-radius:6px; "
            "color:#cdd6f4; font-size:11px; padding:6px 10px; margin-bottom:4px;"
        )
        root.addWidget(hint)

        # ── Interpolator ─────────────────────────────────────────────
        interp_box = QGroupBox("Interpolator Configuration")
        interp_lay = QVBoxLayout(interp_box)
        
        self.f_lock = FeatureRow(
            "Lock Duration to ROS Rate",
            "Locks the 1kHz interpolator step size to exactly 1 / ROS_COMMAND_RATE_HZ preventing jitter from network latency."
        )
        self.f_lock.changed.connect(self.changed)
        interp_lay.addWidget(self.f_lock)
        interp_lay.addStretch()
        root.addWidget(interp_box)

        # ── Core Control Features ─────────────────────────────────────
        ctrl_box = QGroupBox("Control Features")
        ctrl_lay = QVBoxLayout(ctrl_box)

        self.f_filter = FeatureRow(
            "AlphaBeta Filter",
            "Smooths encoder noise for stable velocity estimation. Disable to test raw encoder.")
        self.f_ff = FeatureRow(
            "Velocity Feedforward",
            "Reduces tracking lag during motion. Disable to test pure P-control.")
        self.f_deadband = FeatureRow(
            "Velocity Deadband",
            "Suppresses micro-stepping noise when stationary. Threshold: 0.02 rad/s.")

        self.f_hardware_pwm = FeatureRow(
            "Hardware PWM Stepper Drive",
            "Uses FlexPWM hardware timers to generate STEP pulses (zero CPU overhead). Disable to fallback to basic software bitbang toggling.")

        self.f_hardware_encoder = FeatureRow(
            "Hardware Encoder (QuadTimer)",
            "Uses i.MXRT QuadTimers to count encoder PWM pulses (zero CPU interrupts). Disable to fallback to proven Software ISRs (attachInterrupt).")

        for row in (self.f_filter, self.f_ff, self.f_deadband, self.f_hardware_pwm, self.f_hardware_encoder):
            row.changed.connect(self.changed)
            ctrl_lay.addWidget(row)

        root.addWidget(ctrl_box)

        # ── Safety Features ───────────────────────────────────────────
        safe_box = QGroupBox("Safety Features")
        safe_lay = QVBoxLayout(safe_box)

        self.f_watchdog = FeatureRow(
            "Watchdog Timer",
            "Faults to SOFT_ESTOP if no command received within timeout. Disable ONLY for bench tests.")
        self.f_supervisor = FeatureRow(
            "Safety Supervisor",
            "Enforces per-joint velocity limits and state machine. Disable ONLY for open-loop signal tests.")
        self.f_antiglitch = FeatureRow(
            "Anti-Glitch Filter",
            "Rejects encoder readings that exceed physically possible delta. Prevents multi-turn corruption.")

        for row in (self.f_watchdog, self.f_supervisor, self.f_antiglitch):
            row.changed.connect(self.changed)
            safe_lay.addWidget(row)

        root.addWidget(safe_box)

        # ── Debug / Test Modes ────────────────────────────────────────
        debug_box = QGroupBox("Debug / Test Modes")
        debug_lay = QVBoxLayout(debug_box)

        self.f_enc_test = FeatureRow(
            "Encoder Test Mode",
            "Disables control loop entirely. Just reads and broadcasts encoder angles. Use in Phase 1 & 2.")
        
        self.f_sine_test = FeatureRow(
            "Sine Sweep Test Mode",
            "Internally generates a slow sine wave on all enabled joints to test kinematics. Ignores ROS commands.")
        
        self.f_enc_test.changed.connect(self.changed)
        self.f_sine_test.changed.connect(self.changed)
        debug_lay.addWidget(self.f_enc_test)
        debug_lay.addWidget(self.f_sine_test)

        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Fixed STEP Frequency (Hz):"))
        self.fixed_freq = QSpinBox()
        self.fixed_freq.setRange(0, 20000)
        self.fixed_freq.setSingleStep(100)
        self.fixed_freq.setSpecialValueText("Disabled (0)")
        self.fixed_freq.setToolTip(
            "0 = disabled (normal mode)\n"
            ">0 = all STEP pins output at this constant frequency, no control. Use in Phase 3.")
        self.fixed_freq.valueChanged.connect(self.changed)
        freq_row.addWidget(self.fixed_freq)
        freq_row.addStretch()
        debug_lay.addLayout(freq_row)

        deadband_row = QHBoxLayout()
        deadband_row.addWidget(QLabel("Deadband threshold (rad/s):"))
        self.deadband_val = QDoubleSpinBox()
        self.deadband_val.setRange(0.001, 1.0)
        self.deadband_val.setSingleStep(0.005)
        self.deadband_val.setDecimals(3)
        self.deadband_val.valueChanged.connect(self.changed)
        deadband_row.addWidget(self.deadband_val)
        deadband_row.addStretch()
        debug_lay.addLayout(deadband_row)

        root.addWidget(debug_box)
        root.addStretch()

    # ------------------------------------------------------------------
    def load(self, flags: FeatureFlags) -> None:
        self.f_lock.value      = flags.lock_interpolator
        self.f_filter.value    = flags.alphabeta_filter
        self.f_ff.value        = flags.velocity_feedforward
        self.f_watchdog.value  = flags.watchdog
        self.f_supervisor.value= flags.safety_supervisor
        self.f_antiglitch.value= flags.anti_glitch_filter
        self.f_deadband.value  = flags.velocity_deadband
        self.f_hardware_pwm.value = flags.hardware_pwm_step_dir
        self.f_hardware_encoder.value = flags.hardware_encoder_qtimer
        self.f_enc_test.value  = flags.encoder_test_mode
        self.f_sine_test.value = flags.sine_test_mode
        self.fixed_freq.setValue(flags.fixed_step_freq_hz)
        self.deadband_val.setValue(flags.velocity_deadband_rad_s)

    def save(self, flags: FeatureFlags) -> None:
        flags.lock_interpolator       = self.f_lock.value
        flags.alphabeta_filter        = self.f_filter.value
        flags.velocity_feedforward    = self.f_ff.value
        flags.watchdog                = self.f_watchdog.value
        flags.safety_supervisor       = self.f_supervisor.value
        flags.anti_glitch_filter      = self.f_antiglitch.value
        flags.velocity_deadband       = self.f_deadband.value
        flags.hardware_pwm_step_dir   = self.f_hardware_pwm.value
        flags.hardware_encoder_qtimer = self.f_hardware_encoder.value
        flags.encoder_test_mode       = self.f_enc_test.value
        flags.sine_test_mode          = self.f_sine_test.value
        flags.fixed_step_freq_hz      = self.fixed_freq.value()
        flags.velocity_deadband_rad_s = self.deadband_val.value()
