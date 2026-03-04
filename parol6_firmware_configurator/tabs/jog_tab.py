"""
jog_tab.py — Manual jog controls + encoder test mode readout per joint.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QGridLayout, QFrame, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont

N_JOINTS = 6
JOINT_NAMES = [f"J{i+1}" for i in range(N_JOINTS)]


class JointJogWidget(QFrame):
    """Per-joint jog row: name, encoder readout, jog +/-, velocity target."""
    jog_velocity = pyqtSignal(int, float)   # (joint_idx, vel_rad_s)

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self._idx = idx
        self.setObjectName("JogRow")
        self.setStyleSheet("""
            QFrame#JogRow {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)

        # Joint label
        name_lbl = QLabel(JOINT_NAMES[idx])
        name_lbl.setStyleSheet("font-weight:bold; color:#cba6f7; min-width:30px;")
        lay.addWidget(name_lbl)

        # Encoder position readout — radians
        self.enc_val = QLabel("enc: —")
        self.enc_val.setFont(QFont("Monospace", 10))
        self.enc_val.setStyleSheet("color:#a6e3a1; min-width:130px;")
        lay.addWidget(self.enc_val)

        # Degree readout
        self.enc_deg = QLabel("—°")
        self.enc_deg.setFont(QFont("Monospace", 10))
        self.enc_deg.setStyleSheet("color:#89dceb; min-width:70px;")
        lay.addWidget(self.enc_deg)

        # Velocity slider set
        step_lbl = QLabel("vel:")
        step_lbl.setStyleSheet("color:#a6adc8;")
        lay.addWidget(step_lbl)

        self.vel_spin = QDoubleSpinBox()
        self.vel_spin.setRange(0.01, 6.0)
        self.vel_spin.setValue(0.3)
        self.vel_spin.setSuffix(" rad/s")
        self.vel_spin.setDecimals(2)
        self.vel_spin.setFixedWidth(110)
        lay.addWidget(self.vel_spin)

        neg_btn = QPushButton("◀ −")
        neg_btn.setFixedWidth(60)
        neg_btn.pressed.connect(lambda: self._jog(-1))
        neg_btn.released.connect(lambda: self._jog(0))

        pos_btn = QPushButton("+ ▶")
        pos_btn.setFixedWidth(60)
        pos_btn.pressed.connect(lambda: self._jog(+1))
        pos_btn.released.connect(lambda: self._jog(0))

        lay.addWidget(neg_btn)
        lay.addWidget(pos_btn)

        # Enable toggle
        self.enabled_btn = QPushButton("ON")
        self.enabled_btn.setCheckable(True)
        self.enabled_btn.setChecked(True)
        self.enabled_btn.setFixedWidth(40)
        self.enabled_btn.toggled.connect(self._on_enable)
        lay.addWidget(self.enabled_btn)

        lay.addStretch()

    def _jog(self, sign: int) -> None:
        vel = self.vel_spin.value() * sign
        self.jog_velocity.emit(self._idx, vel)

    def _on_enable(self, checked: bool) -> None:
        self.enabled_btn.setText("ON" if checked else "OFF")
        self.enabled_btn.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e;" if checked else "background:#f38ba8; color:#1e1e2e;")
        if not checked:
            self.jog_velocity.emit(self._idx, 0.0)

    def update_encoder(self, angle_rad: float) -> None:
        import math
        self.enc_val.setText(f"enc: {angle_rad:+8.4f} rad")
        self.enc_deg.setText(f"{math.degrees(angle_rad):+7.2f}°")


class JogTab(QWidget):
    """Manual jog tab: per-joint velocity commands + encoder readout."""

    send_command = pyqtSignal(str)       # raw serial command string

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target_vel = [0.0] * N_JOINTS
        self._target_pos = [0.0] * N_JOINTS
        self._last_pos   = [0.0] * N_JOINTS
        self._seq = 0
        self._jog_timer = QTimer(self)
        self._jog_timer.timeout.connect(self._step_and_send)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        title = QLabel("🕹  Manual Jog & Encoder Monitor")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)

        info = QLabel(
            "Hold +/− to move a joint. Releases send zero velocity automatically. "
            "Encoder readouts update live from serial ACK packets."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#a6adc8; font-size:11px;")
        root.addWidget(info)

        # Global controls
        glob = QHBoxLayout()
        stop_all = QPushButton("🛑 STOP ALL")
        stop_all.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold; font-size:13px; padding:6px 18px;")
        stop_all.clicked.connect(self._stop_all)
        glob.addWidget(stop_all)

        enable_btn = QPushButton("🔓 CLEAR FAULT / ENABLE")
        enable_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold; font-size:13px; padding:6px 18px;")
        enable_btn.clicked.connect(self._send_enable)
        glob.addWidget(enable_btn)

        free_btn = QPushButton("Free Drive (all zero vel)")
        free_btn.clicked.connect(self._free_drive)
        glob.addWidget(free_btn)
        
        home_btn = QPushButton("🏠 HOME ALL")
        home_btn.setStyleSheet("background:#89dceb; color:#1e1e2e; font-weight:bold; font-size:13px; padding:6px 18px;")
        home_btn.clicked.connect(self._send_home)
        glob.addWidget(home_btn)
        
        glob.addStretch()
        root.addLayout(glob)

        # ISR profiler gauge
        isr_row = QHBoxLayout()
        isr_row.addWidget(QLabel("ISR budget:"))
        self.isr_bar = QProgressBar()
        self.isr_bar.setRange(0, 50)
        self.isr_bar.setValue(0)
        self.isr_bar.setFormat("%v µs")
        self.isr_bar.setMaximumHeight(16)
        isr_row.addWidget(self.isr_bar, stretch=1)
        self.isr_label = QLabel("— µs")
        self.isr_label.setStyleSheet("color:#a6adc8; min-width:60px;")
        isr_row.addWidget(self.isr_label)
        root.addLayout(isr_row)

        # Per-joint rows
        self._jog_widgets: list[JointJogWidget] = []
        for i in range(N_JOINTS):
            jw = JointJogWidget(i)
            jw.jog_velocity.connect(self._on_jog)
            self._jog_widgets.append(jw)
            root.addWidget(jw)

        root.addStretch()

    def ingest_telemetry(self, pkt: dict) -> None:
        """Update encoder displays and ISR bar from a parsed ACK packet."""
        pos = pkt.get("pos", [])
        for i, jw in enumerate(self._jog_widgets):
            if i < len(pos):
                jw.update_encoder(pos[i])
                self._last_pos[i] = pos[i]
                
        isr = pkt.get("isr_us")
        if isr is not None:
            v = int(isr)
            self.isr_bar.setValue(min(v, 50))
            colour = "#f38ba8" if v > 25 else "#a6e3a1"
            self.isr_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {colour}; }}")
            self.isr_label.setText(f"{v} µs")

    def _on_jog(self, idx: int, vel: float) -> None:
        self._target_vel[idx] = vel
        is_moving = any(v != 0.0 for v in self._target_vel)
        
        if is_moving and not self._jog_timer.isActive():
            # Snapshot current positions to start interpolating from
            for i in range(N_JOINTS):
                self._target_pos[i] = self._last_pos[i]
            self._jog_timer.start(40)  # 25 Hz stream
        elif not is_moving and self._jog_timer.isActive():
            # Stop timer but send one final zero-velocity halt frame
            for i in range(N_JOINTS):
                self._target_pos[i] = self._last_pos[i]
            self._step_and_send()
            self._jog_timer.stop()

    def _step_and_send(self) -> None:
        dt = 0.04  # 40ms timer -> 0.04s
        for i in range(N_JOINTS):
            if self._target_vel[i] == 0.0:
                # Snap to current true position when not jogging this axis
                self._target_pos[i] = self._last_pos[i]
            else:
                self._target_pos[i] += self._target_vel[i] * dt

        p_str = ",".join(f"{p:.4f}" for p in self._target_pos)
        v_str = ",".join(f"{v:.4f}" for v in self._target_vel)
        cmd = f"<{self._seq},{p_str},{v_str}>"
        
        self.send_command.emit(cmd)
        self._seq += 1

    def _stop_all(self) -> None:
        for i in range(N_JOINTS):
            self._target_vel[i] = 0.0
        self._step_and_send()
        self._jog_timer.stop()

    def _free_drive(self) -> None:
        self._stop_all()

    def _send_home(self) -> None:
        self._stop_all()
        self.send_command.emit("<HOME>")

    def _send_enable(self) -> None:
        self._stop_all()
        self.send_command.emit("<ENABLE>")

