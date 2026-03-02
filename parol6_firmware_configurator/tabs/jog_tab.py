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

        # Encoder position readout
        self.enc_val = QLabel("enc: —")
        self.enc_val.setFont(QFont("Monospace", 10))
        self.enc_val.setStyleSheet("color:#a6e3a1; min-width:130px;")
        lay.addWidget(self.enc_val)

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
        self.enc_val.setText(f"enc: {angle_rad:+8.4f} rad")


class JogTab(QWidget):
    """Manual jog tab: per-joint velocity commands + encoder readout."""

    send_command = pyqtSignal(str)       # raw serial command string

    def __init__(self, parent=None):
        super().__init__(parent)
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

        free_btn = QPushButton("Free Drive (all zero vel)")
        free_btn.clicked.connect(self._free_drive)
        glob.addWidget(free_btn)
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
        for i, jw in enumerate(self._jog_widgets):
            if i < len(pkt.get("pos", [])):
                jw.update_encoder(pkt["pos"][i])
        isr = pkt.get("isr_us")
        if isr is not None:
            v = int(isr)
            self.isr_bar.setValue(min(v, 50))
            colour = "#f38ba8" if v > 25 else "#a6e3a1"
            self.isr_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {colour}; }}")
            self.isr_label.setText(f"{v} µs")

    def _on_jog(self, idx: int, vel: float) -> None:
        # Send a velocity-only jog command: <JOG,idx,vel>
        self.send_command.emit(f"<JOG,{idx},{vel:.4f}>")

    def _stop_all(self) -> None:
        for i in range(N_JOINTS):
            self.send_command.emit(f"<JOG,{i},0.0>")

    def _free_drive(self) -> None:
        self._stop_all()
