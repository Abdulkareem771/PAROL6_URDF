"""
jog_tab.py — Manual jog controls + encoder test mode readout per joint.
"""
from __future__ import annotations
import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QGridLayout, QFrame, QProgressBar,
    QScrollArea, QSlider, QLineEdit, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QDoubleValidator

N_JOINTS = 6
JOINT_NAMES = [f"J{i+1}" for i in range(N_JOINTS)]

# Degree range per joint — matches URDF limits (±3.14 rad → ±180°)
# Customise per joint if your URDF has asymmetric limits.
JOINT_DEG_RANGE = [
    (-180, 180),   # J1
    (-180, 180),   # J2
    (-180, 180),   # J3
    (-180, 180),   # J4
    (-180, 180),   # J5
    (-180, 180),   # J6
]

# Slider integer resolution: 1 tick = 0.5 °
_TICKS_PER_DEG = 2


def _deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0


def _rad_to_deg(r: float) -> float:
    return r * 180.0 / math.pi


class JointJogWidget(QFrame):
    """
    Per-joint jog widget — two rows inside a rounded card:
      Row 1: joint label | encoder readout | vel spinbox | ◀− / +▶ buttons | ON toggle
      Row 2: deg label | slider (−360 … +360°) | degree text input | 🏠 Home joint
    """
    jog_velocity       = pyqtSignal(int, float)   # (joint_idx, vel_rad_s)
    goto_position      = pyqtSignal(int, float)   # (joint_idx, target_rad)
    home_single_joint  = pyqtSignal(int)          # (joint_idx)

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self._idx = idx
        self._slider_updating = False   # guard against slider↔textbox feedback loops
        self.setObjectName("JogRow")
        self.setStyleSheet("""
            QFrame#JogRow {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(4)

        # ── Row 1: velocity jog ────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        name_lbl = QLabel(JOINT_NAMES[idx])
        name_lbl.setStyleSheet("font-weight:bold; color:#cba6f7; min-width:30px;")
        row1.addWidget(name_lbl)

        # Encoder readout — radians
        self.enc_val = QLabel("enc: —")
        self.enc_val.setFont(QFont("Monospace", 10))
        self.enc_val.setStyleSheet("color:#a6e3a1; min-width:130px;")
        row1.addWidget(self.enc_val)

        # Degree readout (live)
        self.enc_deg = QLabel("—°")
        self.enc_deg.setFont(QFont("Monospace", 10))
        self.enc_deg.setStyleSheet("color:#89dceb; min-width:70px;")
        row1.addWidget(self.enc_deg)

        step_lbl = QLabel("vel:")
        step_lbl.setStyleSheet("color:#a6adc8;")
        row1.addWidget(step_lbl)

        self.vel_spin = QDoubleSpinBox()
        self.vel_spin.setRange(0.01, 6.0)
        self.vel_spin.setValue(0.3)
        self.vel_spin.setSuffix(" rad/s")
        self.vel_spin.setDecimals(2)
        self.vel_spin.setFixedWidth(110)
        row1.addWidget(self.vel_spin)

        neg_btn = QPushButton("◀  −")
        neg_btn.setMinimumWidth(76)
        neg_btn.pressed.connect(lambda: self._jog(-1))
        neg_btn.released.connect(lambda: self._jog(0))
        pos_btn = QPushButton("+  ▶")
        pos_btn.setMinimumWidth(76)
        pos_btn.pressed.connect(lambda: self._jog(+1))
        pos_btn.released.connect(lambda: self._jog(0))
        row1.addWidget(neg_btn)
        row1.addWidget(pos_btn)

        self.enabled_btn = QPushButton("ON")
        self.enabled_btn.setCheckable(True)
        self.enabled_btn.setChecked(True)
        self.enabled_btn.setMinimumWidth(50)
        self.enabled_btn.toggled.connect(self._on_enable)
        row1.addWidget(self.enabled_btn)
        row1.addStretch()
        outer.addLayout(row1)

        # ── Row 2: degree slider + text input + per-joint home ─────────
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        deg_lbl = QLabel("°:")
        deg_lbl.setStyleSheet("color:#a6adc8; min-width:14px;")
        row2.addWidget(deg_lbl)

        lo, hi = JOINT_DEG_RANGE[idx]
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(lo * _TICKS_PER_DEG))
        self._slider.setMaximum(int(hi * _TICKS_PER_DEG))
        self._slider.setValue(0)
        self._slider.setTickInterval(int(90 * _TICKS_PER_DEG))
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px; background: #45475a; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #cba6f7; width: 14px; height: 14px;
                margin: -4px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
        """)
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        row2.addWidget(self._slider, stretch=1)

        # Min / max labels
        lo_lbl = QLabel(f"{lo}°")
        lo_lbl.setStyleSheet("color:#585b70; font-size:9px;")
        hi_lbl = QLabel(f"{hi}°")
        hi_lbl.setStyleSheet("color:#585b70; font-size:9px;")
        row2.insertWidget(1, lo_lbl)   # between ° label and slider
        row2.addWidget(hi_lbl)

        # Degree text input (0.0 format, accepts within joint limits)
        self._deg_input = QLineEdit()
        self._deg_input.setFixedWidth(80)
        lo_d, hi_d = JOINT_DEG_RANGE[idx]
        self._deg_input.setPlaceholderText(f"{lo_d}…{hi_d}°")
        self._deg_input.setText("0.0")
        self._deg_input.setValidator(QDoubleValidator(float(lo_d), float(hi_d), 2))
        self._deg_input.setToolTip(
            "Type a degree value (−360 to 360) and press Enter\n"
            "to move this joint to that absolute position."
        )
        self._deg_input.setStyleSheet(
            "background:#1e1e2e; color:#cdd6f4; border:1px solid #45475a; "
            "border-radius:4px; padding:2px 4px;"
        )
        self._deg_input.returnPressed.connect(self._on_deg_input_committed)
        row2.addWidget(QLabel("→"))
        row2.addWidget(self._deg_input)

        go_btn = QPushButton("Go")
        go_btn.setMinimumWidth(52)
        go_btn.setToolTip("Move to the degree value entered in the text box")
        go_btn.setStyleSheet("background:#89b4fa; color:#1e1e2e; font-weight:bold; padding:2px 6px;")
        go_btn.clicked.connect(self._on_deg_input_committed)
        row2.addWidget(go_btn)

        home_j_btn = QPushButton(f"🏠 J{idx+1}")
        home_j_btn.setFixedWidth(60)
        home_j_btn.setToolTip(f"Home joint {idx+1} only")
        home_j_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")
        home_j_btn.clicked.connect(lambda: self.home_single_joint.emit(self._idx))
        row2.addWidget(home_j_btn)

        outer.addLayout(row2)

    # ── internal helpers ───────────────────────────────────────────────

    def _jog(self, sign: int) -> None:
        vel = self.vel_spin.value() * sign
        self.jog_velocity.emit(self._idx, vel)

    def _on_enable(self, checked: bool) -> None:
        self.enabled_btn.setText("ON" if checked else "OFF")
        self.enabled_btn.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e;" if checked
            else "background:#f38ba8; color:#1e1e2e;"
        )
        if not checked:
            self.jog_velocity.emit(self._idx, 0.0)

    def _on_slider_changed(self, tick_val: int) -> None:
        """Update degree-input text while dragging; do NOT send command yet."""
        if self._slider_updating:
            return
        deg = tick_val / _TICKS_PER_DEG
        self._slider_updating = True
        self._deg_input.setText(f"{deg:.1f}")
        self._slider_updating = False

    def _on_slider_released(self) -> None:
        """Send goto command when user releases the slider thumb."""
        deg = self._slider.value() / _TICKS_PER_DEG
        self.goto_position.emit(self._idx, _deg_to_rad(deg))

    def _on_deg_input_committed(self) -> None:
        """Send goto command from the text box (Enter or Go button)."""
        try:
            deg = float(self._deg_input.text())
        except ValueError:
            return
        deg = max(-360.0, min(360.0, deg))
        self._deg_input.setText(f"{deg:.2f}")
        # Sync slider without triggering another goto
        self._slider_updating = True
        self._slider.setValue(int(deg * _TICKS_PER_DEG))
        self._slider_updating = False
        self.goto_position.emit(self._idx, _deg_to_rad(deg))

    # ── public API ─────────────────────────────────────────────────────

    def update_encoder(self, angle_rad: float) -> None:
        deg = _rad_to_deg(angle_rad)
        self.enc_val.setText(f"enc: {angle_rad:+8.4f} rad")
        self.enc_deg.setText(f"{deg:+7.2f}°")

        # Mirror live position into slider and input ONLY when not dragging
        if not self._slider.isSliderDown() and not self._slider_updating:
            self._slider_updating = True
            clamped = max(JOINT_DEG_RANGE[self._idx][0],
                         min(JOINT_DEG_RANGE[self._idx][1], deg))
            self._slider.setValue(int(clamped * _TICKS_PER_DEG))
            self._deg_input.setText(f"{deg:.2f}")
            self._slider_updating = False


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
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        title = QLabel("🕹  Manual Jog & Encoder Monitor")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)

        # ── Instruction panel ─────────────────────────────────────────
        instr = QFrame()
        instr.setStyleSheet(
            "QFrame { background:#1e2a1e; border:1px solid #a6e3a1; "
            "border-radius:8px; padding:2px; }"
        )
        instr_lay = QVBoxLayout(instr)
        instr_lay.setSpacing(4)
        instr_lay.setContentsMargins(12, 8, 12, 8)

        def _lbl(text, color="#cdd6f4", sz=11):
            l = QLabel(text)
            l.setTextFormat(Qt.TextFormat.RichText)
            l.setWordWrap(True)
            l.setStyleSheet(f"color:{color}; font-size:{sz}px; background:transparent; border:none;")
            return l

        instr_lay.addWidget(_lbl(
            "<b>📋 How to use the Jog tab</b>",
            color="#a6e3a1", sz=12
        ))
        instr_lay.addWidget(_lbl(
            "<b>Before jogging:</b> "
            "Connect to the Teensy first (Port → ⚡ Connect). "
            "You should see <code>&lt;ACK,...&gt;</code> frames in 💬 Serial. "
            "If in FAULT state, click <b>🔓 CLEAR FAULT / ENABLE</b> first."
        ))
        instr_lay.addWidget(_lbl(
            "<b>Row 1 — Velocity jog:</b> Set <b>vel</b> (rad/s) → "
            "<b>Hold ◀ −</b> or <b>+ ▶</b>. Releasing stops the joint immediately. "
            "Start at <b>0.1–0.3 rad/s</b> for safety. "
            "<b>Row 2 — Degree control:</b> Drag the <b>slider</b> and release, "
            "or type a value into the <b>degree box → Go / Enter</b> to move to an absolute angle. "
            "Range is ±360°. <b>🏠 Jx</b> button homes that single joint only."
        ))
        row2 = QHBoxLayout()
        row2.addWidget(_lbl(
            "<b>🏠 HOME ALL</b> — starts the full homing sequence. "
            "Firmware replies <code>HOMING_DONE</code> or <code>HOMING_FAULT</code>."
        ))
        row2.addWidget(_lbl(
            "<b>ISR budget bar</b> — 1 kHz loop time in µs. "
            "<span style='color:#f38ba8;'>Red = over 25 µs</span> → overrun risk."
        ))
        instr_lay.addLayout(row2)
        root.addWidget(instr)

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
        self.isr_bar.setFormat("%v µs  (warn >25)")
        self.isr_bar.setMaximumHeight(16)
        isr_row.addWidget(self.isr_bar, stretch=1)
        self.isr_label = QLabel("— µs")
        self.isr_label.setStyleSheet("color:#a6adc8; min-width:60px;")
        isr_row.addWidget(self.isr_label)
        root.addLayout(isr_row)

        # Column headers (Row 1 columns)
        hdr = QHBoxLayout()
        for txt, w, color in [
            ("Joint",          36,  "#a6adc8"),
            ("Position (enc)", 140, "#a6adc8"),
            ("Degrees",        80,  "#a6adc8"),
            ("Jog vel",        120, "#a6adc8"),
            ("◀ −  / + ▶",    130, "#a6adc8"),
            ("En",             50,  "#a6adc8"),
        ]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet(f"color:{color}; font-size:10px; font-weight:bold;")
            hdr.addWidget(lbl)
        hdr.addStretch()
        root.addLayout(hdr)

        # Per-joint rows
        self._jog_widgets: list[JointJogWidget] = []
        for i in range(N_JOINTS):
            jw = JointJogWidget(i)
            jw.jog_velocity.connect(self._on_jog)
            jw.goto_position.connect(self._goto_position)
            jw.home_single_joint.connect(self._home_single_joint)
            self._jog_widgets.append(jw)
            root.addWidget(jw)

        root.addStretch()

    # ── slot handlers ──────────────────────────────────────────────────

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
            for i in range(N_JOINTS):
                self._target_pos[i] = self._last_pos[i]
            self._jog_timer.start(40)   # 25 Hz stream
        elif not is_moving and self._jog_timer.isActive():
            for i in range(N_JOINTS):
                self._target_pos[i] = self._last_pos[i]
            self._step_and_send()
            self._jog_timer.stop()

    def _goto_position(self, idx: int, target_rad: float) -> None:
        """Send a single absolute position command for one joint."""
        self._stop_all()   # zero all velocities first
        # Build position array: hold all other joints at current position
        pos = list(self._last_pos)
        pos[idx] = target_rad
        p_str = ",".join(f"{p:.4f}" for p in pos)
        v_str = ",".join("0.0000" for _ in range(N_JOINTS))
        cmd = f"<{self._seq},{p_str},{v_str}>"
        self.send_command.emit(cmd)
        self._seq += 1
        # Optimistically update our tracking position
        self._target_pos[idx] = target_rad
        self._last_pos[idx]   = target_rad

    def _home_single_joint(self, idx: int) -> None:
        """Send HOME command for a single joint via <HOMEx> convention."""
        self._stop_all()
        # The firmware doesn't yet have per-joint home commands — send the full
        # HOME sequence but document it. When per-joint HOME is implemented,
        # change this to f"<HOME{idx+1}>" or similar.
        self.send_command.emit(f"<HOME{idx+1}>")

    def _step_and_send(self) -> None:
        dt = 0.04  # 40 ms timer → 0.04 s
        for i in range(N_JOINTS):
            if self._target_vel[i] == 0.0:
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
