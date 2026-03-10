"""
joints_tab.py — Per-joint configuration table (pins, gear, limit switches).
Columns include pull-resistor selection; a suggestion panel at the bottom
shows wiring guidance based on the sensor type chosen for each joint.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QAbstractItemView, QFrame, QGroupBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
from core.config_model import JointConfig, LimitSwitchConfig

# Columns order — Pull added after Polarity
COLS = [
    "Enabled", "STEP Pin", "DIR Pin", "Enc Pin",
    "Gear Ratio", "Microstep", "Dir Inv", "ROS Inv",
    "Max Vel\n(rad/s)", "Max Cur\n(mA)", "Kp", "Ki", "Max I\n(windup)",
    "Home\nOffset(rad)",
    "Limit\nType", "Limit\nPin", "Limit\nPolarity", "Limit\nPull"
]

LIMIT_TYPES    = ["NONE", "MECHANICAL", "INDUCTIVE_NPN", "INDUCTIVE_PNP"]
LIMIT_POLARITY = ["FALLING", "RISING"]
LIMIT_PULL     = ["INPUT_PULLUP", "INPUT_PULLDOWN", "INPUT"]

# --- Auto-suggest table for (Type → pull, polarity) ----------------------
# Inductive NPN sensors have an open-collector output normally pulled HIGH
# through the optocoupler; the Teensy sees HIGH=inactive, LOW=triggered.
# Inductive PNP sensors pull HIGH when triggered; Teensy sees LOW=inactive, HIGH=triggered.
_DEFAULTS = {
    "NONE":          ("INPUT_PULLUP",   "FALLING"),
    "MECHANICAL":    ("INPUT_PULLUP",   "FALLING"),   # NC switch ← most common
    "INDUCTIVE_NPN": ("INPUT_PULLUP",   "FALLING"),   # open-collector, active LOW
    "INDUCTIVE_PNP": ("INPUT_PULLDOWN", "RISING"),    # totem-pole, active HIGH
}

_SUGGESTIONS = {
    "NONE": "",
    "MECHANICAL": (
        "🔘 <b>Mechanical switch (NC recommended)</b><br>"
        "• Wire: COM → Teensy pin, NC → GND<br>"
        "• Pull: <b>INPUT_PULLUP</b> (internal 47 kΩ keeps pin HIGH when open)<br>"
        "• Polarity: <b>FALLING</b> — pin drops LOW when switch closes<br>"
        "• ⚠ Debounce is applied in firmware (3-sample, 3 ms window)<br>"
        "• Optocoupler variant: connect as NPN below"
    ),
    "INDUCTIVE_NPN": (
        "🔵 <b>Inductive NPN sensor (open-collector)</b> — most common PAROL6 setup<br>"
        "• Sensor wiring: Brown=VCC (5–24V), Blue=GND, Black=Signal (to optocoupler IN)<br>"
        "• Optocoupler output (collector): Teensy pin, emitter: GND<br>"
        "• Pull: <b>INPUT_PULLUP</b> — Teensy pin pulled HIGH (≈3.3V) when sensor is inactive<br>"
        "• Polarity: <b>FALLING</b> — pin is pulled LOW through optocoupler when triggered<br>"
        "• ✅ Optocouplers invert AND level-shift; Teensy safe from 24V spikes"
    ),
    "INDUCTIVE_PNP": (
        "🟠 <b>Inductive PNP sensor (totem-pole, sourcing)</b><br>"
        "• Sensor wiring: Brown=VCC, Blue=GND, Black=Signal → optocoupler LED+<br>"
        "• Optocoupler output: collector → VCC (3.3V), emitter → Teensy pin<br>"
        "• Pull: <b>INPUT_PULLDOWN</b> — pin stays LOW when sensor is inactive<br>"
        "• Polarity: <b>RISING</b> — optocoupler pulls pin HIGH when triggered<br>"
        "• ⚠ Verify optocoupler CTR and resistor value for reliable switching"
    ),
}


class JointsTab(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        title = QLabel("🔩  Joint Configuration")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)

        info = QLabel(
            "All hardware constants come from the PAROL6 legacy STM32 firmware. "
            "Gear ratios and microsteps are pre-populated — only change if you modify hardware."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#a6adc8; font-size:11px;")
        root.addWidget(info)

        self.table = QTableWidget(6, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setVerticalHeaderLabels([f"J{i+1}" for i in range(6)])
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget { gridline-color: #45475a; }
            QTableWidget::item:selected { background: #45475a; }
            QHeaderView::section { background: #313244; color: #cdd6f4; padding: 4px; }
        """)
        # Emit changed + update suggestion when selection changes
        self.table.currentCellChanged.connect(self._on_selection_changed)
        root.addWidget(self.table, stretch=1)

        # Steps-per-rad info row
        self.steps_label = QLabel()
        self.steps_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        root.addWidget(self.steps_label)

        # ── Limit Switch Suggestion Panel ──────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #45475a;")
        root.addWidget(sep)

        hint_group = QGroupBox("💡  Limit Switch Wiring Guide")
        hint_group.setStyleSheet("""
            QGroupBox {
                color: #fab387;
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 6px;
                padding: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)
        hint_lay = QVBoxLayout(hint_group)
        hint_lay.setContentsMargins(8, 8, 8, 8)

        self._hint_label = QLabel(
            "<i style='color:#6c7086;'>Click any row to see sensor-specific wiring advice.</i>"
        )
        self._hint_label.setTextFormat(Qt.TextFormat.RichText)
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("font-size: 12px; color: #cdd6f4; line-height: 1.6;")
        hint_lay.addWidget(self._hint_label)

        # Quick reference table
        ref_label = QLabel(
            "<table style='color:#cdd6f4; font-size:11px; border-spacing:4px;'>"
            "<tr style='color:#cba6f7;'><th align='left'>Sensor Type</th>"
            "<th align='left'>Suggested Pull</th><th align='left'>Polarity</th>"
            "<th align='left'>Wiring</th></tr>"
            "<tr><td>Mechanical (NC)</td><td>INPUT_PULLUP</td><td>FALLING</td>"
            "<td>COM→Pin, NC→GND</td></tr>"
            "<tr><td>Inductive NPN</td><td>INPUT_PULLUP</td><td>FALLING</td>"
            "<td>Signal→optocoupler→Pin, emitter→GND</td></tr>"
            "<tr><td>Inductive PNP</td><td>INPUT_PULLDOWN</td><td>RISING</td>"
            "<td>Signal→optocoupler→3.3V, collector→Pin</td></tr>"
            "</table>"
        )
        ref_label.setTextFormat(Qt.TextFormat.RichText)
        ref_label.setStyleSheet("margin-top: 6px;")
        hint_lay.addWidget(ref_label)

        root.addWidget(hint_group)

        self._widgets: list[dict] = []  # per-row widget refs

    # ------------------------------------------------------------------
    def _make_spin(self, value, lo, hi, step=1, decimals=None):
        if decimals is not None:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
            w.setSingleStep(step)
        else:
            w = QSpinBox()
            w.setSingleStep(step)
        w.setRange(lo, hi)
        w.setValue(value)
        w.setStyleSheet("background:#1e1e2e; border:none; color:#cdd6f4;")
        w.valueChanged.connect(self._on_cell_changed)
        return w

    def _make_combo(self, items, current, on_change=None):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(current)
        c.setStyleSheet("background:#1e1e2e; color:#cdd6f4; border:none;")
        c.currentTextChanged.connect(self._on_cell_changed)
        if on_change:
            c.currentTextChanged.connect(on_change)
        return c

    def _make_check(self, value: bool):
        ch = QCheckBox()
        ch.setChecked(value)
        ch.stateChanged.connect(self._on_cell_changed)
        return ch

    def _center_widget(self, w: QWidget) -> QWidget:
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.addWidget(w)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setContentsMargins(2, 0, 2, 0)
        return container

    def _on_cell_changed(self) -> None:
        if not self._updating:
            self.changed.emit()

    def _on_selection_changed(self, row, *_) -> None:
        """Update the wiring suggestion when a row is selected."""
        if row < 0 or row >= len(self._widgets):
            return
        sensor_type = self._widgets[row]["lim_type"].currentText()
        suggestion = _SUGGESTIONS.get(sensor_type, "")
        if suggestion:
            self._hint_label.setText(
                f"<b style='color:#fab387;'>J{row+1} — {sensor_type}</b><br>{suggestion}"
            )
        else:
            self._hint_label.setText(
                "<i style='color:#6c7086;'>No limit switch configured for this joint.</i>"
            )

    def _on_type_changed(self, row: int, sensor_type: str) -> None:
        """Auto-suggest Pull and Polarity when sensor type changes."""
        if self._updating or row >= len(self._widgets):
            return
        pull, polarity = _DEFAULTS.get(sensor_type, ("INPUT_PULLUP", "FALLING"))
        self._updating = True
        self._widgets[row]["lim_pull"].setCurrentText(pull)
        self._widgets[row]["lim_pol"].setCurrentText(polarity)
        self._updating = False
        # Update hint
        self._on_selection_changed(row)
        self.changed.emit()

    # ------------------------------------------------------------------
    def load(self, joints: list[JointConfig]) -> None:
        self._updating = True
        self.table.clearContents()
        self._widgets.clear()

        for row, j in enumerate(joints):
            w = {}
            w["enabled"]  = self._make_check(j.enabled)
            w["step_pin"] = self._make_spin(j.step_pin,  0, 55)
            w["dir_pin"]  = self._make_spin(j.dir_pin,   0, 55)
            w["enc_pin"]  = self._make_spin(j.encoder_pin, 0, 55)
            w["gear"]     = self._make_spin(j.gear_ratio, 0.1, 100.0, 0.1, 4)
            w["micro"]    = self._make_spin(j.microsteps, 1, 64)
            w["dir_inv"]  = self._make_check(j.dir_invert)
            w["ros_inv"]  = self._make_check(j.ros_dir_invert)
            w["maxvel"]   = self._make_spin(j.max_vel_rad_s, 0.1, 30.0, 0.5, 2)
            w["maxcur"]   = self._make_spin(j.max_current_ma, 0, 3000, 50)
            w["kp"]       = self._make_spin(j.kp, 0.0, 50.0, 0.5, 2)
            w["ki"]       = self._make_spin(j.ki, 0.0, 50.0, 0.1, 2)
            w["max_i"]    = self._make_spin(j.max_integral, 0.0, 100.0, 1.0, 2)
            w["home"]     = self._make_spin(j.home_offset_rad, -10.0, 10.0, 0.1, 4)

            # Limit columns — type change auto-suggests pull/polarity
            _row = row  # capture for lambda
            w["lim_type"] = self._make_combo(
                LIMIT_TYPES, j.limit.switch_type,
                on_change=lambda txt, r=_row: self._on_type_changed(r, txt)
            )
            w["lim_pin"]  = self._make_spin(j.limit.pin, 0, 55)
            w["lim_pol"]  = self._make_combo(LIMIT_POLARITY, j.limit.polarity)
            w["lim_pull"] = self._make_combo(LIMIT_PULL, j.limit.pull)

            self._widgets.append(w)
            cells = [
                self._center_widget(w["enabled"]),
                w["step_pin"], w["dir_pin"], w["enc_pin"],
                w["gear"], w["micro"],
                self._center_widget(w["dir_inv"]),
                self._center_widget(w["ros_inv"]),
                w["maxvel"], w["maxcur"], w["kp"], w["ki"], w["max_i"], w["home"],
                w["lim_type"], w["lim_pin"], w["lim_pol"], w["lim_pull"],
            ]
            for col, widget in enumerate(cells):
                self.table.setCellWidget(row, col, widget)

        self._updating = False
        self._update_steps_info(joints)

    def _update_steps_info(self, joints: list[JointConfig]) -> None:
        import math
        parts = []
        for i, j in enumerate(joints):
            s = (j.steps_per_rev * j.microsteps * j.gear_ratio) / (2 * math.pi)
            parts.append(f"J{i+1}={s:.0f}")
        self.steps_label.setText("Steps/rad: " + "  |  ".join(parts))

    def save(self, joints: list[JointConfig]) -> None:
        for row, (j, w) in enumerate(zip(joints, self._widgets)):
            j.enabled         = w["enabled"].isChecked()
            j.step_pin        = w["step_pin"].value()
            j.dir_pin         = w["dir_pin"].value()
            j.encoder_pin     = w["enc_pin"].value()
            j.gear_ratio      = w["gear"].value()
            j.microsteps      = w["micro"].value()
            j.dir_invert      = w["dir_inv"].isChecked()
            j.ros_dir_invert  = w["ros_inv"].isChecked()
            j.max_vel_rad_s   = w["maxvel"].value()
            j.max_current_ma  = w["maxcur"].value()
            j.kp              = w["kp"].value()
            j.ki              = w["ki"].value()
            j.max_integral    = w["max_i"].value()
            j.home_offset_rad = w["home"].value()
            j.limit.switch_type = w["lim_type"].currentText()
            j.limit.pin         = w["lim_pin"].value()
            j.limit.polarity    = w["lim_pol"].currentText()
            j.limit.pull        = w["lim_pull"].currentText()
            j.limit.enabled     = j.limit.switch_type != "NONE"
