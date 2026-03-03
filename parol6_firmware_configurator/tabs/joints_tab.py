"""
joints_tab.py — Per-joint configuration table (pins, gear, limit switches).
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QAbstractItemView
)
from PyQt6.QtCore import pyqtSignal, Qt
from core.config_model import JointConfig, LimitSwitchConfig

# Columns order
COLS = [
    "Enabled", "STEP Pin", "DIR Pin", "Enc Pin",
    "Gear Ratio", "Microstep", "Dir Inv", "ROS Inv",
    "Max Vel\n(rad/s)", "Max I\n(mA)", "Kp",
    "Limit\nType", "Limit\nPin", "Limit\nPolarity"
]

LIMIT_TYPES    = ["NONE", "MECHANICAL", "INDUCTIVE_NPN", "INDUCTIVE_PNP"]
LIMIT_POLARITY = ["FALLING", "RISING"]


class JointsTab(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

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
        root.addWidget(self.table)

        # Steps-per-rad info row
        self.steps_label = QLabel()
        self.steps_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        root.addWidget(self.steps_label)

        self._widgets: list[dict] = []  # per-row widget refs

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

    def _make_combo(self, items, current):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(current)
        c.setStyleSheet("background:#1e1e2e; color:#cdd6f4; border:none;")
        c.currentTextChanged.connect(self._on_cell_changed)
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

    def load(self, joints: list[JointConfig]) -> None:
        self._updating = True
        self.table.clearContents()
        self._widgets.clear()

        for row, j in enumerate(joints):
            w = {}
            # Enabled
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
            w["lim_type"] = self._make_combo(LIMIT_TYPES, j.limit.switch_type)
            w["lim_pin"]  = self._make_spin(j.limit.pin, 0, 55)
            w["lim_pol"]  = self._make_combo(LIMIT_POLARITY, j.limit.polarity)

            self._widgets.append(w)
            cells = [
                self._center_widget(w["enabled"]),
                w["step_pin"], w["dir_pin"], w["enc_pin"],
                w["gear"], w["micro"],
                self._center_widget(w["dir_inv"]),
                self._center_widget(w["ros_inv"]),
                w["maxvel"], w["maxcur"], w["kp"],
                w["lim_type"], w["lim_pin"], w["lim_pol"],
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
            j.limit.switch_type = w["lim_type"].currentText()
            j.limit.pin         = w["lim_pin"].value()
            j.limit.polarity    = w["lim_pol"].currentText()
            j.limit.enabled     = j.limit.switch_type != "NONE"
