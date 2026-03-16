"""Live telemetry oscilloscope tab."""
from __future__ import annotations

from collections import deque

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, background="#11111b", foreground="#cdd6f4")
    PYQTGRAPH_OK = True
except ImportError:
    PYQTGRAPH_OK = False


JOINT_COLORS = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89dceb", "#cba6f7"]
N_JOINTS = 6
BUF_SIZE = 2000


class PlotTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._paused = False
        self._t_buf: deque[float] = deque(maxlen=BUF_SIZE)
        self._pos_bufs = [deque(maxlen=BUF_SIZE) for _ in range(N_JOINTS)]
        self._vel_bufs = [deque(maxlen=BUF_SIZE) for _ in range(N_JOINTS)]
        self._pwm_bufs = [deque(maxlen=BUF_SIZE) for _ in range(N_JOINTS)]
        self._isr_buf: deque[float] = deque(maxlen=BUF_SIZE)
        self._t0 = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        if not PYQTGRAPH_OK:
            root.addWidget(QLabel("pyqtgraph not installed. Run: pip install pyqtgraph"))
            return

        toolbar = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self._toggle_pause)
        toolbar.addWidget(self.pause_btn)

        toolbar.addWidget(QLabel("Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 60)
        self.window_spin.setValue(10)
        self.window_spin.setSuffix(" s")
        toolbar.addWidget(self.window_spin)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(clear_btn)

        self.csv_btn = QPushButton("Save CSV")
        self.csv_btn.clicked.connect(self._export_csv)
        toolbar.addWidget(self.csv_btn)

        toolbar.addStretch()

        self._pos_checks = []
        self._vel_checks = []
        self._pwm_checks = []
        for i in range(N_JOINTS):
            color = JOINT_COLORS[i]
            pos_box = QCheckBox(f"J{i+1} pos")
            pos_box.setChecked(True)
            pos_box.setStyleSheet(f"color:{color};")
            toolbar.addWidget(pos_box)
            self._pos_checks.append(pos_box)

        toolbar.addWidget(QLabel("|"))
        for i in range(N_JOINTS):
            color = JOINT_COLORS[i]
            vel_box = QCheckBox(f"J{i+1} vel")
            vel_box.setChecked(True)
            vel_box.setStyleSheet(f"color:{color};")
            toolbar.addWidget(vel_box)
            self._vel_checks.append(vel_box)

        toolbar.addWidget(QLabel("|"))
        for i in range(N_JOINTS):
            color = JOINT_COLORS[i]
            pwm_box = QCheckBox(f"J{i+1} pwm")
            pwm_box.setStyleSheet(f"color:{color};")
            toolbar.addWidget(pwm_box)
            self._pwm_checks.append(pwm_box)

        root.addLayout(toolbar)

        self._pw_pos = pg.PlotWidget(title="Joint Positions")
        self._pw_vel = pg.PlotWidget(title="Joint Velocities")
        self._pw_pwm = pg.PlotWidget(title="PWM Output")
        self._pw_isr = pg.PlotWidget(title="ISR Time")
        self._pw_vel.setXLink(self._pw_pos)
        self._pw_pwm.setXLink(self._pw_pos)
        self._pw_isr.setXLink(self._pw_pos)

        for pw in (self._pw_pos, self._pw_vel, self._pw_pwm, self._pw_isr):
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel("bottom", "Time (s)")
            pw.addLegend(offset=(10, 10))

        self._pw_pwm.setYRange(0, 100, padding=0.05)
        self._isr_limit = pg.InfiniteLine(
            pos=25,
            angle=0,
            pen=pg.mkPen("#f38ba8", width=1, style=Qt.PenStyle.DashLine),
            label="25us limit",
            labelOpts={"color": "#f38ba8"},
        )
        self._pw_isr.addItem(self._isr_limit)

        self._pos_curves = [
            self._pw_pos.plot(pen=pg.mkPen(JOINT_COLORS[i], width=2), name=f"J{i+1}") for i in range(N_JOINTS)
        ]
        self._vel_curves = [
            self._pw_vel.plot(
                pen=pg.mkPen(JOINT_COLORS[i], width=2, style=Qt.PenStyle.DashLine),
                name=f"J{i+1}",
            )
            for i in range(N_JOINTS)
        ]
        self._pwm_curves = [
            self._pw_pwm.plot(
                pen=pg.mkPen(JOINT_COLORS[i], width=2, style=Qt.PenStyle.DotLine),
                name=f"J{i+1}",
            )
            for i in range(N_JOINTS)
        ]
        self._isr_curve = self._pw_isr.plot(pen=pg.mkPen("#f9e2af", width=2), name="ISR us")

        root.addWidget(self._pw_pos, stretch=3)
        root.addWidget(self._pw_vel, stretch=3)
        root.addWidget(self._pw_pwm, stretch=3)
        root.addWidget(self._pw_isr, stretch=2)

    def ingest(self, pkt: dict) -> None:
        if self._paused or not PYQTGRAPH_OK:
            return

        import time

        if not self._t_buf:
            self._t0 = time.monotonic()
        t = time.monotonic() - self._t0
        self._t_buf.append(t)
        pwm_raw = pkt.get("pwm", [0.0] * N_JOINTS)

        for i in range(N_JOINTS):
            self._pos_bufs[i].append(pkt["pos"][i] if i < len(pkt["pos"]) else 0.0)
            self._vel_bufs[i].append(pkt["vel"][i] if i < len(pkt["vel"]) else 0.0)
            self._pwm_bufs[i].append(pwm_raw[i] if i < len(pwm_raw) else 0.0)

        self._isr_buf.append(pkt.get("isr_us") or 0.0)

    def refresh(self) -> None:
        if not PYQTGRAPH_OK or self._paused or len(self._t_buf) < 2:
            return

        window = self.window_spin.value()
        t_arr = np.array(self._t_buf)
        mask = t_arr >= (t_arr[-1] - window)
        t_w = t_arr[mask]

        for i in range(N_JOINTS):
            pos_arr = np.array(self._pos_bufs[i])[mask]
            vel_arr = np.array(self._vel_bufs[i])[mask]
            pwm_arr = np.array(self._pwm_bufs[i])[mask]
            if pwm_arr.size > 0 and pwm_arr.max() <= 1.1:
                pwm_arr = pwm_arr * 100.0

            self._pos_curves[i].setData(t_w if self._pos_checks[i].isChecked() else [], pos_arr if self._pos_checks[i].isChecked() else [])
            self._vel_curves[i].setData(t_w if self._vel_checks[i].isChecked() else [], vel_arr if self._vel_checks[i].isChecked() else [])
            self._pwm_curves[i].setData(t_w if self._pwm_checks[i].isChecked() else [], pwm_arr if self._pwm_checks[i].isChecked() else [])

        isr_arr = np.array(self._isr_buf)[mask]
        self._isr_curve.setData(t_w, isr_arr)

    def clear(self) -> None:
        self._t_buf.clear()
        for buf in self._pos_bufs + self._vel_bufs + self._pwm_bufs:
            buf.clear()
        self._isr_buf.clear()

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self.pause_btn.setText("Resume" if checked else "Pause")

    def _export_csv(self) -> None:
        if not self._t_buf:
            return

        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import csv

        path, _ = QFileDialog.getSaveFileName(self, "Save Telemetry Data", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["Time_s"]
                for i in range(N_JOINTS):
                    header.extend([f"J{i+1}_Pos", f"J{i+1}_Vel", f"J{i+1}_PWM"])
                header.append("ISR_us")
                writer.writerow(header)

                for idx in range(len(self._t_buf)):
                    row = [self._t_buf[idx]]
                    for j in range(N_JOINTS):
                        row.append(self._pos_bufs[j][idx] if idx < len(self._pos_bufs[j]) else 0.0)
                        row.append(self._vel_bufs[j][idx] if idx < len(self._vel_bufs[j]) else 0.0)
                        row.append(self._pwm_bufs[j][idx] if idx < len(self._pwm_bufs[j]) else 0.0)
                    row.append(self._isr_buf[idx] if idx < len(self._isr_buf) else 0.0)
                    writer.writerow(row)

            QMessageBox.information(self, "Export Complete", f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save CSV:\n{e}")
