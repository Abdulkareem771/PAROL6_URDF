"""
plot_tab.py — Real-time oscilloscope for joint position, velocity, ISR time.
Uses pyqtgraph for GPU-accelerated rendering.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
    QPushButton, QSpinBox, QGroupBox, QScrollArea, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background="#11111b", foreground="#cdd6f4")
    PYQTGRAPH_OK = True
except ImportError:
    PYQTGRAPH_OK = False

JOINT_COLORS = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89dceb", "#cba6f7"]
N_JOINTS = 6
BUF_SIZE = 2000  # samples kept in ring buffer


class PlotTab(QWidget):
    """Live oscilloscope: position, velocity, ISR time from <ACK,...> packets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paused = False
        self._t_buf: deque[float] = deque(maxlen=BUF_SIZE)
        self._pos_bufs = [deque(maxlen=BUF_SIZE) for _ in range(N_JOINTS)]
        self._vel_bufs = [deque(maxlen=BUF_SIZE) for _ in range(N_JOINTS)]
        self._isr_buf:  deque[float] = deque(maxlen=BUF_SIZE)
        self._t0 = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        if not PYQTGRAPH_OK:
            root.addWidget(QLabel(
                "pyqtgraph not installed.\nRun: pip install pyqtgraph\nthen restart."))
            return

        # ── Toolbar ───────────────────────────────────────────────────
        tb = QHBoxLayout()

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self._toggle_pause)
        tb.addWidget(self.pause_btn)

        tb.addWidget(QLabel("Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 60)
        self.window_spin.setValue(10)
        self.window_spin.setSuffix(" s")
        tb.addWidget(self.window_spin)

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._clear_data)
        tb.addWidget(clear_btn)

        tb.addStretch()

        # Channel checkboxes
        self._pos_checks = []
        self._vel_checks = []
        for i in range(N_JOINTS):
            c = JOINT_COLORS[i]
            pc = QCheckBox(f"J{i+1} pos")
            pc.setChecked(True)
            pc.setStyleSheet(f"color:{c};")
            self._pos_checks.append(pc)
            tb.addWidget(pc)

        tb.addWidget(QLabel("|"))
        for i in range(N_JOINTS):
            c = JOINT_COLORS[i]
            vc = QCheckBox(f"J{i+1} vel")
            vc.setChecked(False)
            vc.setStyleSheet(f"color:{c}; font-style:italic;")
            self._vel_checks.append(vc)
            tb.addWidget(vc)

        root.addLayout(tb)

        # ── Plot area ─────────────────────────────────────────────────
        self._pw_pos = pg.PlotWidget(title="Joint Positions (rad)")
        self._pw_vel = pg.PlotWidget(title="Joint Velocities (rad/s)")
        self._pw_isr = pg.PlotWidget(title="ISR Time (µs)")

        for pw in (self._pw_pos, self._pw_vel, self._pw_isr):
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel("bottom", "Time (s)")
            pw.addLegend(offset=(10, 10))

        self._pw_isr.setLabel("bottom", "Time (s)")
        # ISR budget line
        self._isr_limit = pg.InfiniteLine(
            pos=25, angle=0, pen=pg.mkPen("#f38ba8", width=1, style=Qt.PenStyle.DashLine),
            label="25µs limit", labelOpts={"color": "#f38ba8"})
        self._pw_isr.addItem(self._isr_limit)

        # Create curves
        self._pos_curves = [
            self._pw_pos.plot(pen=pg.mkPen(JOINT_COLORS[i], width=2),
                              name=f"J{i+1}")
            for i in range(N_JOINTS)
        ]
        self._vel_curves = [
            self._pw_vel.plot(pen=pg.mkPen(JOINT_COLORS[i], width=2, style=Qt.PenStyle.DashLine),
                              name=f"J{i+1}")
            for i in range(N_JOINTS)
        ]
        self._isr_curve = self._pw_isr.plot(pen=pg.mkPen("#f9e2af", width=2), name="ISR µs")

        root.addWidget(self._pw_pos, stretch=3)
        root.addWidget(self._pw_vel, stretch=3)
        root.addWidget(self._pw_isr, stretch=2)

    # ------------------------------------------------------------------
    def ingest(self, pkt: dict) -> None:
        """Call this whenever a parsed ACK packet arrives."""
        if self._paused or not PYQTGRAPH_OK:
            return
        import time
        if not self._t_buf:
            self._t0 = time.monotonic()
        t = time.monotonic() - self._t0
        self._t_buf.append(t)
        for i in range(N_JOINTS):
            self._pos_bufs[i].append(pkt["pos"][i] if i < len(pkt["pos"]) else 0.0)
            self._vel_bufs[i].append(pkt["vel"][i] if i < len(pkt["vel"]) else 0.0)
        self._isr_buf.append(pkt.get("isr_us") or 0.0)

    def refresh(self) -> None:
        """Called by QTimer ~20 Hz to update plots."""
        if not PYQTGRAPH_OK or self._paused or len(self._t_buf) < 2:
            return
        window = self.window_spin.value()
        t_arr = np.array(self._t_buf)
        mask = t_arr >= (t_arr[-1] - window)
        t_w = t_arr[mask]

        for i in range(N_JOINTS):
            show_pos = self._pos_checks[i].isChecked()
            p_arr = np.array(self._pos_bufs[i])[mask]
            self._pos_curves[i].setData(t_w if show_pos else [], p_arr if show_pos else [])

            show_vel = self._vel_checks[i].isChecked()
            v_arr = np.array(self._vel_bufs[i])[mask]
            self._vel_curves[i].setData(t_w if show_vel else [], v_arr if show_vel else [])

        isr_arr = np.array(self._isr_buf)[mask]
        self._isr_curve.setData(t_w, isr_arr)

    def _toggle_pause(self, checked: bool) -> None:
        self._paused = checked
        self.pause_btn.setText("▶ Resume" if checked else "⏸ Pause")

    def _clear_data(self) -> None:
        self._t_buf.clear()
        for b in self._pos_bufs + self._vel_bufs:
            b.clear()
        self._isr_buf.clear()
