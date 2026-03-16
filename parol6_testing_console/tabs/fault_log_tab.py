"""
fault_log_tab.py — Keeps a rolling history of fault events with export.
"""
from __future__ import annotations
import csv
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from core.gui_theme import QPushButton
from PyQt6.QtGui import QColor

MAX_FAULTS = 100
COLS = ["Time (s)", "Supervisor State", "Message", "J1 vel", "J2 vel",
        "J3 vel", "J4 vel", "J5 vel", "J6 vel", "ISR µs"]


class FaultLogTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._faults: list[dict] = []
        self._t0 = datetime.now()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        hdr = QHBoxLayout()
        title = QLabel("⚠️  Fault Log")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        hdr.addWidget(title)
        hdr.addStretch()

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._clear)
        export_btn = QPushButton("📥 Export CSV")
        export_btn.clicked.connect(self._export)
        hdr.addWidget(clear_btn)
        hdr.addWidget(export_btn)
        root.addLayout(hdr)

        hint = QLabel(
            "📋 <b>FAULT</b> (red) = latched — requires firmware restart or reflash to clear.  "
            "<b>SOFT_ESTOP</b> (yellow) = recoverable — send <code>&lt;ENABLE&gt;</code> in 💬 Serial tab "
            "or click <b>🔓 CLEAR FAULT / ENABLE</b> in 🕹 Jog tab.  "
            "Common causes: velocity limit exceeded, command timeout (firmware stopped getting commands), "
            "limit switch triggered outside homing.  "
            "<b>REBOOTING_TO_DFU</b> (blue) = informational — STM32 is entering bootloader for flashing.  "
            "<b>Export CSV</b> saves timestamp, state, and velocities for post-mortem analysis."
        )
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "background:#2a0a0a; border:1px solid #f38ba8; border-radius:6px; "
            "color:#cdd6f4; font-size:11px; padding:6px 10px; margin-bottom:4px;"
        )
        root.addWidget(hint)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget { gridline-color: #45475a; }
            QHeaderView::section { background: #313244; color: #cdd6f4; }
        """)
        root.addWidget(self.table)

    def ingest_telemetry(self, pkt: dict) -> None:
        """Check packet for fault state; record if present."""
        state = pkt.get("supervisor_state", "")
        if state not in ("FAULT", "SOFT_ESTOP"):
            return
        self._record_fault(state, "Telemetry fault", pkt)

    def record_serial_fault(self, line: str, pkt: dict | None = None) -> None:
        """Call this when a FAULT line is detected in raw serial."""
        if "REBOOTING_TO_DFU" in line:
            # Informational only — DFU reboot triggered; don't log as fault
            return
        self._record_fault("FAULT", line, pkt or {})

    def _record_fault(self, state: str, msg: str, pkt: dict) -> None:
        elapsed = (datetime.now() - self._t0).total_seconds()
        row_data = {
            "time": f"{elapsed:.3f}",
            "state": state,
            "msg": msg[:60],
            "vel": pkt.get("vel", [0]*6),
            "isr": pkt.get("isr_us", "—"),
        }
        self._faults.append(row_data)
        if len(self._faults) > MAX_FAULTS:
            self._faults.pop(0)
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self._faults))
        for row, f in enumerate(self._faults):
            vals = ([f["time"], f["state"], f["msg"]]
                    + [f"{v:.3f}" for v in f["vel"]]
                    + [str(f["isr"])])
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if f["state"] == "FAULT":
                    item.setForeground(QColor("#f38ba8"))
                else:
                    item.setForeground(QColor("#f9e2af"))
                self.table.setItem(row, col, item)
        self.table.scrollToBottom()

    def _clear(self) -> None:
        self._faults.clear()
        self.table.setRowCount(0)
        self._t0 = datetime.now()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export fault log", f"faults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLS)
            for fault in self._faults:
                writer.writerow(
                    [fault["time"], fault["state"], fault["msg"]]
                    + [f"{v:.4f}" for v in fault["vel"]]
                    + [str(fault["isr"])]
                )
