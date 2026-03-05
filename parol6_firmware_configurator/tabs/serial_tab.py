"""
serial_tab.py — Serial monitor: connect, display, send, filter, timestamps.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QTextEdit, QLabel, QCheckBox, QSplitter, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt, QDateTime
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont
from core.serial_monitor import SerialWorker, list_serial_ports


class SerialTab(QWidget):
    """Full serial monitor with connect/disconnect, send, filter, timestamps."""

    # Signals for main window's status bar
    connected_changed = pyqtSignal(bool)
    packet_rate_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: SerialWorker | None = None
        self._t0 = QDateTime.currentMSecsSinceEpoch()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Toolbar ───────────────────────────────────────────────────
        toolbar = QHBoxLayout()

        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_connection)
        toolbar.addWidget(self.connect_btn)

        self.ts_check = QCheckBox("Timestamps")
        self.ts_check.setChecked(True)
        toolbar.addWidget(self.ts_check)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter (e.g. ACK or FAULT)…")
        self.filter_edit.setMaximumWidth(200)
        toolbar.addWidget(QLabel("Show only:"))
        toolbar.addWidget(self.filter_edit)

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(clear_btn)

        self.autoscroll = QCheckBox("Autoscroll")
        self.autoscroll.setChecked(True)
        toolbar.addWidget(self.autoscroll)
        toolbar.addStretch()

        root.addLayout(toolbar)

        # ── Output ────────────────────────────────────────────────────
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Monospace", 10))
        self.output.setStyleSheet("""
            QTextEdit {
                background: #11111b;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
            }
        """)
        root.addWidget(self.output, stretch=1)

        # ── Send row ──────────────────────────────────────────────────
        send_row = QHBoxLayout()
        self.send_edit = QLineEdit()
        self.send_edit.setPlaceholderText("Send command…  (Enter to send)")
        self.send_edit.returnPressed.connect(self._send)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send)
        send_row.addWidget(self.send_edit, stretch=1)
        send_row.addWidget(send_btn)
        root.addLayout(send_row)

    # ------------------------------------------------------------------
    def connect_to(self, port: str, baud: int) -> None:
        """Start the serial worker for the given port/baud."""
        if self._worker:
            self._worker.stop()
            self._worker = None

        if not port:
            self._append("[SERIAL] No port selected.", color="#f38ba8")
            return

        self._worker = SerialWorker(port, baud)
        self._worker.raw_line.connect(self._on_line)
        self._worker.error_msg.connect(lambda m: self._append(m, "#f38ba8"))
        self._worker.connected.connect(self._on_connected)
        self._worker.packet_rate.connect(self.packet_rate_changed)
        self._worker.start()

    def disconnect(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("🔌 Connect")

    def send_raw(self, text: str) -> None:
        if self._worker:
            self._worker.send(text)

    # ------------------------------------------------------------------
    def _toggle_connection(self, checked: bool) -> None:
        if checked:
            # Get port/baud from comms tab via main window signal
            self.connect_btn.setText("⏹ Disconnect")
            self.connect_requested.emit()  # type: ignore[attr-defined]
        else:
            self.disconnect()

    def _send(self) -> None:
        text = self.send_edit.text().strip()
        if text and self._worker:
            self._worker.send(text)
            self._append(f">> {text}", color="#89b4fa")
            self.send_edit.clear()

    def _on_connected(self, ok: bool) -> None:
        self.connected_changed.emit(ok)
        colour = "#a6e3a1" if ok else "#f38ba8"
        msg = "Connected" if ok else "Disconnected"
        self._append(f"[SERIAL] {msg}", color=colour)
        self.connect_btn.setChecked(ok)
        self.connect_btn.setText("⏹ Disconnect" if ok else "🔌 Connect")

    def _on_line(self, line: str) -> None:
        flt = self.filter_edit.text().strip()
        if flt and flt.lower() not in line.lower():
            return
        ts = ""
        if self.ts_check.isChecked():
            elapsed = (QDateTime.currentMSecsSinceEpoch() - self._t0) / 1000.0
            ts = f"[+{elapsed:7.3f}s] "
        colour = "#f9e2af" if "FAULT" in line or "ERR" in line.upper() else "#cdd6f4"
        self._append(ts + line, colour)

    def _append(self, text: str, color: str = "#cdd6f4") -> None:
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text + "\n")
        if self.autoscroll.isChecked():
            self.output.setTextCursor(cursor)
            self.output.ensureCursorVisible()

    def _clear(self) -> None:
        self.output.clear()
        self._t0 = QDateTime.currentMSecsSinceEpoch()

    # Declared so main.py can attach connections:
    connect_requested = pyqtSignal()
