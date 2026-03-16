"""Serial monitor tab."""
from __future__ import annotations

from PyQt6.QtCore import QDateTime, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SerialTab(QWidget):
    connect_requested = pyqtSignal()
    connected_changed = pyqtSignal(bool)
    packet_rate_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = QDateTime.currentMSecsSinceEpoch()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_connection)
        toolbar.addWidget(self.connect_btn)

        self.ts_check = QCheckBox("Timestamps")
        self.ts_check.setChecked(True)
        toolbar.addWidget(self.ts_check)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter text")
        self.filter_edit.setMaximumWidth(220)
        toolbar.addWidget(QLabel("Show only:"))
        toolbar.addWidget(self.filter_edit)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(clear_btn)

        self.autoscroll = QCheckBox("Autoscroll")
        self.autoscroll.setChecked(True)
        toolbar.addWidget(self.autoscroll)
        toolbar.addStretch()
        root.addLayout(toolbar)

        hint = QLabel(
            "Raw serial terminal. Use it for `<HOME>`, `<ENABLE>`, custom packets, and direct board debugging."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "background:#1f2430; border:1px solid #4c566a; border-radius:6px; padding:6px 10px;"
        )
        root.addWidget(hint)

        from PyQt6.QtWidgets import QGroupBox
        macro_group = QGroupBox("Quick Macros")
        macro_layout = QHBoxLayout(macro_group)
        self.btn_reset_faults = QPushButton("RST Faults")
        self.btn_reset_faults.clicked.connect(lambda: self.window().send_serial_text("<RESET>"))
        self.btn_enable = QPushButton("Enable")
        self.btn_enable.clicked.connect(lambda: self.window().send_serial_text("<ENABLE>"))
        self.btn_disable = QPushButton("Disable")
        self.btn_disable.clicked.connect(lambda: self.window().send_serial_text("<DISABLE>"))
        self.btn_home = QPushButton("Home All")
        self.btn_home.clicked.connect(lambda: self.window().send_serial_text("<HOME>"))
        
        self.btn_reboot = QPushButton("HW Reboot (DTR)")
        self.btn_reboot.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
        self.btn_reboot.clicked.connect(self._trigger_hw_reboot)

        for btn in (self.btn_reset_faults, self.btn_enable, self.btn_disable, self.btn_home):
            macro_layout.addWidget(btn)
        macro_layout.addStretch()
        macro_layout.addWidget(self.btn_reboot)
        root.addWidget(macro_group)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Monospace", 10))
        self.output.setStyleSheet("background:#11111b; border:1px solid #45475a; border-radius:6px;")
        root.addWidget(self.output, stretch=1)

        send_row = QHBoxLayout()
        self.send_edit = QLineEdit()
        self.send_edit.setPlaceholderText("Send command…")
        self.send_edit.returnPressed.connect(self._send)
        send_row.addWidget(self.send_edit, stretch=1)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send)
        send_row.addWidget(send_btn)
        root.addLayout(send_row)

    def set_connected(self, connected: bool) -> None:
        self.connected_changed.emit(connected)
        self.connect_btn.setChecked(connected)
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.append_line("[SERIAL] Connected" if connected else "[SERIAL] Disconnected", "#a6e3a1" if connected else "#f38ba8")

    def append_line(self, line: str, color: str = "#cdd6f4") -> None:
        flt = self.filter_edit.text().strip()
        if flt and flt.lower() not in line.lower():
            return

        prefix = ""
        if self.ts_check.isChecked():
            elapsed = (QDateTime.currentMSecsSinceEpoch() - self._t0) / 1000.0
            prefix = f"[+{elapsed:7.3f}s] "

        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(prefix + line + "\n")
        if self.autoscroll.isChecked():
            self.output.setTextCursor(cursor)
            self.output.ensureCursorVisible()

    def clear(self) -> None:
        self.output.clear()
        self._t0 = QDateTime.currentMSecsSinceEpoch()

    def _toggle_connection(self, checked: bool) -> None:
        self.connect_btn.setText("Disconnect" if checked else "Connect")
        self.connect_requested.emit()

    def _send(self) -> None:
        text = self.send_edit.text().strip()
        if not text:
            return
        self.window().send_serial_text(text)  # type: ignore[attr-defined]
        self.append_line(f">> {text}", "#89b4fa")
        self.send_edit.clear()

    def _trigger_hw_reboot(self) -> None:
        mw = self.window()
        if getattr(mw, "_serial_worker", None):
            mw._serial_worker.pulse_dtr()
            self.append_line("--- Triggered Hardware Reset (DTR Pulse) ---", "#f9e2af")
        else:
            self.append_line("Error: Not connected to serial port.", "#f38ba8")
