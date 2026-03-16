import math
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QCheckBox, QLabel, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor, QKeyEvent

# Known firmware state descriptions
_STATE_LABELS = {
    0: ("IDLE",     "#a6adc8"),
    1: ("RUNNING",  "#a6e3a1"),
    2: ("HOMING",   "#89dceb"),
    3: ("FAULT",    "#f38ba8"),
    4: ("ESTOP",    "#f38ba8"),
    5: ("DISABLED", "#f9e2af"),
}


class _HistoryLineEdit(QLineEdit):
    """QLineEdit with Up/Down arrow command history."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._history_idx = -1

    def push_history(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_idx = -1

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Up:
            if self._history:
                self._history_idx = max(0, self._history_idx - 1 if self._history_idx >= 0
                                        else len(self._history) - 1)
                self.setText(self._history[self._history_idx])
        elif event.key() == Qt.Key.Key_Down:
            if self._history_idx >= 0:
                self._history_idx += 1
                if self._history_idx >= len(self._history):
                    self._history_idx = -1
                    self.clear()
                else:
                    self.setText(self._history[self._history_idx])
        else:
            super().keyPressEvent(event)


class SerialTab(QWidget):
    connect_requested = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        self._autoscroll = True
        self._show_timestamps = True
        self._build_ui()
        
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Firmware State Badge ─────────────────────────────────────────
        badge_row = QHBoxLayout()
        badge_row.addWidget(QLabel("Firmware state:"))
        self.state_badge = QLabel("—")
        self.state_badge.setStyleSheet(
            "background:#313244; border:1px solid #45475a; border-radius:4px; "
            "color:#a6adc8; font-weight:bold; padding:2px 10px;"
        )
        badge_row.addWidget(self.state_badge)
        badge_row.addStretch()

        # ── Macros & Options Row ──
        opt_layout = QHBoxLayout()
        
        self.ts_cb = QCheckBox("Timestamps")
        self.ts_cb.setChecked(True)
        self.ts_cb.stateChanged.connect(self._on_ts_toggled)
        opt_layout.addWidget(self.ts_cb)

        self.scroll_cb = QCheckBox("Autoscroll")
        self.scroll_cb.setChecked(True)
        self.scroll_cb.stateChanged.connect(self._on_scroll_toggled)
        opt_layout.addWidget(self.scroll_cb)

        opt_layout.addWidget(QLabel(" Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("e.g. FAULT")
        self.filter_edit.setMaximumWidth(150)
        opt_layout.addWidget(self.filter_edit)
        
        opt_layout.addStretch()
        
        # Macros based on registry
        proj = self._main_window.current_project()
        macros = proj.get("serial", {}).get("macros", [])
        caps = proj.get("capabilities", {})
        
        if macros:
            macro_box = QGroupBox("Macros")
            macro_lay = QHBoxLayout(macro_box)
            macro_lay.setContentsMargins(4, 12, 4, 4)
            for m in macros:
                btn = QPushButton(m)
                # Colour REBOOT_DFU differently so it's obviously destructive
                if "REBOOT_DFU" in m:
                    btn.setStyleSheet("background:#cba6f7; color:#1e1e2e; font-weight:bold;")
                    btn.setToolTip("Send <REBOOT_DFU> — STM32 will reset into DFU bootloader")
                elif "RESET" in m:
                    btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")
                btn.clicked.connect(lambda checked, text=m: self._main_window.send_serial_text(text))
                macro_lay.addWidget(btn)
            opt_layout.addWidget(macro_box)
            
        # DTR Reboot (Capability driven)
        if caps.get("supports_dtr_reboot", False):
            self.hw_reboot_btn = QPushButton("💥 HW Reboot (DTR)")
            self.hw_reboot_btn.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
            self.hw_reboot_btn.setToolTip("Pulls the DTR line low briefly to reset the microcontroller")
            self.hw_reboot_btn.clicked.connect(self._trigger_hw_reboot)
            opt_layout.addWidget(self.hw_reboot_btn)

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._clear)
        opt_layout.addWidget(clear_btn)

        layout.addLayout(badge_row)
        layout.addLayout(opt_layout)

        # ── Output Area ──
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.output.setStyleSheet("font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 12px; background: #11111b;")
        layout.addWidget(self.output, stretch=1)

        # ── Input Row ──
        in_layout = QHBoxLayout()
        self.input_field = _HistoryLineEdit()
        self.input_field.setPlaceholderText("Type command and press Enter (↑↓ = history)...")
        self.input_field.returnPressed.connect(self._send)
        in_layout.addWidget(self.input_field)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send)
        in_layout.addWidget(send_btn)

        layout.addLayout(in_layout)

    def _on_ts_toggled(self, state: int) -> None:
        self._show_timestamps = bool(state)

    def _on_scroll_toggled(self, state: int) -> None:
        self._autoscroll = bool(state)

    def _clear(self) -> None:
        self.output.clear()

    def _trigger_hw_reboot(self) -> None:
        if self._main_window._serial_worker:
            self.output.append("<span style='color:#f9e2af;'>[GUI] Pulsing DTR line for Hardware Reboot...</span>")
            self._main_window._serial_worker.pulse_dtr()
        else:
            self.output.append("<span style='color:#f38ba8;'>[GUI] Cannot Reboot: Serial disconnected.</span>")

    def _send(self) -> None:
        text = self.input_field.text()
        if not text:
            return
        self.input_field.push_history(text)
        self.input_field.clear()
        self._append(f"> {text}", color="#a6e3a1")
        self._main_window.send_serial_text(text)

    def _on_line(self, line: str) -> None:
        filt = self.filter_edit.text()
        if filt and filt.lower() not in line.lower():
            return

        # Suppress ACK spam from flooding the terminal if filter is empty
        if not filt and line.startswith("<ACK,"):
            return

        text = ""
        if self._show_timestamps:
            ms = int((time.time() % 1.0) * 1000)
            t_str = time.strftime(f"%H:%M:%S.{ms:03d}")
            text += f"<span style='color:#6c7086;'>[{t_str}]</span> "

        # Colour known firmware messages
        if "FAULT" in line or "ESTOP" in line:
            text += f"<span style='color:#f38ba8;'>{line}</span>"
        elif "INIT_OK" in line or "HOMING_DONE" in line:
            text += f"<span style='color:#a6e3a1;'>{line}</span>"
        elif "HOMING" in line or "REBOOTING" in line:
            text += f"<span style='color:#89dceb;'>{line}</span>"
        elif "STALE" in line or "WARN" in line:
            text += f"<span style='color:#f9e2af;'>{line}</span>"
        else:
            text += line

        self._append(text)

    def update_state_badge(self, state_byte: int | None) -> None:
        """Called by main window when a telemetry packet with state_byte arrives."""
        if state_byte is None:
            return
        label, color = _STATE_LABELS.get(state_byte, (f"STATE_{state_byte}", "#a6adc8"))
        self.state_badge.setText(f" {label} ")
        self.state_badge.setStyleSheet(
            f"background:{color}22; border:1px solid {color}; border-radius:4px; "
            f"color:{color}; font-weight:bold; padding:2px 10px;"
        )

    def _append(self, html: str, color: str | None = None) -> None:
        if color:
            html = f"<span style='color:{color};'>{html}</span>"
            
        if self.output.document().blockCount() > 5000:
            cursor = self.output.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, 500)
            cursor.removeSelectedText()
            
        self.output.append(html)
        if self._autoscroll:
            self.output.moveCursor(QTextCursor.MoveOperation.End)

    def disconnect(self) -> None:
        self._append("<em>Disconnected.</em>", color="#f38ba8")
        self.state_badge.setText("—")
        self.state_badge.setStyleSheet(
            "background:#313244; border:1px solid #45475a; border-radius:4px; "
            "color:#a6adc8; font-weight:bold; padding:2px 10px;"
        )
