"""
flash_tab.py — Generate config.h, preview it, build, flash via PlatformIO.
"""
from __future__ import annotations
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QGroupBox, QLineEdit, QFileDialog, QComboBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ..core.flash_manager import FlashWorker, BuildWorker


class FlashTab(QWidget):
    """Generate config.h, preview, build-check, and flash to Teensy."""

    # Emitted so main window can call code_generator before flashing
    generate_requested = pyqtSignal()
    build_requested    = pyqtSignal()
    flash_requested    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flash_worker: FlashWorker | None = None
        self._build_worker: BuildWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("⚡  Flash Manager")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)

        # ── PlatformIO path ───────────────────────────────────────────
        path_box = QGroupBox("PlatformIO Project")
        path_lay = QHBoxLayout(path_box)
        path_lay.addWidget(QLabel("Firmware dir:"))
        self.fw_path = QLineEdit()
        self.fw_path.setPlaceholderText("/path/to/parol6_firmware/")
        path_lay.addWidget(self.fw_path, stretch=1)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse_fw_path)
        path_lay.addWidget(browse_btn)

        path_lay.addWidget(QLabel("Env:"))
        self.env_combo = QComboBox()
        self.env_combo.addItems(["teensy41", "native"])
        path_lay.addWidget(self.env_combo)

        path_lay.addWidget(QLabel("Mode:"))
        self.build_mode = QComboBox()
        self.build_mode.addItems(["debug", "release"])
        self.build_mode.setToolTip(
            "debug: ISR profiler ON, verbose serial output.\n"
            "release: lean build, no profiler overhead.")
        path_lay.addWidget(self.build_mode)
        root.addWidget(path_box)

        # ── Action buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        gen_btn = QPushButton("⚙️  Generate config.h")
        gen_btn.setToolTip("Writes generated/config.h from current GUI settings.")
        gen_btn.clicked.connect(self.generate_requested)

        build_btn = QPushButton("🔨  Build Only")
        build_btn.setToolTip("Compile without flashing. Useful for catching errors before connecting Teensy.")
        build_btn.clicked.connect(self._run_build)

        self.flash_btn = QPushButton("⚡  Generate & Flash")
        self.flash_btn.setStyleSheet("background:#cba6f7; color:#1e1e2e; font-weight:bold; padding:6px 18px;")
        self.flash_btn.setToolTip("Generates config.h then runs pio run --upload.")
        self.flash_btn.clicked.connect(self._run_flash)

        self.abort_btn = QPushButton("✖ Abort")
        self.abort_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e;")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._abort)

        for btn in (gen_btn, build_btn, self.flash_btn, self.abort_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # ── config.h preview ──────────────────────────────────────────
        prev_box = QGroupBox("Generated config.h Preview")
        prev_lay = QVBoxLayout(prev_box)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Monospace", 9))
        self.preview.setStyleSheet("background:#11111b; color:#a6e3a1; border:none;")
        self.preview.setMaximumHeight(220)
        prev_lay.addWidget(self.preview)
        root.addWidget(prev_box)

        # ── Build log ─────────────────────────────────────────────────
        log_box = QGroupBox("Build / Flash Log")
        log_lay = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Monospace", 9))
        self.log.setStyleSheet("background:#11111b; color:#cdd6f4; border:none;")
        log_lay.addWidget(self.log)
        root.addWidget(log_box, stretch=1)

    # ------------------------------------------------------------------
    def set_preview(self, content: str) -> None:
        self.preview.setPlainText(content)

    def set_firmware_path(self, path: str) -> None:
        self.fw_path.setText(path)

    def append_log(self, line: str) -> None:
        colour = "#f38ba8" if ("error" in line.lower() or "❌" in line) \
            else "#a6e3a1" if ("✅" in line) \
            else "#cdd6f4"
        self.log.append(f'<span style="color:{colour};">{line}</span>')

    def _browse_fw_path(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select firmware directory")
        if d:
            self.fw_path.setText(d)

    def _run_flash(self) -> None:
        self.generate_requested.emit()   # main window writes config.h first
        fw_dir = self.fw_path.text().strip()
        if not fw_dir:
            self.append_log("[FLASH] ⚠️  No firmware directory set.")
            return
        self.flash_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._flash_worker = FlashWorker(fw_dir, self.env_combo.currentText())
        self._flash_worker.output_line.connect(self.append_log)
        self._flash_worker.finished_ok.connect(self._on_done_ok)
        self._flash_worker.finished_err.connect(self._on_done_err)
        self._flash_worker.start()

    def _run_build(self) -> None:
        self.generate_requested.emit()
        fw_dir = self.fw_path.text().strip()
        if not fw_dir:
            self.append_log("[BUILD] ⚠️  No firmware directory set.")
            return
        self._build_worker = BuildWorker(fw_dir, self.env_combo.currentText())
        self._build_worker.output_line.connect(self.append_log)
        self._build_worker.finished_ok.connect(lambda: None)
        self._build_worker.finished_err.connect(lambda _: None)
        self._build_worker.start()

    def _abort(self) -> None:
        if self._flash_worker:
            self._flash_worker.abort()
        if self._build_worker:
            self._build_worker.abort()
        self._on_done_err(-1)

    def _on_done_ok(self) -> None:
        self.flash_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)

    def _on_done_err(self, _rc: int) -> None:
        self.flash_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
