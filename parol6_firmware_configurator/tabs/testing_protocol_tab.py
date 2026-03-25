"""
testing_protocol_tab.py — Shows the 10-phase testing guide with one-click preset loading.
"""
from __future__ import annotations
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

# Phase metadata: (number, title, preset_file, risk_colour, summary)
PHASES = [
    (0,  "Hardware Sanity",         "phase0_hardware_check.json",    "#6c7086",
     "No power to motors. Continuity checks only. Zero risk."),
    (1,  "Encoder — Interrupt Mode","phase1_encoder_interrupt.json", "#6c7086",
     "Control loop OFF. Turn motor by hand. Verify encoder reads change."),
    (2,  "Encoder — QuadTimer Mode","phase2_quadtimer.json",         "#6c7086",
     "Switch to QuadTimer. Compare ISR profiler with Phase 1."),
    (3,  "STEP/DIR Signal Check",   "phase3_step_dir_test.json",     "#f9e2af",
     "Fixed 1 kHz STEP output. Verify on oscilloscope. No control loop."),
    (4,  "Open-Loop Motor Rotation","phase4_open_loop.json",         "#f9e2af",
     "First time motor moves. PSU current limit to 1.5 A."),
    (5,  "Closed-Loop, No Filter",  "phase5_closed_loop_j1.json",    "#f38ba8",
     "J1 only. Feedback ON, filter OFF. Motor may oscillate — expected."),
    (6,  "Add AlphaBeta Filter",    "phase6_with_filter.json",       "#a6e3a1",
     "Toggle filter ON. Compare overshoot vs Phase 5."),
    (7,  "Add Vel. Feedforward",    "phase7_with_feedforward.json",  "#a6e3a1",
     "Add feedforward. Track a slow sine trajectory."),
    (8,  "Add Watchdog",            "phase8_with_watchdog.json",     "#a6e3a1",
     "Pull USB cable — motor must stop within 200 ms."),
    (9,  "Multi-Axis",              "phase9_multiaxis.json",         "#a6e3a1",
     "Enable joints one at a time. J4→J5→J6→J1→J2→J3."),
    (10, "Full Feature Stack",      "phase10_full_stack.json",       "#cba6f7",
     "All features ON. Raise ROS rate to 100 Hz. Measure tracking error."),
]


class PhaseCard(QFrame):
    load_requested = pyqtSignal(str)  # emits preset filename

    def __init__(self, num: int, title: str, preset: str,
                 colour: str, summary: str, parent=None):
        super().__init__(parent)
        self._preset = preset
        self.setObjectName("PhaseCard")
        self.setStyleSheet(f"""
            QFrame#PhaseCard {{
                background: #313244;
                border: 1px solid #45475a;
                border-left: 4px solid {colour};
                border-radius: 8px;
                margin: 2px 0;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Phase badge
        badge = QLabel(f"P{num}")
        badge.setFixedWidth(36)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"color: {colour}; font-weight: bold; font-size: 14px;")
        layout.addWidget(badge)

        # Text
        text_col = QVBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        summary_lbl = QLabel(summary)
        summary_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        summary_lbl.setWordWrap(True)
        text_col.addWidget(title_lbl)
        text_col.addWidget(summary_lbl)
        layout.addLayout(text_col, stretch=1)

        # Load button
        btn = QPushButton("Load Preset")
        btn.setFixedWidth(100)
        btn.setToolTip(f"Load {preset}")
        btn.clicked.connect(lambda: self.load_requested.emit(preset))
        layout.addWidget(btn)


class TestingProtocolTab(QWidget):
    """Tab showing the 10-phase bring-up guide with one-click preset loading."""

    load_preset = pyqtSignal(str)  # emits absolute path of chosen preset file

    def __init__(self, configs_dir: str, parent=None):
        super().__init__(parent)
        self._configs_dir = configs_dir
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        header = QLabel("🔬  Bring-Up Testing Protocol")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #cba6f7;")
        root.addWidget(header)

        subtitle = QLabel(
            "Follow phases top-to-bottom. Each card loads the matching config preset. "
            "Click 'Load Preset' then go to the Flash tab."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #a6adc8; font-size: 12px;")
        root.addWidget(subtitle)

        # Scrollable phase list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(6)
        vbox.setContentsMargins(0, 0, 0, 0)

        for num, title, preset, colour, summary in PHASES:
            card = PhaseCard(num, title, preset, colour, summary)
            card.load_requested.connect(self._on_load_preset)
            vbox.addWidget(card)

        vbox.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)

        # Links row
        link_row = QHBoxLayout()
        doc_btn = QPushButton("📄 Open Full Testing Protocol")
        doc_btn.clicked.connect(self._open_protocol_doc)
        link_row.addWidget(doc_btn)
        link_row.addStretch()
        root.addLayout(link_row)

    def _on_load_preset(self, filename: str) -> None:
        path = os.path.join(self._configs_dir, filename)
        if os.path.exists(path):
            self.load_preset.emit(path)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Preset Missing",
                f"Preset file not found:\n{path}\n\nThe default config will be used."
            )

    def _open_protocol_doc(self) -> None:
        doc = os.path.join(os.path.dirname(self._configs_dir), "docs", "TESTING_PROTOCOL.md")
        if os.path.exists(doc):
            import subprocess, sys
            subprocess.Popen(["xdg-open" if sys.platform == "linux" else "open", doc])
