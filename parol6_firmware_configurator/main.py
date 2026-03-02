"""
main.py — PAROL6 Firmware Configurator
Entry point: QMainWindow wiring all tabs together with dark theme.
"""
from __future__ import annotations
import os
import sys
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QStatusBar, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QInputDialog, QToolBar, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont

# ── Core ──────────────────────────────────────────────────────────────────
from core.config_model import RobotConfig
from core.code_generator import generate_config_h
from core.serial_monitor import SerialWorker, list_serial_ports

# ── Tabs ──────────────────────────────────────────────────────────────────
from tabs.testing_protocol_tab import TestingProtocolTab
from tabs.features_tab import FeaturesTab
from tabs.joints_tab import JointsTab
from tabs.comms_tab import CommsTab
from tabs.jog_tab import JogTab
from tabs.serial_tab import SerialTab
from tabs.plot_tab import PlotTab
from tabs.flash_tab import FlashTab
from tabs.fault_log_tab import FaultLogTab
from tabs.launch_tab import LaunchTab

# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR  = os.path.join(BASE_DIR, "saved_configs")
GEN_DIR      = os.path.join(BASE_DIR, "generated")
GEN_CONFIG_H = os.path.join(GEN_DIR, "config.h")
FW_DIR       = os.path.join(BASE_DIR, "..", "parol6_firmware")

DARK_STYLESHEET = """
QMainWindow, QWidget, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 0 8px 8px 8px;
    background: #1e1e2e;
}
QTabBar::tab {
    background: #181825;
    color: #a6adc8;
    padding: 7px 16px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #313244; color: #cba6f7; font-weight: bold; }
QTabBar::tab:hover    { background: #313244; color: #cdd6f4; }
QPushButton {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #cdd6f4;
}
QPushButton:hover   { background: #45475a; border-color: #cba6f7; }
QPushButton:pressed { background: #585b70; }
QPushButton:disabled{ color: #585b70; border-color: #313244; }
QPushButton:checked { background: #a6e3a1; color: #1e1e2e; font-weight: bold; }
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: #a6adc8;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #11111b;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
    padding: 3px 6px;
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #cba6f7;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #313244; border: 1px solid #45475a; }
QScrollBar:vertical   { background:#181825; width:8px;  border-radius:4px; }
QScrollBar::handle:vertical { background:#45475a; border-radius:4px; min-height:24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:#181825; height:8px; border-radius:4px; }
QScrollBar::handle:horizontal { background:#45475a; border-radius:4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }
QTableWidget { gridline-color: #45475a; alternate-background-color: #252535; }
QHeaderView::section { background:#313244; color:#a6adc8; padding:4px; border:none; }
QStatusBar { background: #181825; color: #a6adc8; border-top: 1px solid #45475a; }
QProgressBar { background:#181825; border:1px solid #45475a; border-radius:4px; text-align:center; }
QProgressBar::chunk { background:#a6e3a1; border-radius:4px; }
QCheckBox { color: #cdd6f4; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #45475a; border-radius:3px; background:#11111b; }
QCheckBox::indicator:checked { background:#cba6f7; border-color:#cba6f7; }
QLabel { color: #cdd6f4; }
QToolBar { background: #181825; border-bottom: 1px solid #45475a; spacing: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6 Firmware Configurator")
        self.resize(1280, 820)
        self._cfg  = RobotConfig()
        self._serial_worker: SerialWorker | None = None
        self._settings = QSettings("PAROL6", "FirmwareConfigurator")

        self._build_toolbar()
        self._build_tabs()
        self._build_status_bar()
        self._wire_signals()
        self._load_session()

        # Plot refresh timer
        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._plot_tab.refresh)
        self._plot_timer.start(50)  # 20 Hz refresh

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        from PyQt6.QtWidgets import QSizePolicy

        # ── Row 1: file operations + profile name ──────────────────────
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        new_btn    = QPushButton("New")
        open_btn   = QPushButton("Open…")
        save_btn   = QPushButton("Save")
        saveas_btn = QPushButton("Save As…")
        new_btn.clicked.connect(self._new_config)
        open_btn.clicked.connect(self._open_config)
        save_btn.clicked.connect(self._save_config)
        saveas_btn.clicked.connect(self._save_config_as)
        for b in (new_btn, open_btn, save_btn, saveas_btn):
            tb.addWidget(b)

        tb.addSeparator()
        self._profile_name = QLabel("  Profile: default")
        self._profile_name.setStyleSheet("color:#cba6f7; font-weight:bold;")
        tb.addWidget(self._profile_name)

        sp1 = QWidget()
        sp1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(sp1)

        rename_btn = QPushButton("Rename…")
        rename_btn.clicked.connect(self._rename_config)
        tb.addWidget(rename_btn)

        # ── Row 2: serial connection ────────────────────────────────
        tb2 = self.addToolBar("Serial")
        tb2.setMovable(False)

        tb2.addWidget(QLabel(" 🔌 Port: "))

        # Plain text field — always works in X11/Docker, user types directly
        self._sb_port = QLineEdit()
        self._sb_port.setMinimumWidth(200)
        self._sb_port.setPlaceholderText("/dev/ttyACM0")
        self._sb_port.setToolTip("Type the serial port path (e.g. /dev/ttyACM0)")
        tb2.addWidget(self._sb_port)

        # Scan button: detects available ports and lets user pick via simple dialog
        scan_btn = QPushButton("🔍 Scan")
        scan_btn.setFixedWidth(64)
        scan_btn.setToolTip("Scan for available serial ports and select one")
        scan_btn.clicked.connect(self._scan_ports)
        tb2.addWidget(scan_btn)

        tb2.addSeparator()
        tb2.addWidget(QLabel(" Baud: "))

        self._sb_baud = QLineEdit("115200")
        self._sb_baud.setFixedWidth(80)
        self._sb_baud.setToolTip("Baud rate (115200, 921600, 9600…)")
        tb2.addWidget(self._sb_baud)

        tb2.addSeparator()

        self._sb_connect_btn = QPushButton("⚡ Connect")
        self._sb_connect_btn.setCheckable(True)
        self._sb_connect_btn.setStyleSheet(
            "QPushButton { padding: 4px 16px; }"
            "QPushButton:checked { background:#a6e3a1; color:#1e1e2e; font-weight:bold; }"
        )
        self._sb_connect_btn.clicked.connect(self._toggle_serial)
        tb2.addWidget(self._sb_connect_btn)

        sp2 = QWidget()
        sp2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb2.addWidget(sp2)

        # Pre-fill with the first detected port if available
        ports = list_serial_ports()
        if ports:
            self._sb_port.setText(ports[0])

    def _build_tabs(self) -> None:
        self._proto_tab = TestingProtocolTab(CONFIGS_DIR)
        self._feat_tab  = FeaturesTab()
        self._joints_tab= JointsTab()
        self._comms_tab = CommsTab()
        self._jog_tab   = JogTab()
        self._serial_tab= SerialTab()
        self._plot_tab  = PlotTab()
        self._flash_tab = FlashTab()
        self._fault_tab = FaultLogTab()
        self._launch_tab= LaunchTab()

        self._tabs = QTabWidget()
        self._tabs.addTab(self._proto_tab,  "🔬 Protocol")
        self._tabs.addTab(self._feat_tab,   "⚙️ Features")
        self._tabs.addTab(self._joints_tab, "🔩 Joints")
        self._tabs.addTab(self._comms_tab,  "📡 Comms")
        self._tabs.addTab(self._jog_tab,    "🕹 Jog")
        self._tabs.addTab(self._serial_tab, "💬 Serial")
        self._tabs.addTab(self._plot_tab,   "📈 Oscilloscope")
        self._tabs.addTab(self._flash_tab,  "⚡ Flash")
        self._tabs.addTab(self._launch_tab, "🚀 ROS2 Launch")
        self._tabs.addTab(self._fault_tab,  "⚠️ Faults")

        self.setCentralWidget(self._tabs)

        self._flash_tab.set_firmware_path(os.path.abspath(FW_DIR))

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._sb_conn  = QLabel("🔴 Disconnected")
        self._sb_rate  = QLabel("0 pkt/s")
        self._sb_state = QLabel("State: —")
        self._sb_cfg   = QLabel("")

        for w in (self._sb_conn, QLabel("|"), self._sb_rate,
                  QLabel("|"), self._sb_state, QLabel("|"), self._sb_cfg):
            sb.addPermanentWidget(w)

    def _scan_ports(self) -> None:
        """Detect available serial ports and let user choose via a simple dialog."""
        ports = list_serial_ports()
        if not ports:
            QMessageBox.information(
                self, "No Ports Found",
                "No serial ports detected.\n\n"
                "Check:\n"
                "  • Teensy is plugged in\n"
                "  • Container has device access (-v /dev:/dev)\n"
                "  • Run: ls /dev/ttyACM* /dev/ttyUSB*\n\n"
                "You can still type the port manually in the Port field."
            )
            return
        port, ok = QInputDialog.getItem(
            self, "Select Port", "Available serial ports:", ports, 0, False
        )
        if ok and port:
            self._sb_port.setText(port)

    def _refresh_sb_ports(self) -> None:
        """Legacy no-op kept so any stale references don't crash."""
        pass

    # ------------------------------------------------------------------
    # Signal Wiring
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        # Feature/joint/comms changes → mark unsaved
        self._feat_tab.changed.connect(self._on_config_changed)
        self._joints_tab.changed.connect(self._on_config_changed)
        self._comms_tab.changed.connect(self._on_config_changed)

        # Protocol tab: load preset file
        self._proto_tab.load_preset.connect(self._load_preset_file)

        # Flash tab
        self._flash_tab.generate_requested.connect(self._generate_config)
        self._serial_tab.connect_requested.connect(self._toggle_serial)

        # Jog tab send
        self._jog_tab.send_command.connect(self._serial_send)

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------
    def _read_gui_into_cfg(self) -> None:
        self._feat_tab.save(self._cfg.features)
        self._joints_tab.save(self._cfg.joints)
        self._comms_tab.save(self._cfg.comms)

    def _push_cfg_to_gui(self) -> None:
        self._feat_tab.load(self._cfg.features)
        self._joints_tab.load(self._cfg.joints)
        self._comms_tab.load(self._cfg.comms)
        self._profile_name.setText(f"  Profile: {self._cfg.name}")

    def _on_config_changed(self) -> None:
        self._profile_name.setText(f"  Profile: {self._cfg.name} *")

    def _new_config(self) -> None:
        self._cfg = RobotConfig()
        self._push_cfg_to_gui()

    def _open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open config", CONFIGS_DIR, "JSON (*.json)")
        if path:
            self._load_from_path(path)

    def _load_from_path(self, path: str) -> None:
        try:
            self._cfg = RobotConfig.load(path)
            self._push_cfg_to_gui()
            self._sb_cfg.setText(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _load_preset_file(self, path: str) -> None:
        self._load_from_path(path)
        self._tabs.setCurrentWidget(self._feat_tab)

    def _save_config(self) -> None:
        self._read_gui_into_cfg()
        path = os.path.join(CONFIGS_DIR, f"{self._cfg.name}.json")
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        self._cfg.save(path)
        self._profile_name.setText(f"  Profile: {self._cfg.name}")
        self._sb_cfg.setText(f"Saved: {os.path.basename(path)}")

    def _save_config_as(self) -> None:
        self._read_gui_into_cfg()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", CONFIGS_DIR, "JSON (*.json)")
        if path:
            self._cfg.name = os.path.splitext(os.path.basename(path))[0]
            self._cfg.save(path)
            self._profile_name.setText(f"  Profile: {self._cfg.name}")

    def _rename_config(self) -> None:
        name, ok = QInputDialog.getText(self, "Rename Profile", "New name:", text=self._cfg.name)
        if ok and name:
            self._cfg.name = name
            self._profile_name.setText(f"  Profile: {name}")

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    def _generate_config(self) -> None:
        self._read_gui_into_cfg()
        try:
            content = generate_config_h(self._cfg, GEN_CONFIG_H)
            self._flash_tab.set_preview(content)
            self._sb_cfg.setText(f"config.h generated ✓ ({len(content.splitlines())} lines)")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", str(e))

    # ------------------------------------------------------------------
    # Serial connection
    # ------------------------------------------------------------------
    def _toggle_serial(self) -> None:
        if self._serial_worker:
            self._serial_worker.stop()
            self._serial_worker = None
            self._sb_connect_btn.setText("⚡ Connect")
            self._sb_connect_btn.setChecked(False)
            self._serial_tab.disconnect()
            return

        # Port: toolbar text field first, fall back to comms tab
        port = self._sb_port.text().strip()
        if not port:
            self._read_gui_into_cfg()
            port = self._cfg.comms.serial_port or self._comms_tab.serial_port
        try:
            baud = int(self._sb_baud.text().strip())
        except ValueError:
            baud = 115200

        if not port or "(no ports" in port:
            QMessageBox.warning(
                self, "No Port",
                "No serial port selected.\n\nClick 🔄 to refresh the port list,\n"
                "or select a port from the dropdown in the status bar."
            )
            self._sb_connect_btn.setChecked(False)
            return

        self._serial_worker = SerialWorker(port, baud)
        self._serial_worker.raw_line.connect(self._on_raw_line)
        self._serial_worker.telemetry.connect(self._on_telemetry)
        self._serial_worker.connected.connect(self._on_serial_connected)
        self._serial_worker.packet_rate.connect(self._on_packet_rate)
        self._serial_worker.error_msg.connect(lambda m: self._serial_tab._append(m, "#f38ba8"))
        self._serial_worker.raw_line.connect(self._serial_tab._on_line)
        self._serial_worker.start()
        self._sb_connect_btn.setText("⏹ Disconnect")

    def _serial_send(self, text: str) -> None:
        if self._serial_worker:
            self._serial_worker.send(text)

    def _on_serial_connected(self, ok: bool) -> None:
        self._sb_conn.setText("🟢 Connected" if ok else "🔴 Disconnected")
        self._sb_conn.setStyleSheet("color:#a6e3a1;" if ok else "color:#f38ba8;")

    def _on_packet_rate(self, rate: float) -> None:
        self._sb_rate.setText(f"{rate:.1f} pkt/s")

    def _on_raw_line(self, line: str) -> None:
        # Route fault lines to fault log
        if "FAULT" in line or "SOFT_ESTOP" in line:
            self._fault_tab.record_serial_fault(line)

    def _on_telemetry(self, pkt: dict) -> None:
        self._plot_tab.ingest(pkt)
        self._jog_tab.ingest_telemetry(pkt)
        self._fault_tab.ingest_telemetry(pkt)
        state = pkt.get("supervisor_state", "")
        if state:
            self._sb_state.setText(f"State: {state}")

    # ------------------------------------------------------------------
    # Session save/restore
    # ------------------------------------------------------------------
    def _load_session(self) -> None:
        last = self._settings.value("last_config_path", "")
        if last and os.path.exists(last):
            self._load_from_path(last)
        else:
            self._push_cfg_to_gui()

    def closeEvent(self, event) -> None:
        self._read_gui_into_cfg()
        if self._serial_worker:
            self._serial_worker.stop()
        # Auto-save session state
        path = os.path.join(CONFIGS_DIR, "_session_autosave.json")
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        self._cfg.save(path)
        self._settings.setValue("last_config_path", path)
        event.accept()


# ---------------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    app.setFont(QFont("Inter", 11))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
