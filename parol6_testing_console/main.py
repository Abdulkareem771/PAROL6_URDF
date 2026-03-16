import os
import sys
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QStatusBar, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QMessageBox, QComboBox, QToolBar, QLineEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont

from core.serial_monitor import SerialWorker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(BASE_DIR, "project_registry.json")

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
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: #a6adc8;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {
    background: #11111b;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
    padding: 3px 6px;
}
QComboBox QAbstractItemView { background: #313244; border: 1px solid #45475a; }
QStatusBar { background: #181825; color: #a6adc8; border-top: 1px solid #45475a; }
QToolBar { background: #181825; border-bottom: 1px solid #45475a; spacing: 6px; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6 Testing Console")
        self.resize(1280, 820)
        self._settings = QSettings("PAROL6", "TestingConsole")
        self._registry = []
        self._current_project = None
        
        self._serial_worker = None
        
        self._load_registry()
        self._build_toolbar()
        self._build_tabs()
        self._build_status_bar()
        
        last_pid = self._settings.value("last_project_id", "realtime_servo_blackpill")
        if last_pid == "parol6_stm32_bringup":
            # Force transition for the user's current session
            last_pid = "realtime_servo_blackpill"
        self._set_project(last_pid)

    def _load_registry(self):
        if not os.path.exists(REGISTRY_PATH):
            QMessageBox.critical(self, "Registry Error", f"Missing {REGISTRY_PATH}")
            sys.exit(1)
        try:
            with open(REGISTRY_PATH, 'r') as f:
                data = json.load(f)
                self._registry = data.get("projects", [])
            if not self._registry:
                raise ValueError("No projects found in registry")
        except Exception as e:
            QMessageBox.critical(self, "Registry Error", f"Failed to parse registry: {e}")
            sys.exit(1)

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        
        tb.addWidget(QLabel(" Project: "))
        self.project_combo = QComboBox()
        for proj in self._registry:
            self.project_combo.addItem(proj.get("name", "Unknown"), proj.get("id"))
        
        self.project_combo.currentIndexChanged.connect(self._on_project_combo_changed)
        tb.addWidget(self.project_combo)
        
        # Spacer
        sp1 = QWidget()
        sp1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(sp1)
        
        tb.addWidget(QLabel(" 🔌 Port: "))
        self._sb_port = QLineEdit()
        self._sb_port.setMinimumWidth(150)
        tb.addWidget(self._sb_port)
        
        scan_btn = QPushButton("🔍 Scan")
        scan_btn.clicked.connect(self._scan_ports)
        tb.addWidget(scan_btn)
        
        tb.addWidget(QLabel(" Baud: "))
        self._sb_baud = QLineEdit("115200")
        self._sb_baud.setFixedWidth(80)
        tb.addWidget(self._sb_baud)
        
        self._sb_connect_btn = QPushButton("⚡ Connect")
        self._sb_connect_btn.setCheckable(True)
        self._sb_connect_btn.clicked.connect(self._toggle_serial)
        tb.addWidget(self._sb_connect_btn)

    def _build_tabs(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)
        # Tabs will be instantiated and added when a project is selected
        # since their visibility depends on the active project's capabilities.
        self._tabs_dict = {}
        self._tabs.currentChanged.connect(self._animate_tab_change)
        self._anim = None

    def _animate_tab_change(self, index: int):
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
        from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
        widget = self._tabs.widget(index)
        if not widget:
            return
            
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        self._anim = QPropertyAnimation(effect, b"opacity")
        self._anim.setDuration(250)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._sb_conn  = QLabel("🔴 Disconnected")
        self._sb_rate  = QLabel("0 pkt/s")
        self._sb_data_rate = QLabel("0 B/s")
        self._sb_state = QLabel("State: —")

        for w in (self._sb_conn, QLabel("|"), 
                  self._sb_rate, QLabel("("), self._sb_data_rate, QLabel(") |"), 
                  self._sb_state):
            sb.addPermanentWidget(w)

    def _on_project_combo_changed(self, idx):
        proj_id = self.project_combo.itemData(idx)
        self._set_project(proj_id)

    def _set_project(self, project_id: str):
        proj = next((p for p in self._registry if p["id"] == project_id), self._registry[0])
        self._current_project = proj
        
        idx = self.project_combo.findData(proj["id"])
        if idx >= 0:
            self.project_combo.blockSignals(True)
            self.project_combo.setCurrentIndex(idx)
            self.project_combo.blockSignals(False)
            
        self._settings.setValue("last_project_id", proj["id"])
        
        # Set default serial params if text is empty or we force it (user requested)
        ser_cfg = proj.get("serial", {})
        if ser_cfg:
            self._sb_port.setText(ser_cfg.get("default_port", "/dev/ttyACM0"))
            self._sb_baud.setText(str(ser_cfg.get("default_baud", 115200)))
            
        self._rebuild_tabs_for_project()

    def _rebuild_tabs_for_project(self):
        self._tabs.clear()
        self._tabs_dict.clear()
        
        proj = self._current_project
        caps = proj.get("capabilities", {})
        
        # 1. Docs Tab
        if proj.get("docs_path"):
            from tabs.docs_tab import DocsTab
            self._tabs_dict["docs"] = DocsTab(self)
            self._tabs.addTab(self._tabs_dict["docs"], "📖 Docs")
            
        # 2. Serial Tab
        from tabs.serial_tab import SerialTab
        self._tabs_dict["serial"] = SerialTab(self)
        self._tabs.addTab(self._tabs_dict["serial"], "💬 Serial")
        
        # 3. Plot Tab (Oscilloscope)
        if caps.get("supports_ack_plot", True):
            from tabs.plot_tab import PlotTab
            self._tabs_dict["plot"] = PlotTab(self)
            self._tabs.addTab(self._tabs_dict["plot"], "📈 Oscilloscope")
            
        # 4. Flash Tab
        if proj.get("flash"):
            from tabs.flash_tab import FlashTab
            self._tabs_dict["flash"] = FlashTab(self)
            self._tabs.addTab(self._tabs_dict["flash"], "⚡ Flash")
            
        # 5. Launch Tab
        if proj.get("ros2", {}).get("launch_actions"):
            from tabs.launch_tab import LaunchTab
            self._tabs_dict["launch"] = LaunchTab(self)
            self._tabs.addTab(self._tabs_dict["launch"], "🚀 ROS2 Launch")
            
        # 6. Fault Log Tab
        if caps.get("supports_fault_log", False):
            from tabs.fault_log_tab import FaultLogTab
            self._tabs_dict["faults"] = FaultLogTab(self)
            self._tabs.addTab(self._tabs_dict["faults"], "⚠️ Faults")
            
        # 7. Jog Tab
        if caps.get("supports_jog", False):
            from tabs.jog_tab import JogTab
            self._tabs_dict["jog"] = JogTab(self)
            self._tabs_dict["jog"].send_command.connect(self.send_serial_text)
            self._tabs.addTab(self._tabs_dict["jog"], "🕹 Jog")

    # -------------------------------------------------------------
    # Exposing the API Contract to Tabs
    # -------------------------------------------------------------
    def current_project(self) -> dict:
        return self._current_project

    def current_environment(self) -> str | None:
        return self._current_project.get("flash", {}).get("default_environment")

    def runtime_env(self) -> dict:
        env = os.environ.copy()
        env["PAROL6_SERIAL_PORT"] = self._sb_port.text().strip()
        env["PAROL6_BAUD_RATE"] = self._sb_baud.text().strip()
        return env

    def send_serial_text(self, text: str) -> None:
        if self._serial_worker:
            self._serial_worker.send(text)

    def is_serial_connected(self) -> bool:
        return self._serial_worker is not None

    def toggle_serial_connection(self) -> None:
        self._toggle_serial()

    def resolve_path(self, rel_path: str) -> str:
        # Resolve paths relative to the current project_dir
        proj_dir = self._current_project.get("flash", {}).get("project_dir", ".")
        # Fallback to the dir where main.py lives if proj_dir is missing
        base = os.path.abspath(os.path.join(BASE_DIR, proj_dir))
        return os.path.normpath(os.path.join(base, rel_path))

    def _scan_ports(self):
        from core.serial_monitor import list_serial_ports
        from PyQt6.QtWidgets import QInputDialog
        ports = list_serial_ports()
        if not ports:
            QMessageBox.information(self, "No Ports Found", "No serial ports detected.")
            return
        port, ok = QInputDialog.getItem(self, "Select Port", "Available serial ports:", ports, 0, False)
        if ok and port:
            self._sb_port.setText(port)

    # ------------------------------------------------------------------
    # Serial Connection
    # ------------------------------------------------------------------
    def _toggle_serial(self) -> None:
        if self._serial_worker:
            self._serial_worker.stop()
            self._serial_worker = None
            self._sb_connect_btn.setText("⚡ Connect")
            self._sb_connect_btn.setChecked(False)
            if "serial" in self._tabs_dict:
                self._tabs_dict["serial"].disconnect()
            return

        port = self._sb_port.text().strip()
        try:
            baud = int(self._sb_baud.text().strip())
        except ValueError:
            baud = 115200

        if not port:
            QMessageBox.warning(self, "No Port", "Please enter a valid serial port.")
            self._sb_connect_btn.setChecked(False)
            return

        self._serial_worker = SerialWorker(port, baud)
        self._serial_worker.raw_line.connect(self._on_raw_line)
        self._serial_worker.telemetry.connect(self._on_telemetry)
        self._serial_worker.connected.connect(self._on_serial_connected)
        self._serial_worker.packet_rate.connect(self._on_packet_rate)
        self._serial_worker.data_rate.connect(self._on_data_rate)
        
        if "serial" in self._tabs_dict:
            self._serial_worker.error_msg.connect(lambda m: self._tabs_dict["serial"]._append(m, "#f38ba8"))
            self._serial_worker.raw_line.connect(self._tabs_dict["serial"]._on_line)
            
        self._serial_worker.start()
        self._sb_connect_btn.setText("⏹ Disconnect")

    def _on_serial_connected(self, ok: bool) -> None:
        self._sb_conn.setText("🟢 Connected" if ok else "🔴 Disconnected")
        self._sb_conn.setStyleSheet("color:#a6e3a1;" if ok else "color:#f38ba8;")

    def _on_packet_rate(self, rate: float) -> None:
        self._sb_rate.setText(f"{rate:.1f} pkt/s")

    def _on_data_rate(self, bps: float) -> None:
        if bps >= 1048576:
            self._sb_data_rate.setText(f"{bps/1048576.0:.2f} MB/s")
        elif bps >= 1024:
            self._sb_data_rate.setText(f"{bps/1024.0:.1f} KB/s")
        else:
            self._sb_data_rate.setText(f"{bps:.0f} B/s")

    def _on_raw_line(self, line: str) -> None:
        # Route fault lines to fault log
        if "FAULT" in line or "SOFT_ESTOP" in line:
            if "faults" in self._tabs_dict:
                self._tabs_dict["faults"].record_serial_fault(line)

    def _on_telemetry(self, pkt: dict) -> None:
        if "plot" in self._tabs_dict:
            self._tabs_dict["plot"].ingest(pkt)
        if "jog" in self._tabs_dict:
            self._tabs_dict["jog"].ingest_telemetry(pkt)
        if "faults" in self._tabs_dict:
            self._tabs_dict["faults"].ingest_telemetry(pkt)
        if "serial" in self._tabs_dict:
            self._tabs_dict["serial"].update_state_badge(pkt.get("state_byte"))

        state = pkt.get("supervisor_state", "")
        if state:
            self._sb_state.setText(f"State: {state}")

    def closeEvent(self, event) -> None:
        if self._serial_worker:
            self._serial_worker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    app.setFont(QFont("Inter", 11))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
