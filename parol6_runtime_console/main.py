"""Standalone runtime console for flashing, serial monitoring, plotting, and ROS2 commands."""
from __future__ import annotations

import json
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QToolBar,
)

from parol6_runtime_console.core.serial_monitor import SerialWorker, list_serial_ports
from parol6_runtime_console.tabs.flash_tab import FlashTab
from parol6_runtime_console.tabs.plot_tab import PlotTab
from parol6_runtime_console.tabs.ros2_tab import Ros2Tab
from parol6_runtime_console.tabs.serial_tab import SerialTab


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
QTabBar::tab:selected { background: #313244; color: #89dceb; font-weight: bold; }
QPushButton {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #cdd6f4;
}
QPushButton:hover { background: #45475a; }
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background: #11111b;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
    padding: 3px 6px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: #a6adc8;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QStatusBar { background: #181825; color: #a6adc8; border-top: 1px solid #45475a; }
QToolBar { background: #181825; border-bottom: 1px solid #45475a; spacing: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6 Runtime Console")
        self.resize(1280, 840)
        self._settings = QSettings("PAROL6", "RuntimeConsole")
        self._registry = self._load_registry()
        self._serial_worker: SerialWorker | None = None

        self._build_toolbar()
        self._build_tabs()
        self._build_status_bar()
        self._load_session()

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._plot_tab.refresh)
        self._plot_timer.start(50)

    def _load_registry(self) -> dict:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Runtime")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel("Project:"))
        self._project_combo = QComboBox()
        for project in self._registry.get("projects", []):
            self._project_combo.addItem(project["name"], project)
        self._project_combo.currentIndexChanged.connect(self.refresh_project_bindings)
        tb.addWidget(self._project_combo)

        tb.addSeparator()
        tb.addWidget(QLabel("Env:"))
        self._env_combo = QComboBox()
        self._env_combo.setEditable(True)
        tb.addWidget(self._env_combo)
        env_scan_btn = QPushButton("Reload Envs")
        env_scan_btn.clicked.connect(self._reload_envs_from_current_project)
        tb.addWidget(env_scan_btn)

        tb.addSeparator()
        tb.addWidget(QLabel("Port:"))
        self._port_edit = QLineEdit()
        self._port_edit.setMinimumWidth(180)
        tb.addWidget(self._port_edit)

        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self._scan_ports)
        tb.addWidget(scan_btn)

        tb.addSeparator()
        tb.addWidget(QLabel("Baud:"))
        self._baud_edit = QLineEdit("115200")
        self._baud_edit.setFixedWidth(90)
        tb.addWidget(self._baud_edit)

    def _build_tabs(self) -> None:
        self._tabs = QTabWidget()
        self._serial_tab = SerialTab(self)
        self._plot_tab = PlotTab(self)
        self._flash_tab = FlashTab(self)
        self._ros2_tab = Ros2Tab(self)
        self._tabs.addTab(self._serial_tab, "Serial")
        self._tabs.addTab(self._plot_tab, "Oscilloscope")
        self._tabs.addTab(self._flash_tab, "Flash")
        self._tabs.addTab(self._ros2_tab, "ROS2")
        self.setCentralWidget(self._tabs)

        self._serial_tab.connect_requested.connect(self._toggle_serial)
        self.refresh_project_bindings()

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_conn = QLabel("Disconnected")
        self._status_rate = QLabel("0 pkt/s")
        self._status_data = QLabel("0 B/s")
        sb.addPermanentWidget(self._status_conn)
        sb.addPermanentWidget(QLabel("|"))
        sb.addPermanentWidget(self._status_rate)
        sb.addPermanentWidget(QLabel("|"))
        sb.addPermanentWidget(self._status_data)

    def _load_session(self) -> None:
        last_project = self._settings.value("last_project_id", "")
        for idx in range(self._project_combo.count()):
            project = self._project_combo.itemData(idx)
            if project and project.get("id") == last_project:
                self._project_combo.setCurrentIndex(idx)
                break

        self._port_edit.setText(self._settings.value("serial_port", self._port_edit.text()))
        self._baud_edit.setText(self._settings.value("serial_baud", self._baud_edit.text()))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        project = self.current_project()
        self._settings.setValue("last_project_id", project.get("id", "") if project else "")
        self._settings.setValue("serial_port", self._port_edit.text().strip())
        self._settings.setValue("serial_baud", self._baud_edit.text().strip())
        if self._serial_worker:
            self._serial_worker.stop()
        super().closeEvent(event)

    def current_project(self) -> dict | None:
        data = self._project_combo.currentData()
        return data if isinstance(data, dict) else None

    def current_environment(self) -> str:
        return self._env_combo.currentText().strip()

    def resolve_path(self, rel_path: str) -> str:
        return os.path.abspath(os.path.join(BASE_DIR, "..", rel_path))

    def runtime_env(self) -> dict[str, str]:
        env = {}
        port = self._port_edit.text().strip()
        baud = self._baud_edit.text().strip()
        if port:
            env["PAROL6_SERIAL_PORT"] = port
        if baud:
            env["PAROL6_SERIAL_BAUD"] = baud
        return env

    def refresh_project_bindings(self) -> None:
        project = self.current_project()
        if not project:
            self._env_combo.clear()
            self._flash_tab.load_project(None)
            self._ros2_tab.load_project(None)
            return

        self._port_edit.setText(project.get("default_serial_port", self._port_edit.text().strip()))
        self._baud_edit.setText(str(project.get("default_baud", self._baud_edit.text().strip() or 115200)))
        self._flash_tab.load_project(project)
        self._ros2_tab.load_project(project)
        self._reload_envs_from_current_project()

    def _scan_ports(self) -> None:
        ports = list_serial_ports()
        if ports:
            self._port_edit.setText(ports[0])
        else:
            QMessageBox.information(
                self,
                "No Ports Found",
                "No serial ports were detected.\n\n"
                "If you are flashing STM32 over ST-Link, that is normal: ST-Link is not a serial port.\n"
                "Use the Flash tab for upload and only use Port/Scan for serial monitor access.",
            )

    def _reload_envs_from_current_project(self, project_dir_override: str | None = None) -> None:
        project = self.current_project()
        current_text = self._env_combo.currentText().strip()
        self._env_combo.clear()

        envs: list[str] = []
        project_dir = ""
        if project:
            flash_cfg = project.get("flash", {})
            project_dir = project_dir_override or self.resolve_path(flash_cfg.get("project_dir", ""))
            envs = self._discover_platformio_envs(project_dir)
            if not envs:
                envs = list(flash_cfg.get("environments", []))

        for env_name in envs:
            self._env_combo.addItem(env_name)

        if current_text:
            if self._env_combo.findText(current_text) < 0:
                self._env_combo.addItem(current_text)
            self._env_combo.setCurrentText(current_text)
        elif envs:
            self._env_combo.setCurrentText(envs[0])

    def _discover_platformio_envs(self, project_dir: str) -> list[str]:
        if not project_dir:
            return []
        ini_path = os.path.join(project_dir, "platformio.ini")
        if not os.path.isfile(ini_path):
            return []
        envs: list[str] = []
        pattern = re.compile(r"^\[env:([^\]]+)\]\s*$")
        with open(ini_path, "r", encoding="utf-8") as fh:
            for line in fh:
                match = pattern.match(line.strip())
                if match:
                    envs.append(match.group(1))
        return envs

    def _toggle_serial(self) -> None:
        if self._serial_worker:
            self._serial_worker.stop()
            self._serial_worker = None
            self._serial_tab.set_connected(False)
            self._status_conn.setText("Disconnected")
            return

        port = self._port_edit.text().strip()
        try:
            baud = int(self._baud_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Invalid Baud", "Baud must be an integer.")
            self._serial_tab.set_connected(False)
            return

        if not port:
            QMessageBox.critical(self, "No Port", "Select a serial port first.")
            self._serial_tab.set_connected(False)
            return

        self._serial_worker = SerialWorker(port, baud)
        self._serial_worker.raw_line.connect(self._on_serial_line)
        self._serial_worker.telemetry.connect(self._plot_tab.ingest)
        self._serial_worker.error_msg.connect(lambda msg: self._serial_tab.append_line(msg, "#f38ba8"))
        self._serial_worker.connected.connect(self._on_serial_connected)
        self._serial_worker.packet_rate.connect(lambda rate: self._status_rate.setText(f"{rate:.1f} pkt/s"))
        self._serial_worker.data_rate.connect(lambda rate: self._status_data.setText(f"{rate:.0f} B/s"))
        self._serial_worker.start()

    def _on_serial_connected(self, connected: bool) -> None:
        self._serial_tab.set_connected(connected)
        self._status_conn.setText("Connected" if connected else "Disconnected")
        if not connected:
            self._serial_worker = None

    def _on_serial_line(self, line: str) -> None:
        color = "#f9e2af" if "FAULT" in line or "ERR" in line.upper() else "#cdd6f4"
        self._serial_tab.append_line(line, color)

    def send_serial_text(self, text: str) -> None:
        if self._serial_worker:
            self._serial_worker.send(text)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
