from __future__ import annotations

import os
import shutil
import subprocess
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIGURATOR_DIR = os.path.join(REPO_ROOT, "parol6_firmware_configurator")
if CONFIGURATOR_DIR not in sys.path:
    sys.path.insert(0, CONFIGURATOR_DIR)

from core.flash_manager import BuildWorker, FlashWorker  # noqa: E402
from core.serial_monitor import SerialWorker, list_serial_ports  # noqa: E402


class ScriptWorker(QThread):
    output_line = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(int)

    def __init__(self, command: list[str], workdir: str | None = None, parent=None):
        super().__init__(parent)
        self._command = command
        self._workdir = workdir
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        self.output_line.emit(f"[RUN] $ {' '.join(self._command)}")
        try:
            self._proc = subprocess.Popen(
                self._command,
                cwd=self._workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._proc.stdout:
                self.output_line.emit(line.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as exc:
            self.output_line.emit(f"[RUN] ERROR: {exc}")
            rc = -1

        if rc == 0:
            self.finished_ok.emit()
        else:
            self.finished_err.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6 Experimental Tool")
        self.resize(1200, 820)

        self._serial_worker: SerialWorker | None = None
        self._build_worker: BuildWorker | None = None
        self._flash_worker: FlashWorker | None = None
        self._script_worker: ScriptWorker | None = None
        self._command_seq = 0
        self._joint_targets: list[QLineEdit] = []
        self._joint_velocities: list[QLineEdit] = []
        self._telemetry_labels: list[QLabel] = []

        central = QWidget()
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.addWidget(self._build_serial_group())
        root.addWidget(self._build_command_group())
        root.addWidget(self._build_flash_group())
        root.addWidget(self._build_log_group(), stretch=1)
        self.setCentralWidget(central)

        self._refresh_ports()
        self._set_busy(False)

    def _build_serial_group(self) -> QGroupBox:
        group = QGroupBox("Serial")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Port"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        layout.addWidget(self.port_combo, 0, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        layout.addWidget(refresh_btn, 0, 2)

        layout.addWidget(QLabel("Baud"), 0, 3)
        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(9600, 2000000)
        self.baud_spin.setSingleStep(115200)
        self.baud_spin.setValue(115200)
        layout.addWidget(self.baud_spin, 0, 4)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_serial)
        layout.addWidget(self.connect_btn, 0, 5)

        self.rate_label = QLabel("0 pkt/s")
        self.state_label = QLabel("state: -")
        self.conn_label = QLabel("disconnected")
        layout.addWidget(self.conn_label, 1, 0, 1, 2)
        layout.addWidget(self.rate_label, 1, 2)
        layout.addWidget(self.state_label, 1, 3, 1, 3)

        return group

    def _build_command_group(self) -> QGroupBox:
        group = QGroupBox("Commands")
        layout = QVBoxLayout(group)

        top = QHBoxLayout()
        enable_btn = QPushButton("Enable")
        enable_btn.clicked.connect(lambda: self._send_text("<ENABLE>"))
        disable_btn = QPushButton("Disable")
        disable_btn.clicked.connect(lambda: self._send_text("<DISABLE>"))
        zero_btn = QPushButton("Zero Pose")
        zero_btn.clicked.connect(lambda: self._send_text("<ZERO>"))
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(lambda: self._send_text("<STOP>"))
        for btn in (enable_btn, disable_btn, zero_btn, stop_btn):
            top.addWidget(btn)
        top.addStretch()
        layout.addLayout(top)

        grid = QGridLayout()
        grid.addWidget(QLabel("Joint"), 0, 0)
        grid.addWidget(QLabel("Target Pos"), 0, 1)
        grid.addWidget(QLabel("FF Vel"), 0, 2)
        grid.addWidget(QLabel("Telemetry"), 0, 3)
        for axis in range(6):
            grid.addWidget(QLabel(f"J{axis + 1}"), axis + 1, 0)
            pos_edit = QLineEdit("0.0")
            vel_edit = QLineEdit("0.0")
            telem = QLabel("pos 0.0000 vel 0.0000")
            self._joint_targets.append(pos_edit)
            self._joint_velocities.append(vel_edit)
            self._telemetry_labels.append(telem)
            grid.addWidget(pos_edit, axis + 1, 1)
            grid.addWidget(vel_edit, axis + 1, 2)
            grid.addWidget(telem, axis + 1, 3)
        layout.addLayout(grid)

        send_btn = QPushButton("Send Joint Command")
        send_btn.clicked.connect(self._send_joint_command)
        layout.addWidget(send_btn)

        raw_row = QHBoxLayout()
        self.raw_input = QLineEdit()
        self.raw_input.setPlaceholderText("<0,0,0,0,0,0,0,0,0,0,0,0,0>")
        raw_send = QPushButton("Send Raw")
        raw_send.clicked.connect(self._send_raw)
        raw_row.addWidget(self.raw_input)
        raw_row.addWidget(raw_send)
        layout.addLayout(raw_row)

        return group

    def _build_flash_group(self) -> QGroupBox:
        group = QGroupBox("Build And Flash")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Project Dir"), 0, 0)
        self.project_edit = QLineEdit(os.path.join(REPO_ROOT, "parol6_experimental_firmware"))
        layout.addWidget(self.project_edit, 0, 1, 1, 3)
        browse_project = QPushButton("Browse")
        browse_project.clicked.connect(self._browse_project_dir)
        layout.addWidget(browse_project, 0, 4)

        layout.addWidget(QLabel("PIO Env"), 1, 0)
        self.env_combo = QComboBox()
        self.env_combo.addItems(["teensy41"])
        layout.addWidget(self.env_combo, 1, 1)

        build_btn = QPushButton("Build")
        build_btn.clicked.connect(self._run_build)
        layout.addWidget(build_btn, 1, 2)

        flash_btn = QPushButton("Flash")
        flash_btn.clicked.connect(self._run_flash)
        layout.addWidget(flash_btn, 1, 3)

        layout.addWidget(QLabel("Script"), 2, 0)
        self.script_edit = QLineEdit()
        layout.addWidget(self.script_edit, 2, 1, 1, 3)
        browse_script = QPushButton("Browse")
        browse_script.clicked.connect(self._browse_script)
        layout.addWidget(browse_script, 2, 4)

        run_script_btn = QPushButton("Run Script")
        run_script_btn.clicked.connect(self._run_script)
        layout.addWidget(run_script_btn, 3, 3)

        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Monitor")
        layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return group

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _set_busy(self, busy: bool) -> None:
        self.connect_btn.setEnabled(not busy or self.connect_btn.isChecked())

    def _refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = list_serial_ports()
        if not ports:
            ports = ["/dev/ttyACM0"]
        self.port_combo.addItems(ports)
        if current:
            self.port_combo.setCurrentText(current)

    def _toggle_serial(self) -> None:
        if self._serial_worker:
            self._serial_worker.stop()
            self._serial_worker = None
            self.connect_btn.setText("Connect")
            self.conn_label.setText("disconnected")
            self.connect_btn.setChecked(False)
            return

        port = self.port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Missing Port", "Set a serial port first.")
            self.connect_btn.setChecked(False)
            return

        self._serial_worker = SerialWorker(port, self.baud_spin.value())
        self._serial_worker.raw_line.connect(self._append_log)
        self._serial_worker.telemetry.connect(self._handle_telemetry)
        self._serial_worker.connected.connect(self._handle_connected)
        self._serial_worker.error_msg.connect(self._append_log)
        self._serial_worker.packet_rate.connect(lambda rate: self.rate_label.setText(f"{rate:.1f} pkt/s"))
        self._serial_worker.start()
        self.connect_btn.setText("Disconnect")

    def _handle_connected(self, connected: bool) -> None:
        self.conn_label.setText("connected" if connected else "disconnected")
        if not connected and self._serial_worker is None:
            self.connect_btn.setText("Connect")

    def _handle_telemetry(self, pkt: dict) -> None:
        state = pkt.get("state_byte")
        self.state_label.setText(f"state: {state}")
        pos = pkt.get("pos", [])
        vel = pkt.get("vel", [])
        for axis, label in enumerate(self._telemetry_labels):
            p = pos[axis] if axis < len(pos) else 0.0
            v = vel[axis] if axis < len(vel) else 0.0
            label.setText(f"pos {p:+.4f} vel {v:+.4f}")

    def _send_text(self, text: str) -> None:
        if not self._serial_worker:
            QMessageBox.warning(self, "Disconnected", "Connect to the Teensy first.")
            return
        self._serial_worker.send(text)
        self._append_log(f"[TX] {text}")

    def _send_joint_command(self) -> None:
        positions = ",".join(edit.text().strip() or "0.0" for edit in self._joint_targets)
        velocities = ",".join(edit.text().strip() or "0.0" for edit in self._joint_velocities)
        text = f"<{self._command_seq},{positions},{velocities}>"
        self._command_seq += 1
        self._send_text(text)

    def _send_raw(self) -> None:
        text = self.raw_input.text().strip()
        if text:
            self._send_text(text)

    def _browse_project_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select PlatformIO Project")
        if path:
            self.project_edit.setText(path)

    def _browse_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Script")
        if path:
            self.script_edit.setText(path)

    def _run_build(self) -> None:
        project_dir = self.project_edit.text().strip()
        if not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Bad Project Dir", "Set a valid PlatformIO project directory.")
            return
        self._build_worker = BuildWorker(project_dir, self.env_combo.currentText())
        self._build_worker.output_line.connect(self._append_log)
        self._build_worker.finished_ok.connect(lambda: self._append_log("[BUILD] done"))
        self._build_worker.finished_err.connect(lambda rc: self._append_log(f"[BUILD] failed rc={rc}"))
        self._build_worker.start()

    def _run_flash(self) -> None:
        project_dir = self.project_edit.text().strip()
        if not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Bad Project Dir", "Set a valid PlatformIO project directory.")
            return
        self._flash_worker = FlashWorker(project_dir, self.env_combo.currentText())
        self._flash_worker.output_line.connect(self._append_log)
        self._flash_worker.finished_ok.connect(lambda: self._append_log("[FLASH] done"))
        self._flash_worker.finished_err.connect(lambda rc: self._append_log(f"[FLASH] failed rc={rc}"))
        self._flash_worker.start()

    def _run_script(self) -> None:
        script = self.script_edit.text().strip()
        if not script:
            QMessageBox.warning(self, "Missing Script", "Choose a script first.")
            return
        if not os.path.exists(script):
            QMessageBox.warning(self, "Missing Script", "The selected script does not exist.")
            return

        if script.endswith(".sh"):
            shell = shutil.which("bash") or "/bin/bash"
            command = [shell, script]
        else:
            command = [script]
        self._script_worker = ScriptWorker(command, workdir=os.path.dirname(script) or None)
        self._script_worker.output_line.connect(self._append_log)
        self._script_worker.finished_ok.connect(lambda: self._append_log("[RUN] done"))
        self._script_worker.finished_err.connect(lambda rc: self._append_log(f"[RUN] failed rc={rc}"))
        self._script_worker.start()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
