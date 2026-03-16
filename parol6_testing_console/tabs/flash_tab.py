import os
import re
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QComboBox, QGroupBox, QSizePolicy,
    QLineEdit, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from core.process_workers import ProcessWorker
from core.diagnostics import build_diagnostic_report
from core.gui_theme import QPushButton

class FlashTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        self._build_worker = None
        self._dfu_poll_timer = QTimer()
        self._dfu_poll_timer.setInterval(2000)  # Every 2s
        self._dfu_poll_timer.timeout.connect(self._probe_usb)
        self._build_ui()
        self._refresh_diagnostics()
        self._refresh_envs()
        self._probe_usb()
        self._dfu_poll_timer.start()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Toolchain Diagnostics ──
        self.diag_box = QGroupBox("Toolchain Diagnostics")
        self.diag_box.setStyleSheet("color: #f38ba8;")
        diag_layout = QVBoxLayout(self.diag_box)
        self.diag_lbl = QLabel("Checking tools...")
        self.diag_lbl.setWordWrap(True)
        diag_layout.addWidget(self.diag_lbl)
        layout.addWidget(self.diag_box)

        # ── USB / DFU Status Panel ──
        dfu_box = QGroupBox("USB / DFU Device Status")
        dfu_layout = QHBoxLayout(dfu_box)

        self.dfu_status_lbl = QLabel("🔍 Scanning...")
        self.dfu_status_lbl.setWordWrap(True)
        dfu_layout.addWidget(self.dfu_status_lbl, stretch=1)

        probe_btn = QPushButton("🔍 Probe USB")
        probe_btn.setToolTip("Scan USB for DFU devices and serial ports")
        probe_btn.clicked.connect(self._probe_usb)
        dfu_layout.addWidget(probe_btn)

        self.detach_btn = QPushButton("⏏ Detach DFU")
        self.detach_btn.setToolTip("Send DFU detach command so the board boots into the new firmware")
        self.detach_btn.setEnabled(False)
        self.detach_btn.clicked.connect(self._detach_dfu)
        dfu_layout.addWidget(self.detach_btn)

        self.dfu_flash_btn = QPushButton("⚡ Flash via DFU")
        self.dfu_flash_btn.setToolTip("Board is in DFU mode — flash immediately without a serial connection")
        self.dfu_flash_btn.setEnabled(False)
        self.dfu_flash_btn.setStyleSheet("background-color: #cba6f7; border: 1px solid #1e1e2e; color: #1e1e2e; font-weight: bold;")
        self.dfu_flash_btn.clicked.connect(self._start_upload)
        dfu_layout.addWidget(self.dfu_flash_btn)

        layout.addWidget(dfu_box)

        # ── Flash Control Box ──
        ctrl_box = QGroupBox("PlatformIO Flash Orchestration")
        ctrl_layout = QHBoxLayout(ctrl_box)

        ctrl_layout.addWidget(QLabel("Environment:"))
        self.env_combo = QComboBox()
        self.env_combo.setMinimumWidth(200)
        ctrl_layout.addWidget(self.env_combo)

        ctrl_layout.addWidget(QLabel("  Project Dir:"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setMinimumWidth(200)
        self.dir_edit.setStyleSheet("font-family: monospace; color: #a6adc8;")
        self.dir_edit.textChanged.connect(self._refresh_envs_from_path)
        ctrl_layout.addWidget(self.dir_edit)

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse_dir)
        ctrl_layout.addWidget(browse_btn)

        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ctrl_layout.addWidget(sp)

        self.upload_btn = QPushButton("🚀 Build & Upload")
        self.upload_btn.setStyleSheet("background-color: #a6e3a1; border: 1px solid #1e1e2e; color: #1e1e2e; font-weight: bold;")
        self.upload_btn.clicked.connect(self._start_upload)
        ctrl_layout.addWidget(self.upload_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_worker)
        ctrl_layout.addWidget(self.stop_btn)

        layout.addWidget(ctrl_box)

        # ── Logs ──
        self.log_primary = QTextEdit()
        self.log_primary.setReadOnly(True)
        self.log_primary.setStyleSheet("font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 12px; background: #11111b;")
        layout.addWidget(self.log_primary, stretch=1)

    # ------------------------------------------------------------------
    # USB / DFU Probe
    # ------------------------------------------------------------------
    def _probe_usb(self) -> None:
        """Scan USB via dfu-util and /dev/ttyACM* to show board state."""
        lines = []
        in_dfu = False
        in_serial = False

        # Check DFU
        try:
            result = subprocess.run(
                ["dfu-util", "-l"], capture_output=True, text=True, timeout=3
            )
            combined = result.stdout + result.stderr
            dfu_devices = [l for l in combined.splitlines() if "Found DFU" in l]
            if dfu_devices:
                in_dfu = True
                lines.append(f"<span style='color:#cba6f7;'>⚡ DFU mode detected ({len(dfu_devices)} interface(s)):</span>")
                for d in dfu_devices[:2]:
                    # Show just the key parts
                    m = re.search(r'name="([^"]+)"', d)
                    name = m.group(1) if m else d[:60]
                    lines.append(f"&nbsp;&nbsp;• {name}")
        except Exception:
            pass

        # Check serial ACM
        import glob
        acm_ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        if acm_ports:
            in_serial = True
            lines.append(f"<span style='color:#a6e3a1;'>🟢 Serial port(s): {', '.join(acm_ports)}</span>")

        if not in_dfu and not in_serial:
            lines.append("<span style='color:#f38ba8;'>🔴 No STM32 device detected on USB</span>")
            lines.append("<span style='color:#a6adc8;'>&nbsp;&nbsp;→ Press NRST on the board to reboot, or hold BOOT0+NRST for DFU mode</span>")

        self.dfu_status_lbl.setText("<br>".join(lines))
        self.detach_btn.setEnabled(in_dfu)
        self.dfu_flash_btn.setEnabled(in_dfu)

    def _detach_dfu(self) -> None:
        """Send DFU detach command so the chip reboots into the new firmware."""
        self.log_primary.append("<span style='color:#f9e2af;'>==> Sending DFU detach (dfu-util -e)...</span>")
        try:
            result = subprocess.run(
                ["dfu-util", "-e"], capture_output=True, text=True, timeout=5
            )
            out = result.stdout + result.stderr
            for line in out.splitlines():
                self.log_primary.append(line)
            self.log_primary.append("<span style='color:#a6e3a1;'>==> Detach sent — board should reboot now.</span>")
        except Exception as e:
            self.log_primary.append(f"<span style='color:#f38ba8;'>Detach error: {e}</span>")
        self._probe_usb()


    def _refresh_diagnostics(self):
        project = self._main_window.current_project()
        report = build_diagnostic_report(project)
        
        if report["is_ok"]:
            self.diag_box.setTitle("Toolchain Diagnostics (OK)")
            self.diag_box.setStyleSheet("QGroupBox { color: #a6e3a1; }")
            self.diag_lbl.setText("All probably required toolchains (like pio or related tools) are present on this system.")
            self.diag_lbl.setStyleSheet("color: #cdd6f4;")
        else:
            self.diag_box.setTitle("Toolchain Diagnostics (WARNING)")
            self.diag_box.setStyleSheet("QGroupBox { color: #f38ba8; }")
            missing = ", ".join(report["missing_required"])
            toolhint = project.get("flash", {}).get("tooling_hint", "")
            msg = f"Missing required tooling from PATH: <b>{missing}</b><br><br>"
            msg += f"<i>Project Flash Hint:</i> {toolhint}"
            self.diag_lbl.setText(msg)
            self.diag_lbl.setStyleSheet("color: #f38ba8;")

    def _refresh_envs(self):
        self.env_combo.clear()
        project = self._main_window.current_project()
        
        # Check if flashing is supported
        flash_cfg = project.get("flash", {})
        if not flash_cfg:
            self.env_combo.addItem("Flashing not configured for this project")
            self.env_combo.setEnabled(False)
            self.upload_btn.setEnabled(False)
            return

        proj_dir = self._main_window.resolve_path(flash_cfg.get("project_dir", "."))
        
        # Unhook temporarily so we don't trigger _refresh_envs_from_path infinitely
        self.dir_edit.blockSignals(True)
        self.dir_edit.setText(proj_dir)
        self.dir_edit.blockSignals(False)
        
        self._refresh_envs_from_path(proj_dir)
        
        # Select default
        def_env = flash_cfg.get("default_environment", "")
        if def_env:
            idx = self.env_combo.findText(def_env)
            if idx >= 0:
                self.env_combo.setCurrentIndex(idx)


    def _refresh_envs_from_path(self, proj_dir: str):
        self.env_combo.clear()
        
        ini_path = os.path.join(proj_dir, "platformio.ini")
        if not os.path.exists(ini_path):
            self.env_combo.addItem(f"platformio.ini not found")
            self.env_combo.setEnabled(False)
            self.upload_btn.setEnabled(False)
            return
            
        # Parse PlatformIO environments
        envs = []
        try:
            with open(ini_path, 'r', encoding='utf-8') as f:
                for line in f:
                    m = re.match(r'^\[env:([^\]]+)\]', line.strip())
                    if m:
                        envs.append(m.group(1))
        except Exception as e:
            self.log_primary.append(f"<span style='color:#f38ba8;'>Failed to read platformio.ini: {e}</span>")
            
        if not envs:
            self.env_combo.addItem("No [env:...] found in platformio.ini")
            self.env_combo.setEnabled(False)
            self.upload_btn.setEnabled(False)
            return
            
        for env in envs:
            self.env_combo.addItem(env)
            
        self.env_combo.setEnabled(True)
        self.upload_btn.setEnabled(True)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select firmware directory")
        if d:
            self.dir_edit.setText(d)

    def _start_upload(self) -> None:
        env = self.env_combo.currentText()
        if not env or not self.env_combo.isEnabled():
            return
            
        proj_dir = self.dir_edit.text()
        project = self._main_window.current_project()

        # Check for Software DFU Reboot capability
        caps = project.get("capabilities", {})
        if caps.get("supports_dfu_reboot", False):
            self.log_primary.append("<span style='color:#f9e2af;'>==> Attempting Software DFU Reboot before upload...</span>")
            self._main_window.send_serial_text("<REBOOT_DFU>")
            import time
            time.sleep(1.0) # Wait longer for STM32 to restart in DFU mode

        # Disconnect GUI serial so Pio can access the port
        if self._main_window.is_serial_connected():
            self.log_primary.append("<span style='color:#f9e2af;'>==> Auto-disconnecting serial port for flashing...</span>")
            self._main_window.toggle_serial_connection()
            self._auto_reconnect_serial = True
        else:
            self._auto_reconnect_serial = False

        self.log_primary.append(f"<span style='color:#89b4fa;'>==> Running: pio run --target upload -e {env}</span>")

        cmd = ["pio", "run", "--target", "upload", "-e", env]
        self._build_worker = ProcessWorker(cmd=cmd, cwd=proj_dir, env=self._main_window.runtime_env())
        
        self._build_worker.output_line.connect(self._on_log)
        self._build_worker.finished_ok.connect(self._on_done)
        self._build_worker.finished_err.connect(self._on_error)
        
        self.upload_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._build_worker.start()

    def _stop_worker(self) -> None:
        if self._build_worker:
            self.log_primary.append("<span style='color:#f9e2af;'>Stopping PlatformIO process...</span>")
            self._build_worker.stop()
            self._build_worker = None
        self._reset_btns()

    def _on_log(self, line: str, color: str = "") -> None:
        if color:
            self.log_primary.append(f"<span style='color:{color};'>{line}</span>")
        else:
            # Colorize PIO output
            if "SUCCESS" in line:
                line = f"<span style='color:#a6e3a1;'>{line}</span>"
            elif "FAILED" in line or "Error" in line:
                line = f"<span style='color:#f38ba8;'>{line}</span>"
            elif "Processing" in line:
                line = f"<span style='color:#89b4fa; font-weight:bold;'>{line}</span>"
            self.log_primary.append(line)

    def _on_done(self) -> None:
        self.log_primary.append("<span style='color:#a6e3a1; font-weight:bold;'>==> Upload Completed Successfully.</span>")
        # Auto-detach if the board is still in DFU mode so it boots the new firmware
        try:
            result = subprocess.run(
                ["dfu-util", "-e"], capture_output=True, text=True, timeout=5
            )
            if "Transitioning" in result.stdout or "Resetting" in result.stdout or result.returncode == 0:
                self.log_primary.append("<span style='color:#89dceb;'>==> DFU detach sent — board is booting new firmware.</span>")
            else:
                # No DFU device found — that's fine, board was already in serial mode
                pass
        except Exception:
            pass
        self._reset_btns()
        # Give the board time to enumerate then re-probe
        QTimer.singleShot(2500, self._probe_usb)

    def _on_error(self, code: int) -> None:
        self.log_primary.append(f"<span style='color:#f38ba8; font-weight:bold;'>==> Upload Process Failed (Exit logic code {code}).</span>")
        self._reset_btns()

    def _reset_btns(self) -> None:
        self.upload_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._build_worker = None
        
        if getattr(self, "_auto_reconnect_serial", False):
            if not self._main_window.is_serial_connected():
                self.log_primary.append("<span style='color:#f9e2af;'>==> Auto-reconnecting serial port...</span>")
                self._main_window.toggle_serial_connection()
            self._auto_reconnect_serial = False
