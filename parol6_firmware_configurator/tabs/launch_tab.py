"""
launch_tab.py — GUI wrapper for run_robot.sh.
Allows launching RViz, Gazebo, or Real Hardware modes directly from the configurator.
"""
import os
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QGroupBox, QComboBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from PyQt6.QtGui import QFont

class LaunchWorker(QThread):
    """Runs a bash script in a background thread and emits its output."""
    output_line = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(int)

    def __init__(self, script_path: str, args: list[str], parent=None):
        super().__init__(parent)
        self._script_path = script_path
        self._args = args
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        cmd = [self._script_path] + self._args
        self.output_line.emit(f"[LAUNCH] $ {' '.join(cmd)}")
        try:
            env = os.environ.copy()
            # Ensure standard binary paths are present just in case the GUI was launched strangely
            if "PATH" in env:
                env["PATH"] += os.pathsep + "/usr/local/bin:/usr/bin:/bin"
            else:
                env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in self._proc.stdout:
                self.output_line.emit(line.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as e:
            self.output_line.emit(f"[LAUNCH] ERROR: {e}")
            rc = -1

        if rc == 0:
            self.output_line.emit("[LAUNCH] ✅ Process exited cleanly.")
            self.finished_ok.emit()
        else:
            self.output_line.emit(f"[LAUNCH] ❌ Process failed or was terminated (code {rc}).")
            self.finished_err.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


class LaunchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: LaunchWorker | None = None
        
        # Resolve the scripts/launchers directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._launchers_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "scripts", "launchers"))
        
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("🚀 ROS2 / MoveIt Launcher")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)
        
        if not os.path.exists(self._launchers_dir):
            err = QLabel(f"⚠️ Launchers directory not found at:\n{self._launchers_dir}")
            err.setStyleSheet("color:#f38ba8; font-weight:bold;")
            root.addWidget(err)
            root.addStretch()
            return

        # ── Controls ────────────────────────────────────────────────────────
        ctrls = QGroupBox("Launch Mode")
        cl = QHBoxLayout(ctrls)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Method 1: Gazebo Only (Simulation World)", "launch_gazebo_only.sh")
        self.mode_combo.addItem("Method 2: MoveIt With Gazebo (External)", "launch_moveit_with_gazebo.sh")
        self.mode_combo.addItem("Method 3: MoveIt Fake (Standalone RViz)", "launch_moveit_fake.sh")
        self.mode_combo.addItem("Method 4: MoveIt Real Hardware", "launch_moveit_real_hw.sh")
        self.mode_combo.setMinimumWidth(300)
        cl.addWidget(QLabel("Target:"))
        cl.addWidget(self.mode_combo)
        
        self.launch_btn = QPushButton("🚀 Launch")
        self.launch_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")
        self.launch_btn.clicked.connect(self._toggle_launch)
        cl.addWidget(self.launch_btn)
        
        cl.addStretch()
        root.addWidget(ctrls)

        # ── Output Log ──────────────────────────────────────────────────────
        log_group = QGroupBox("Terminal Output (RViz / Gazebo Logs)")
        ll = QVBoxLayout(log_group)
        
        self.log_out = QTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setFont(QFont("Monospace", 10))
        self.log_out.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_out.setStyleSheet("background:#11111b; color:#a6adc8;")
        ll.addWidget(self.log_out)
        
        root.addWidget(log_group)

    def _toggle_launch(self) -> None:
        if self._worker:
            # Terminate current process
            self.log_out.append("[LAUNCH] Stopping process...")
            self._worker.abort()
            self._worker = None
            self._set_button_state(False)
            return

        # Start new process
        self.log_out.clear()
        script_name = self.mode_combo.currentData()
        script_path = os.path.join(self._launchers_dir, script_name)
        
        if not os.path.exists(script_path):
            self.log_out.append(f"[LAUNCH] ❌ Error: Cannot find script {script_path}")
            return
            
        self._worker = LaunchWorker(script_path, [])
        self._worker.output_line.connect(self.log_out.append)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_err.connect(self._on_finished)
        self._worker.start()
        
        self._set_button_state(True)

    def _set_button_state(self, is_running: bool) -> None:
        if is_running:
            self.launch_btn.setText("🛑 Stop")
            self.launch_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
            self.mode_combo.setEnabled(False)
        else:
            self.launch_btn.setText("🚀 Launch")
            self.launch_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")
            self.mode_combo.setEnabled(True)

    def _on_finished(self) -> None:
        self._worker = None
        self._set_button_state(False)
