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
    output_rviz = pyqtSignal(str)
    output_gazebo = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(int)

    def __init__(self, script_path: str, args: list[str], parent=None):
        super().__init__(parent)
        self._script_path = script_path
        self._args = args
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        cmd = [self._script_path] + self._args
        msg = f"[LAUNCH] $ {' '.join(cmd)}"
        self.output_rviz.emit(msg)
        self.output_gazebo.emit(msg)
        
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
                line = line.rstrip()
                # Basic heuristic: if it mentions ruby, ign, gazebo, or spawn, it's probably physics
                lower = line.lower()
                if "ign" in lower or "gazebo" in lower or "/usr/bin/ruby" in lower or "spawn" in lower:
                    self.output_gazebo.emit(line)
                else:
                    self.output_rviz.emit(line)
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as e:
            msg = f"[LAUNCH] ERROR: {e}"
            self.output_rviz.emit(msg)
            self.output_gazebo.emit(msg)
            rc = -1

        msg = "[LAUNCH] Process Exited." if rc == 0 else f"[LAUNCH] ❌ Stopped (code {rc})"
        self.output_rviz.emit(msg)
        self.output_gazebo.emit(msg)
        
        if rc == 0:
            self.finished_ok.emit()
        else:
            self.finished_err.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            # Send SIGINT (Ctrl+C) instead of terminate so the bash script's trap catches it
            import signal
            os.kill(self._proc.pid, signal.SIGINT)
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
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
        self.mode_combo.addItem("Method 2: Gazebo AND MoveIt (Simulated)", "launch_moveit_with_gazebo.sh")
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

        # ── Output Logs (Split) ─────────────────────────────────────────────
        logs_layout = QHBoxLayout()
        logs_layout.setContentsMargins(0, 0, 0, 0)
        
        # RViz / MoveIt
        rviz_group = QGroupBox("ROS 2 / MoveIt Logs")
        rl = QVBoxLayout(rviz_group)
        self.log_rviz = QTextEdit()
        self.log_rviz.setReadOnly(True)
        self.log_rviz.setFont(QFont("Monospace", 9))
        self.log_rviz.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_rviz.setStyleSheet("background:#11111b; color:#a6adc8;")
        rl.addWidget(self.log_rviz)
        logs_layout.addWidget(rviz_group)
        
        # Gazebo
        gazebo_group = QGroupBox("Gazebo / Physics Logs")
        gl = QVBoxLayout(gazebo_group)
        self.log_gazebo = QTextEdit()
        self.log_gazebo.setReadOnly(True)
        self.log_gazebo.setFont(QFont("Monospace", 9))
        self.log_gazebo.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_gazebo.setStyleSheet("background:#11111b; color:#89b4fa;")
        gl.addWidget(self.log_gazebo)
        logs_layout.addWidget(gazebo_group)
        
        root.addLayout(logs_layout)

    def _toggle_launch(self) -> None:
        if self._worker:
            # Terminate current process
            self.log_rviz.append("[LAUNCH] Stopping process...")
            self.log_gazebo.append("[LAUNCH] Stopping process...")
            self._worker.abort()
            self._worker = None
            self._set_button_state(False)
            return

        # Start new process
        self.log_rviz.clear()
        self.log_gazebo.clear()
        script_name = self.mode_combo.currentData()
        script_path = os.path.join(self._launchers_dir, script_name)
        
        if not os.path.exists(script_path):
            self.log_rviz.append(f"[LAUNCH] ❌ Error: Cannot find script {script_path}")
            return
            
        self._worker = LaunchWorker(script_path, [])
        self._worker.output_rviz.connect(self.log_rviz.append)
        self._worker.output_gazebo.connect(self.log_gazebo.append)
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
