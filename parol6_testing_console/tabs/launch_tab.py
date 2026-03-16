"""
launch_tab.py — GUI wrapper for run_robot.sh.
Allows launching RViz, Gazebo, or Real Hardware modes directly from the console.
"""
import os
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QGroupBox, QComboBox, QMessageBox, QLineEdit, QSlider
)
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from PyQt6.QtGui import QFont
import serial.tools.list_ports

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
            # Inject serial port if provided as a kwarg
            if hasattr(self, '_env_extras'):
                env.update(self._env_extras)
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
    def __init__(self, main_window):
        super().__init__()
        self._main_window = main_window
        self._worker: LaunchWorker | None = None
        self._test_worker: LaunchWorker | None = None
        
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

        hint = QLabel(
            "📋 <b>Method 1</b> Gazebo Only — load the 3D simulation world (no robot control).  "
            "<b>Method 2</b> Gazebo + MoveIt — full simulated robot you can plan and execute on.  "
            "<b>Method 3</b> Fake Hardware — RViz + fake joint states (no Teensy needed, for path planning).  "
            "<b>Method 4</b> Real Hardware (Current) — hardware bringup first, then MoveIt.  "
            "<b>Method 5</b> Real Hardware (Tested Single-Motor Legacy) — branch-locked bringup used in the verified single-motor branch.  "
            "<span style='color:#fab387;'>⚠️ Only use real hardware methods after flashing firmware, homing, and testing limit switches.</span>  "
            "<b>☠️ Kill All</b> forcefully stops all running ROS 2 / Gazebo processes if something hangs."
        )
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "background:#1e1a2e; border:1px solid #cba6f7; border-radius:6px; "
            "color:#cdd6f4; font-size:11px; padding:6px 10px; margin-bottom:4px;"
        )
        root.addWidget(hint)

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
        self.mode_combo.addItem("Method 5: MoveIt Real Hardware (Tested Single-Motor Legacy)", "launch_moveit_real_hw_tested_single_motor.sh")
        self.mode_combo.setMinimumWidth(300)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        cl.addWidget(QLabel("Target:"))
        cl.addWidget(self.mode_combo)
        
        self.launch_btn = QPushButton("🚀 Launch")
        self.launch_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")
        self.launch_btn.clicked.connect(self._toggle_launch)
        cl.addWidget(self.launch_btn)

        self.kill_btn = QPushButton("☠️ Kill All")
        self.kill_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
        self.kill_btn.clicked.connect(self._kill_all_nodes)
        self.kill_btn.setToolTip("Forcefully kills all Gazebo, RViz, and MoveIt processes to clean up the environment.")
        cl.addWidget(self.kill_btn)
        
        # Spacer for separation
        cl.addSpacing(20)
        
        cl.addWidget(QLabel("Test Plan:"))
        self.test_shape_combo = QComboBox()
        self.test_shape_combo.addItems(["Straight", "Curve", "Circle", "ZigZag", "Live Camera (No Inject)"])
        cl.addWidget(self.test_shape_combo)
        
        self.test_btn = QPushButton("▶️ Run Auto-Test")
        self.test_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")
        self.test_btn.clicked.connect(self._run_auto_test)
        self.test_btn.setToolTip("Starts the moveit_controller, injects the selected path shape, and executes it.")
        cl.addWidget(self.test_btn)
        
        cl.addStretch()
        root.addWidget(ctrls)

        # ── Added: Manual Joint Jog (ROS2) ──────────────────────────────────
        self._jog_box = QGroupBox("Manual Joint Jog (MoveIt Forward Position Controller)")
        self._jog_box.setVisible(False)
        jog_layout = QVBoxLayout(self._jog_box)
        
        self._jog_sliders = []
        self._jog_labels = []
        
        for i in range(6):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"J{i+1}:"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-314, 314) # -3.14 to 3.14 rad * 100
            slider.setValue(0)
            
            lbl = QLabel("0.00 rad")
            lbl.setMinimumWidth(60)
            
            # Connect slider to update label and publish ROS2 message
            slider.valueChanged.connect(lambda val, idx=i, label=lbl: self._on_jog_slider_changed(idx, val, label))
            slider.sliderReleased.connect(self._publish_jog_target)
            
            row.addWidget(slider, stretch=1)
            row.addWidget(lbl)
            
            self._jog_sliders.append(slider)
            self._jog_labels.append(lbl)
            jog_layout.addLayout(row)
            
        root.addWidget(self._jog_box)

        # ── Real Hardware Options (Method 4 only) ───────────────────────────
        self._hw_options = QGroupBox("🔌 Real Hardware Settings")
        hw_lay = QHBoxLayout(self._hw_options)

        hw_lay.addWidget(QLabel("Serial Port:"))
        self._port_combo = QComboBox()
        self._port_combo.setEditable(True)
        self._port_combo.setMinimumWidth(160)
        self._refresh_ports()
        hw_lay.addWidget(self._port_combo)

        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedWidth(32)
        refresh_btn.setToolTip("Refresh available serial ports")
        refresh_btn.clicked.connect(self._refresh_ports)
        hw_lay.addWidget(refresh_btn)

        hw_lay.addSpacing(16)
        hw_lay.addWidget(QLabel("Baud:"))
        self._baud_combo = QComboBox()
        self._baud_combo.addItems(["115200", "57600", "250000", "500000", "1000000"])
        self._baud_combo.setCurrentText("115200")
        hw_lay.addWidget(self._baud_combo)

        hw_lay.addSpacing(16)
        note = QLabel("⚠️ Disconnect GUI serial before launching!")
        note.setStyleSheet("color:#fab387; font-size:10px;")
        hw_lay.addWidget(note)
        hw_lay.addStretch()

        root.addWidget(self._hw_options)
        self._on_mode_changed()  # Set initial visibility

        # ── Added: Description ──────────────────────────────────────────────
        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setStyleSheet(
            "background:#1f2430; border:1px solid #4c566a; border-radius:6px; padding:6px 10px;"
        )
        root.addWidget(self.description)
        self._sync_description()

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
        # Pass serial port env vars for Method 4
        if script_name == "launch_moveit_real_hw.sh" or script_name == "launch_moveit_real_hw_tested_single_motor.sh":
            port = self._port_combo.currentText().strip()
            baud = self._baud_combo.currentText().strip()
            if port:
                self._worker._env_extras = {
                    "PAROL6_SERIAL_PORT": port,
                    "PAROL6_BAUD_RATE": baud,
                }
                self.log_rviz.append(f"[LAUNCH] Using port={port} baud={baud}")
        self._worker.output_rviz.connect(self.log_rviz.append)
        self._worker.output_gazebo.connect(self.log_gazebo.append)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_err.connect(self._on_finished)
        self._worker.start()
        
        self._set_button_state(True)

    def _on_mode_changed(self) -> None:
        """Show/hide hardware settings depending on selected launch mode."""
        data = self.mode_combo.currentData()
        is_real_hw = data in ("launch_moveit_real_hw.sh", "launch_moveit_real_hw_tested_single_motor.sh")
        self._hw_options.setVisible(is_real_hw)
        self._sync_description()

    def _sync_description(self) -> None:
        if not hasattr(self, 'description'):
            return
        data = self.mode_combo.currentData()
        text = "No description available."
        if data == "launch_gazebo_only.sh":
            text = "<b>Gazebo Only:</b> Spawns the robot in an empty Gazebo world. The Joint State Broadcaster is active, but MoveIt is NOT running. You cannot plan paths."
        elif data == "launch_moveit_with_gazebo.sh":
            text = "<b>Gazebo + MoveIt:</b> Full simulation testing. Spawns the robot in Gazebo and launches MoveIt controllers. The ROS2 Jog sliders will command the simulated robot."
        elif data == "launch_moveit_fake.sh":
            text = "<b>Fake Hardware:</b> Launches RViz and MoveIt using fake joint states. Excellent for testing trajectory planning logic without a physics engine overhead."
        elif data == "launch_moveit_real_hw.sh":
            text = "<b>Real Hardware:</b> Standard bringup for the PAROL6 using ros2_control to talk to the Teensy. Do not use unless limits are tested and homing is complete."
        elif data == "launch_moveit_real_hw_tested_single_motor.sh":
            text = "<b>Real Hardware (Single Motor Test):</b> Uses the isolated verified launch parameters from the previous 1-motor branch."
        
        self.description.setText(text)

    def _refresh_ports(self) -> None:
        """Populate the port combo with currently visible serial devices."""
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            ports = []
        # Also add common Linux paths that comports() may miss inside Docker
        for p in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]:
            if p not in ports and os.path.exists(p):
                ports.append(p)
        current = self._port_combo.currentText()
        self._port_combo.clear()
        self._port_combo.addItems(ports if ports else ["/dev/ttyACM0"])
        if current in ports:
            self._port_combo.setCurrentText(current)

    def _set_button_state(self, is_running: bool) -> None:
        script = self.mode_combo.currentData()
        
        if is_running:
            self.launch_btn.setText("🛑 Stop")
            self.launch_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
            self.mode_combo.setEnabled(False)
            
            # Show jog box if we launched moveit (not just gazebo)
            self._jog_box.setVisible("moveit" in script.lower())
        else:
            self.launch_btn.setText("🚀 Launch")
            self.launch_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")
            self.mode_combo.setEnabled(True)
            self._jog_box.setVisible(False)

    def _on_finished(self) -> None:
        self._worker = None
        self._set_button_state(False)

    def _kill_all_nodes(self) -> None:
        if hasattr(self, 'log_rviz'):
            self.log_rviz.append("[LAUNCH] ⚠️ Sending KILL signal to all Gazebo/RViz/MoveIt processes...")
        
        # Send pkill via docker exec if on host, or directly if in container
        cmd = "pkill -9 -f 'rviz2|ign|gazebo|ruby|move_group|parameter_bridge'"
        if os.path.exists("/.dockerenv"):
            full_cmd = ["bash", "-c", cmd]
        else:
            full_cmd = ["docker", "exec", "parol6_dev", "bash", "-c", cmd]
            
        try:
            subprocess.run(full_cmd, check=False)
            if hasattr(self, 'log_rviz'):
                self.log_rviz.append("[LAUNCH] ✅ Kill command executed. Zombie processes terminated.")
        except Exception as e:
            if hasattr(self, 'log_rviz'):
                self.log_rviz.append(f"[LAUNCH] ❌ Error executing kill: {e}")

    def _run_auto_test(self) -> None:
        if self._test_worker:
            self.log_rviz.append("[TEST] Stopping currently running test...")
            self._test_worker.abort()
            self._test_worker = None
            self.test_btn.setText("▶️ Run Auto-Test")
            self.test_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")
            return
            
        script_path = os.path.join(self._launchers_dir, "launch_auto_test.sh")
        if not os.path.exists(script_path):
            self.log_rviz.append(f"[TEST] ❌ Error: Cannot find script {script_path}")
            return
            
        shape = self.test_shape_combo.currentText()
        self.log_rviz.append(f"\n[TEST] Launching comprehensive Auto-Test ({shape})...")
        self.log_rviz.append("[TEST] Spawning moveit_controller and waiting for services...")
        
        self._test_worker = LaunchWorker(script_path, [shape])
        self._test_worker.output_rviz.connect(self.log_rviz.append)
        
        # Hook up resetting the button when the worker naturally finishes
        self._test_worker.finished_ok.connect(self._on_test_finished)
        self._test_worker.finished_err.connect(self._on_test_finished)
        
        self._test_worker.start()
        
        self.test_btn.setText("🛑 Stop Test")
        self.test_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")

    def _on_test_finished(self):
        self._test_worker = None
        self.test_btn.setText("▶️ Run Auto-Test")
        self.test_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")

    # ── ROS2 Joint Jogging ──────────────────────────────────────────────
    def _on_jog_slider_changed(self, idx: int, value: int, label: QLabel) -> None:
        rad = value / 100.0
        label.setText(f"{rad:.2f} rad")

    def _publish_jog_target(self) -> None:
        if not self._worker:
            return
            
        pos = [s.value() / 100.0 for s in self._jog_sliders]
        
        # We use a pure bash wrapper to avoid depending on rclpy in this basic PyQt console
        pub_cmd = [
            "bash", "-lc",
            f"ros2 topic pub --once /forward_position_controller/commands std_msgs/msg/Float64MultiArray \"{{layout: {{dim: [], data_offset: 0}}, data: {pos}}}\""
        ]
        
        from core.process_workers import ProcessWorker
        ProcessWorker(
            cmd=pub_cmd,
            cwd=self._main_window.resolve_path("."),
            env=self._main_window.runtime_env()
        ).start()
        
        self.log_rviz.append(f">> manually publishing jog target: {pos}")

