"""Generic ROS2 command runner tab."""
from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSlider,
)
from PyQt6.QtCore import Qt

from parol6_runtime_console.core.serial_monitor import list_serial_ports
from parol6_runtime_console.core.process_workers import ProcessWorker


class Ros2Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._launch_worker: ProcessWorker | None = None
        self._test_worker: ProcessWorker | None = None
        self._kill_worker: ProcessWorker | None = None
        self._project: dict | None = None
        self._launch_anim_phase = False
        self._test_anim_phase = False
        self._launch_anim = QTimer(self)
        self._launch_anim.setInterval(450)
        self._launch_anim.timeout.connect(self._tick_launch_animation)
        self._test_anim = QTimer(self)
        self._test_anim.setInterval(450)
        self._test_anim.timeout.connect(self._tick_test_animation)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        controls = QGroupBox("Launch Modes")
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(QLabel("Mode:"))
        self.launch_combo = QComboBox()
        self.launch_combo.setMinimumWidth(320)
        controls_layout.addWidget(self.launch_combo)

        self.launch_btn = QPushButton("Launch")
        self.launch_btn.clicked.connect(self._toggle_launch)
        controls_layout.addWidget(self.launch_btn)

        self.kill_btn = QPushButton("Kill All")
        self.kill_btn.clicked.connect(self._kill_all)
        controls_layout.addWidget(self.kill_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._main_window().refresh_project_bindings)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addStretch()
        root.addWidget(controls)

        test_box = QGroupBox("Auto-Test")
        test_layout = QHBoxLayout(test_box)
        test_layout.addWidget(QLabel("Plan:"))
        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(280)
        test_layout.addWidget(self.test_combo)
        self.test_btn = QPushButton("Run Test")
        self.test_btn.clicked.connect(self._toggle_test)
        test_layout.addWidget(self.test_btn)
        test_layout.addStretch()
        root.addWidget(test_box)

        # Added: Manual Joint Jog
        self._jog_box = QGroupBox("Manual Joint Jog (ROS2)")
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

        self._hw_options = QGroupBox("Real Hardware Settings")
        hw_layout = QHBoxLayout(self._hw_options)
        hw_layout.addWidget(QLabel("Serial Port:"))
        self._port_combo = QComboBox()
        self._port_combo.setEditable(True)
        self._port_combo.setMinimumWidth(170)
        hw_layout.addWidget(self._port_combo)
        self._port_refresh_btn = QPushButton("Refresh Ports")
        self._port_refresh_btn.clicked.connect(self._refresh_ports)
        hw_layout.addWidget(self._port_refresh_btn)
        hw_layout.addSpacing(16)
        hw_layout.addWidget(QLabel("Baud:"))
        self._baud_combo = QComboBox()
        self._baud_combo.addItems(["115200", "57600", "250000", "500000", "1000000"])
        self._baud_combo.setCurrentText("115200")
        hw_layout.addWidget(self._baud_combo)
        hw_layout.addSpacing(16)
        note = QLabel("Disconnect the Serial tab before launching real hardware.")
        note.setStyleSheet("color:#fab387; font-size:10px;")
        hw_layout.addWidget(note)
        hw_layout.addStretch()
        root.addWidget(self._hw_options)

        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setStyleSheet(
            "background:#1f2430; border:1px solid #4c566a; border-radius:6px; padding:6px 10px;"
        )
        root.addWidget(self.description)

        logs_box = QGroupBox("ROS2 Logs")
        logs_layout = QHBoxLayout(logs_box)
        self.log_primary = QTextEdit()
        self.log_primary.setReadOnly(True)
        self.log_primary.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        logs_layout.addWidget(self.log_primary, stretch=1)
        self.log_secondary = QTextEdit()
        self.log_secondary.setReadOnly(True)
        self.log_secondary.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        logs_layout.addWidget(self.log_secondary, stretch=1)
        root.addWidget(logs_box, stretch=1)

        self.launch_combo.currentIndexChanged.connect(self._sync_launch_description)
        self.test_combo.currentIndexChanged.connect(self._sync_test_description)
        self._set_launch_button_state(False)
        self._set_test_button_state(False)
        self._refresh_ports()
        self._sync_hw_visibility()
        self._sync_jog_box_visibility(False)

    def load_project(self, project: dict | None) -> None:
        self._project = project
        ros2_cfg = project.get("ros2", {}) if project else {}
        self.launch_combo.clear()
        for action in ros2_cfg.get("launch_actions", []):
            self.launch_combo.addItem(action["name"], action)
        self.test_combo.clear()
        for shape in ros2_cfg.get("auto_test", {}).get("shapes", []):
            self.test_combo.addItem(shape)
        self._sync_description()
        self._refresh_ports()
        self._sync_hw_visibility()

    def _sync_description(self) -> None:
        launch_action = self.current_launch_action()
        parts = []
        if launch_action:
            parts.append(f"<b>Launch:</b> {launch_action.get('description', '')}")
        current_shape = self.current_test_shape()
        if current_shape:
            parts.append(
                "<b>Test:</b> Starts the same auto-test script used by the configurator, "
                f"publishing the selected shape (`{current_shape}`) before triggering execution."
            )
        if not parts:
            self.description.setText("No ROS2 actions defined for this project.")
            return
        self.description.setText("<br><br>".join(parts))

    def _sync_launch_description(self) -> None:
        self._sync_description()
        self._sync_hw_visibility()

    def _sync_test_description(self) -> None:
        self._sync_description()

    def current_launch_action(self) -> dict | None:
        data = self.launch_combo.currentData()
        return data if isinstance(data, dict) else None

    def current_test_shape(self) -> str:
        return self.test_combo.currentText().strip()

    def _toggle_launch(self) -> None:
        if self._launch_worker:
            self.log_primary.append("Stopping ROS2 launch...")
            self._launch_worker.abort()
            return

        action = self.current_launch_action()
        if not action:
            self.log_primary.append("No ROS2 launch mode selected.")
            return

        self.log_primary.clear()
        self.log_secondary.clear()
        self._launch_worker = self._start_worker(action, self._append_launch_line, self._on_launch_done, self._on_launch_error)
        self._set_launch_button_state(True)

    def _toggle_test(self) -> None:
        if self._test_worker:
            self.log_primary.append("Stopping auto-test...")
            self._test_worker.abort()
            return

        shape = self.current_test_shape()
        if not shape:
            self.log_primary.append("No auto-test selected.")
            return

        script_rel = self._project.get("ros2", {}).get("auto_test", {}).get("script") if self._project else None
        if not script_rel:
            self.log_primary.append("No auto-test script configured for this project.")
            return
        script_path = self._main_window().resolve_path(script_rel)
        self.log_primary.append(f"[TEST] Launching comprehensive Auto-Test ({shape})...")
        self.log_primary.append("[TEST] Spawning moveit_controller and waiting for services...")
        action = {
            "cwd": ".",
            "command": ["bash", script_path, shape],
        }
        self._test_worker = self._start_worker(action, self.log_primary.append, self._on_test_done, self._on_test_error)
        self._set_test_button_state(True)

    def _start_worker(self, action: dict, line_cb, ok_cb, err_cb) -> ProcessWorker:
        cwd = self._main_window().resolve_path(action.get("cwd", "."))
        env = self._main_window().runtime_env()
        extra_env = action.get("env", {})
        env.update(extra_env)
        if action.get("requires_serial"):
            port = self._port_combo.currentText().strip()
            baud = self._baud_combo.currentText().strip()
            if port:
                env["PAROL6_SERIAL_PORT"] = port
            if baud:
                env["PAROL6_BAUD_RATE"] = baud
        worker = ProcessWorker(cmd=action["command"], cwd=cwd, env=env)
        worker.output_line.connect(line_cb)
        worker.finished_ok.connect(ok_cb)
        worker.finished_err.connect(err_cb)
        worker.start()
        return worker

    def _append_launch_line(self, line: str) -> None:
        lower = line.lower()
        if "ign" in lower or "gazebo" in lower or "spawn" in lower or "ruby" in lower:
            self.log_secondary.append(line)
        else:
            self.log_primary.append(line)

    def _kill_all(self) -> None:
        if not self._project:
            self.log_primary.append("No project selected.")
            return
        kill_cmd = self._project.get("ros2", {}).get("kill_command")
        if not kill_cmd:
            self.log_primary.append("No kill command configured for this project.")
            return
        self._kill_worker = ProcessWorker(
            cmd=kill_cmd,
            cwd=self._main_window().resolve_path("."),
            env=self._main_window().runtime_env(),
        )
        self._kill_worker.output_line.connect(self.log_primary.append)
        self._kill_worker.finished_ok.connect(self._on_kill_done)
        self._kill_worker.finished_err.connect(self._on_kill_error)
        self._kill_worker.start()

    def _on_launch_error(self, code: int) -> None:
        self.log_primary.append(f"ROS2 launch stopped with exit code {code}.")
        self._launch_worker = None
        self._set_launch_button_state(False)
        self._sync_jog_box_visibility(False)

    def _on_launch_done(self) -> None:
        self.log_primary.append("ROS2 launch finished.")
        self._launch_worker = None
        self._set_launch_button_state(False)
        self._sync_jog_box_visibility(False)

    def _on_test_done(self) -> None:
        self.log_primary.append("Auto-test finished.")
        self._test_worker = None
        self._set_test_button_state(False)

    def _on_test_error(self, code: int) -> None:
        self.log_primary.append(f"Auto-test stopped with exit code {code}.")
        self._test_worker = None
        self._set_test_button_state(False)

    def _on_kill_done(self) -> None:
        self.log_primary.append("Kill command completed.")
        self._kill_worker = None

    def _on_kill_error(self, code: int) -> None:
        self.log_primary.append(f"Kill command exited with {code}.")
        self._kill_worker = None

    def _main_window(self):
        return self.window()

    def _refresh_ports(self) -> None:
        ports = list_serial_ports()
        current = self._port_combo.currentText().strip() or self._main_window().runtime_env().get("PAROL6_SERIAL_PORT", "")
        self._port_combo.clear()
        self._port_combo.addItems(ports if ports else ["/dev/ttyACM0"])
        if current:
            self._port_combo.setCurrentText(current)
        baud = self._main_window().runtime_env().get("PAROL6_SERIAL_BAUD", "")
        if baud:
            self._baud_combo.setCurrentText(baud)

    def _sync_hw_visibility(self) -> None:
        action = self.current_launch_action()
        self._hw_options.setVisible(bool(action and action.get("requires_serial")))

    def _set_launch_button_state(self, running: bool) -> None:
        self.launch_combo.setEnabled(not running)
        self._sync_jog_box_visibility(running)
        if running:
            self.launch_btn.setText("Stop")
            self._launch_anim_phase = False
            self._launch_anim.start()
            self._tick_launch_animation()
        else:
            self._launch_anim.stop()
            self.launch_btn.setText("Launch")
            self.launch_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")

    def _set_test_button_state(self, running: bool) -> None:
        self.test_combo.setEnabled(not running)
        if running:
            self.test_btn.setText("Stop Test")
            self._test_anim_phase = False
            self._test_anim.start()
            self._tick_test_animation()
        else:
            self._test_anim.stop()
            self.test_btn.setText("Run Test")
            self.test_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")

    def _tick_launch_animation(self) -> None:
        self._launch_anim_phase = not self._launch_anim_phase
        bg = "#f38ba8" if self._launch_anim_phase else "#fab387"
        self.launch_btn.setStyleSheet(f"background:{bg}; color:#1e1e2e; font-weight:bold;")

    def _tick_test_animation(self) -> None:
        self._test_anim_phase = not self._test_anim_phase
        bg = "#f9e2af" if self._test_anim_phase else "#fab387"
        self.test_btn.setStyleSheet(f"background:{bg}; color:#1e1e2e; font-weight:bold;")

    def _sync_jog_box_visibility(self, is_running: bool) -> None:
        action = self.current_launch_action()
        is_real_or_fake = action and "MoveIt" in action.get("name", "")
        self._jog_box.setVisible(is_running and is_real_or_fake)

    def _on_jog_slider_changed(self, idx: int, value: int, label: QLabel) -> None:
        rad = value / 100.0
        label.setText(f"{rad:.2f} rad")

    def _publish_jog_target(self) -> None:
        if not self._launch_worker:
            return
            
        pos = [s.value() / 100.0 for s in self._jog_sliders]
        
        # We use a pure bash wrapper to avoid depending on rclpy in this basic PyQt console
        # The topic type is likely std_msgs/Float64MultiArray or sensor_msgs/JointState.
        # Assuming the simplest method for MoveIt is joint_angle_targets via custom service or
        # using standard CLI pub to a forward_position_controller.
        pub_cmd = [
            "bash", "-lc",
            f"ros2 topic pub --once /forward_position_controller/commands std_msgs/msg/Float64MultiArray \"{{layout: {{dim: [], data_offset: 0}}, data: {pos}}}\""
        ]
        
        ProcessWorker(
            cmd=pub_cmd,
            cwd=self._main_window().resolve_path("."),
            env=self._main_window().runtime_env()
        ).start()
        
        self.log_primary.append(f">> manually publishing jog target: {pos}")
