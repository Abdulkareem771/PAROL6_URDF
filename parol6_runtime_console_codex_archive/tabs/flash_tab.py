"""Generic PlatformIO flash tab."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from parol6_runtime_console.core.process_workers import PlatformIOWorker


class FlashTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PlatformIOWorker | None = None
        self._project: dict | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Build and flash the selected PlatformIO project/environment.")
        title.setWordWrap(True)
        root.addWidget(title)

        workflow = QLabel(
            "Workflow: pick a project, confirm the PlatformIO directory and environment, then use Build or Flash. "
            "This tab does not generate code; it only compiles and uploads existing projects."
        )
        workflow.setWordWrap(True)
        workflow.setStyleSheet(
            "background:#1a2a1a; border:1px solid #a6e3a1; border-radius:6px; padding:6px 10px;"
        )
        root.addWidget(workflow)

        path_box = QGroupBox("PlatformIO Project")
        path_row = QHBoxLayout(path_box)
        path_row.addWidget(QLabel("Project dir:"))
        self.project_dir_edit = QLineEdit()
        path_row.addWidget(self.project_dir_edit, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_project_dir)
        path_row.addWidget(browse_btn)
        reload_envs_btn = QPushButton("Reload Envs")
        reload_envs_btn.clicked.connect(self._reload_envs)
        path_row.addWidget(reload_envs_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_project_dir)
        path_row.addWidget(reset_btn)
        root.addWidget(path_box)

        stm_box = QGroupBox("STM32 Flash Notes")
        stm_layout = QVBoxLayout(stm_box)
        stm_hint = QLabel(
            "Best path for STM32: choose the env from the project's platformio.ini. "
            "For ST-Link uploads, no serial port is required. For USB DFU, put the board in bootloader mode first. "
            "Keep the toolbar Port field for serial monitor only."
        )
        stm_hint.setWordWrap(True)
        stm_layout.addWidget(stm_hint)
        root.addWidget(stm_box)

        button_row = QHBoxLayout()
        self.build_btn = QPushButton("Build")
        self.build_btn.clicked.connect(self._build_only)
        button_row.addWidget(self.build_btn)

        self.flash_btn = QPushButton("Flash")
        self.flash_btn.clicked.connect(self._flash)
        button_row.addWidget(self.flash_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)

        clear_btn = QPushButton("Clear Log")
        button_row.addWidget(clear_btn)
        button_row.addStretch()
        root.addLayout(button_row)

        log_group = QGroupBox("PlatformIO Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log)
        root.addWidget(log_group, stretch=1)
        clear_btn.clicked.connect(self.log.clear)

    def load_project(self, project: dict | None) -> None:
        self._project = project
        if not project:
            self.project_dir_edit.clear()
            return
        flash_cfg = project.get("flash", {})
        self.project_dir_edit.setText(self._main_window().resolve_path(flash_cfg.get("project_dir", "")))

    def start_action(self, upload: bool) -> None:
        project = self._main_window().current_project()
        if not project:
            self._append("No project selected.")
            return
        env_name = self._main_window().current_environment()
        if not env_name:
            self._append("Selected project has no PlatformIO environment.")
            return

        project_dir = self.project_dir_edit.text().strip()
        if not project_dir:
            self._append("No PlatformIO project directory set.")
            return
        target = "upload" if upload else None

        self._worker = PlatformIOWorker(project_dir=project_dir, environment=env_name, target=target)
        self._worker.output_line.connect(self._append)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.finished_err.connect(self._on_error)
        self._set_running(True)
        self._worker.start()

    def _build_only(self) -> None:
        self.start_action(upload=False)

    def _flash(self) -> None:
        self.start_action(upload=True)

    def _stop(self) -> None:
        if self._worker:
            self._append("Stopping PlatformIO process...")
            self._worker.abort()

    def _set_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.flash_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _on_done(self) -> None:
        self._append("Completed successfully.")
        self._worker = None
        self._set_running(False)

    def _on_error(self, code: int) -> None:
        self._append(f"Failed with exit code {code}.")
        self._worker = None
        self._set_running(False)

    def _append(self, line: str) -> None:
        self.log.append(line)

    def _browse_project_dir(self) -> None:
        project_dir = QFileDialog.getExistingDirectory(self, "Select PlatformIO project directory")
        if project_dir:
            self.project_dir_edit.setText(project_dir)
            self._reload_envs()

    def _reset_project_dir(self) -> None:
        if self._project:
            flash_cfg = self._project.get("flash", {})
            self.project_dir_edit.setText(self._main_window().resolve_path(flash_cfg.get("project_dir", "")))
            self._reload_envs()

    def _reload_envs(self) -> None:
        self._main_window()._reload_envs_from_current_project(self.project_dir_edit.text().strip())

    def _main_window(self):
        return self.window()
