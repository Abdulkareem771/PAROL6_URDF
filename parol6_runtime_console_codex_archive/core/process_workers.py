"""Background workers for PlatformIO and ROS command execution."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess

from PyQt6.QtCore import QThread, pyqtSignal


class ProcessWorker(QThread):
    output_line = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(int)

    def __init__(self, cmd: list[str], cwd: str, env: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self._cmd = cmd
        self._cwd = cwd
        self._env = env or {}
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        env = os.environ.copy()
        env.update(self._env)
        if "PATH" in env:
            env["PATH"] += os.pathsep + "/usr/local/bin:/usr/bin:/bin"
        else:
            env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

        self.output_line.emit(f"$ {' '.join(self._cmd)}")
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.output_line.emit(line.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as exc:
            self.output_line.emit(f"ERROR: {exc}")
            rc = -1

        if rc == 0:
            self.finished_ok.emit()
        else:
            self.finished_err.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            os.kill(self._proc.pid, signal.SIGINT)
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.terminate()


class PlatformIOWorker(ProcessWorker):
    def __init__(
        self,
        project_dir: str,
        environment: str,
        target: str | None = None,
        extra_args: list[str] | None = None,
        parent=None,
    ):
        pio = shutil.which("pio") or "pio"
        cmd = [pio, "run", "--environment", environment]
        if target:
            cmd += ["--target", target]
        if extra_args:
            cmd += extra_args
        super().__init__(cmd=cmd, cwd=project_dir, parent=parent)
