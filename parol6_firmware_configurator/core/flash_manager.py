"""
flash_manager.py — Runs `pio run --upload` and streams output via Qt signals.
"""
from __future__ import annotations
import subprocess
import shutil
from PyQt6.QtCore import QThread, pyqtSignal


class FlashWorker(QThread):
    """Runs PlatformIO in a background thread. Emits stdout/stderr line-by-line."""
    output_line  = pyqtSignal(str)   # plain text line
    finished_ok  = pyqtSignal()
    finished_err = pyqtSignal(int)   # return code

    def __init__(self, project_dir: str, environment: str = "teensy41",
                 extra_args: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._project_dir = project_dir
        self._environment = environment
        self._extra_args  = extra_args or []
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        pio = shutil.which("pio") or "pio"
        cmd = [pio, "run", "--environment", self._environment, "--target", "upload"]
        cmd += self._extra_args

        self.output_line.emit(f"[FLASH] $ {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=self._project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._proc.stdout:  # type: ignore[union-attr]
                self.output_line.emit(line.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
        except FileNotFoundError:
            self.output_line.emit("[FLASH] ERROR: `pio` not found. Install PlatformIO.")
            rc = -1

        if rc == 0:
            self.output_line.emit("[FLASH] ✅ Upload successful.")
            self.finished_ok.emit()
        else:
            self.output_line.emit(f"[FLASH] ❌ Upload failed (exit code {rc}).")
            self.finished_err.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


class BuildWorker(QThread):
    """Runs `pio run` (build only, no upload) to check for compile errors."""
    output_line  = pyqtSignal(str)
    finished_ok  = pyqtSignal()
    finished_err = pyqtSignal(int)

    def __init__(self, project_dir: str, environment: str = "teensy41", parent=None):
        super().__init__(parent)
        self._project_dir = project_dir
        self._environment = environment
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        pio = shutil.which("pio") or "pio"
        cmd = [pio, "run", "--environment", self._environment]
        self.output_line.emit(f"[BUILD] $ {' '.join(cmd)}")
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=self._project_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in self._proc.stdout:  # type: ignore[union-attr]
                self.output_line.emit(line.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
        except FileNotFoundError:
            self.output_line.emit("[BUILD] ERROR: `pio` not found.")
            rc = -1

        if rc == 0:
            self.output_line.emit("[BUILD] ✅ Build successful.")
            self.finished_ok.emit()
        else:
            self.output_line.emit(f"[BUILD] ❌ Build failed (exit code {rc}).")
            self.finished_err.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
