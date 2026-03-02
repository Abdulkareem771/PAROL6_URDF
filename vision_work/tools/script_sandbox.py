"""
Script Sandbox — PySide6 Tool
==============================
Load any Python image processing script and run it on any image.
See stdout / stderr in a live console. Zero changes to the script needed.

The script is expected to either:
  - Accept a filepath as argv[1] and print results to stdout, OR
  - Define a 'process_image(path)' or 'segment_blocks(path)' function
    that returns an annotated image (numpy array) which will be displayed.
"""
import sys
import os
import cv2
import numpy as np
import subprocess
import threading
import importlib.util
import time

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QLineEdit, QFileDialog, QTextEdit, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QColor

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C, STYLE_SHEET


class ScriptRunner(QObject):
    """Runs the script in a background thread and emits console output."""
    log = Signal(str)
    image_ready = Signal(np.ndarray)
    done = Signal(float)

    def __init__(self, script_path, image_path):
        super().__init__()
        self.script_path = script_path
        self.image_path = image_path

    def run(self):
        start = time.perf_counter()

        # --- Strategy 1: Try importing and calling process_image / segment_blocks ---
        try:
            spec = importlib.util.spec_from_file_location("sandbox_module", self.script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            result_img = None
            if hasattr(mod, 'process_image'):
                self.log.emit("[Sandbox] Calling process_image(path)...\n")
                result_img = mod.process_image(self.image_path)
            elif hasattr(mod, 'segment_blocks'):
                self.log.emit("[Sandbox] Calling segment_blocks(path)...\n")
                res = mod.segment_blocks(self.image_path)
                # Teammate's script returns: G, R, img_annotated, bbox_G, bbox_R, bbox_I
                if isinstance(res, tuple) and len(res) >= 3:
                    result_img = res[2]

            if result_img is not None:
                if isinstance(result_img, np.ndarray):
                    if result_img.ndim == 2:   # Grayscale
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
                    elif result_img.shape[2] == 3:
                        # Check if it looks like BGR (common from OpenCV)
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    self.image_ready.emit(result_img)
                    self.log.emit("[Sandbox] Image result rendered on canvas.\n")
            else:
                self.log.emit("[Sandbox] No image returned. Script ran OK.\n")

        except Exception as e:
            self.log.emit(f"[Import Error] {e}\n")
            # --- Strategy 2: Fallback — run as subprocess and capture stdout ---
            self.log.emit("[Sandbox] Falling back to subprocess execution...\n")
            try:
                proc = subprocess.run(
                    [sys.executable, self.script_path, self.image_path],
                    capture_output=True, text=True, timeout=30
                )
                if proc.stdout:
                    self.log.emit(proc.stdout)
                if proc.stderr:
                    self.log.emit(f"[stderr]\n{proc.stderr}")
            except subprocess.TimeoutExpired:
                self.log.emit("[Error] Script timed out after 30s.\n")
            except Exception as e2:
                self.log.emit(f"[Subprocess Error] {e2}\n")

        elapsed = (time.perf_counter() - start) * 1000
        self.done.emit(elapsed)


class ScriptSandbox(BaseVisionApp):
    """
    Script Sandbox: Load any Python image processing script.
    Run it on any image and see live console output + rendered result.
    """
    def __init__(self):
        super().__init__(title="Script Sandbox — Image Processing Tester", width=1450, height=820)
        self._script_path = None
        self._image_path = None
        self._runner_thread = None
        self._setup_ui()

    def _setup_ui(self):
        # --- Sidebar ---
        self._add_section_header("📁 Input Image")
        self._build_default_image_loader()
        self.image_loaded.connect(self._on_image_loaded)

        self._add_section_header("🐍 Script")
        self.script_entry = QLineEdit()
        self.script_entry.setReadOnly(True)
        self.script_entry.setPlaceholderText("No script loaded...")
        self.sidebar_layout.addWidget(self.script_entry)
        self._add_button("Load Python Script (.py)", self._load_script)

        self._add_section_header("Controls")
        self._add_button("▶  Run Script", self._run_script, primary=True)
        self._add_button("🗑  Clear Console", self._clear_console)

        self._add_section_header("Info")
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.sidebar_layout.addWidget(self.lbl_status)
        self.sidebar_layout.addStretch()

        # --- Console Panel below the canvas ---
        # Re-structure the central widget to add a console at the bottom
        self._build_console_panel()

    def _build_console_panel(self):
        """Injects a live console pane below the main QGraphicsView."""
        # The splitter currently has [sidebar | view].
        # We replace the right side with a vertical splitter [view | console].
        self.splitter.widget(1).setParent(None)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.view)

        console_frame = QFrame()
        console_frame.setStyleSheet(f"background-color: {C['panel']}; border-top: 1px solid {C['border']};")
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(8, 6, 8, 6)
        console_layout.setSpacing(4)

        lbl = QLabel("CONSOLE OUTPUT")
        lbl.setProperty("class", "header")
        lbl.setStyleSheet(f"color: {C['text2']}; font-size: 10px; font-weight: bold;")
        console_layout.addWidget(lbl)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Monospace", 10))
        self.console.setStyleSheet(
            f"background-color: #11111b; color: {C['green']}; border: none;"
            f"selection-background-color: {C['accent']};"
        )
        self.console.setPlaceholderText("Run a script to see output here...")
        console_layout.addWidget(self.console)

        right_splitter.addWidget(console_frame)
        right_splitter.setSizes([550, 220])

        self.splitter.addWidget(right_splitter)
        self.splitter.setSizes([350, 1100])

    def _on_image_loaded(self, rgb: np.ndarray):
        self._image_path = getattr(self, '_last_loaded_path', None)
        self.lbl_status.setText(f"Image: {self._image_path or 'Clipboard'}")

    def _handle_browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if p:
            self._last_loaded_path = p
            self._load_from_path(p)

    def _load_script(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Python Script", "", "Python (*.py)")
        if p:
            self._script_path = p
            self.script_entry.setText(os.path.basename(p))
            self._console_log(f"[Sandbox] Script loaded: {p}\n")

    def _run_script(self):
        if not self._script_path:
            self._console_log("[Error] No script loaded. Use 'Load Python Script'.\n")
            return
        if self._rgb is None:
            self._console_log("[Error] No image loaded. Load or paste an image first.\n")
            return

        # Write current canvas image to a temp file for the script
        tmp = "/tmp/sandbox_input.jpg"
        bgr = cv2.cvtColor(self._rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp, bgr)
        self._last_loaded_path = tmp

        self._console_log(f"[Sandbox] Running {os.path.basename(self._script_path)}...\n")
        self.lbl_status.setText("Running...")

        runner = ScriptRunner(self._script_path, tmp)
        runner.log.connect(self._console_log)
        runner.image_ready.connect(self._display_rgb)
        runner.done.connect(self._on_run_done)

        self._runner_thread = threading.Thread(target=runner.run, daemon=True)
        self._runner_thread.start()

    def _on_run_done(self, elapsed_ms: float):
        self._console_log(f"[Sandbox] ✓ Finished in {elapsed_ms:.1f} ms\n")
        self.lbl_status.setText(f"Done — {elapsed_ms:.1f} ms")

    def _console_log(self, text: str):
        self.console.moveCursor(self.console.textCursor().End)
        self.console.insertPlainText(text)
        self.console.ensureCursorVisible()

    def _clear_console(self):
        self.console.clear()


if __name__ == "__main__":
    run_app(ScriptSandbox)
