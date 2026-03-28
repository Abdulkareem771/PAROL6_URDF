#!/usr/bin/env python3
"""
Vision Pipeline GUI — PAROL6 Vision Launcher
==============================================
Unified PySide6 launcher for the PAROL6 vision pipeline.

Pipeline Flow:
  Stage 1: Camera (Live Kinect) + Capture Image node
  Stage 2: Processing mode — choose ONE of:
             • YOLO Segment
             • Color Mode
             • Manual Red Line (embedded canvas like manual_annotator)
  Stage 3: Path Optimizer → Depth Matcher → Path Generator
  Stage 4: Send to MoveIt (ros2 run / ros2 service call)

  "Legacy Tools" section mirrors vision_work/launcher.py buttons.

The node management pattern is borrowed from the firmware configurator
launch_tab.py: QThread-based NodeWorker wraps subprocess.Popen, streams
stdout/stderr to the log pane, and supports SIGINT abort.
"""

from __future__ import annotations
import faulthandler
faulthandler.enable()

import os
import sys
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QLabel, QPushButton, QRadioButton,
    QButtonGroup, QTextEdit, QScrollArea, QFrame, QTabWidget,
    QSizePolicy, QGridLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSpinBox, QComboBox, QSlider, QDialog, QDialogButtonBox,
    QFileDialog, QMessageBox, QCheckBox, QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QFont, QColor, QPalette, QPixmap, QImage, QIcon, QPainter,
    QPen, QCursor,
)

# ─── Try importing ROS / cv_bridge for live topic previews ───────────────────
ROS2_OK = False
ROS2_IMPORT_ERROR = ""
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.parameter import Parameter as RclpyParameter
    from sensor_msgs.msg import Image as ROSImage
    from std_srvs.srv import Trigger
    from rcl_interfaces.srv import SetParameters
    from cv_bridge import CvBridge
    ROS2_OK = True
except Exception as exc:
    ROS2_IMPORT_ERROR = str(exc)

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
VISION_WORK   = WORKSPACE_DIR / "vision_work"
VISION_PKG    = "parol6_vision"
FALLBACK_INSTALL_DIR = Path("/tmp/parol6_install")
ROS_LOG_DIR = Path("/tmp/parol6_ros_logs")


def _wrap_ros_command(cmd: list[str]) -> list[str]:
    """
    Ensure every ROS 2 subprocess runs in a shell with the required setup files
    sourced, even when this GUI itself was launched from plain Python.
    """
    if not cmd or cmd[0] != "ros2":
        return cmd

    quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
    workspace_setup = shlex.quote(str(WORKSPACE_DIR / "install" / "setup.bash"))
    fallback_setup = shlex.quote(str(FALLBACK_INSTALL_DIR / "setup.bash"))
    setup_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "if [ -f /opt/kinect_ws/install/setup.bash ]; then "
        "source /opt/kinect_ws/install/setup.bash; "
        "fi && "
        f"if [ -f {workspace_setup} ]; then "
        f"source {workspace_setup}; "
        "fi && "
        f"if [ -f {fallback_setup} ]; then "
        f"source {fallback_setup}; "
        "fi && "
        f"cd {shlex.quote(str(WORKSPACE_DIR))} && "
        f"{quoted_cmd}"
    )
    return ["bash", "-lc", setup_cmd]


def _ros_node_check_cmd(node_name: str) -> list[str]:
    quoted_name = shlex.quote(node_name)
    workspace_setup = shlex.quote(str(WORKSPACE_DIR / "install" / "setup.bash"))
    fallback_setup = shlex.quote(str(FALLBACK_INSTALL_DIR / "setup.bash"))
    check_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "if [ -f /opt/kinect_ws/install/setup.bash ]; then source /opt/kinect_ws/install/setup.bash; fi && "
        f"if [ -f {workspace_setup} ]; then source {workspace_setup}; fi && "
        f"if [ -f {fallback_setup} ]; then source {fallback_setup}; fi && "
        f"ros2 node list | grep -qx {quoted_name}"
    )
    return ["bash", "-lc", check_cmd]

# ─── Color Palette (matches firmware configurator dark theme) ─────────────────
C = {
    "bg":      "#1e1e2e",
    "panel":   "#181825",
    "border":  "#45475a",
    "accent":  "#cba6f7",
    "green":   "#a6e3a1",
    "red":     "#f38ba8",
    "yellow":  "#f9e2af",
    "blue":    "#89b4fa",
    "text":    "#cdd6f4",
    "text2":   "#a6adc8",
    "running": "#a6e3a1",
    "stopped": "#f38ba8",
}

STYLE = f"""
QMainWindow, QWidget, QDialog {{
    background: {C['bg']};
    color: {C['text']};
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 6px;
    font-weight: bold;
    color: {C['text2']};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
QPushButton {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 5px 14px;
    color: {C['text']};
}}
QPushButton:hover   {{ background: #313244; border-color: {C['accent']}; }}
QPushButton:pressed {{ background: #585b70; }}
QPushButton:disabled{{ color: #585b70; border-color: #313244; }}
QTextEdit, QPlainTextEdit {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    color: {C['text']};
    font-family: 'Monospace', 'Courier New';
    font-size: 11px;
}}
QLineEdit, QSpinBox, QComboBox {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    color: {C['text']};
    padding: 3px 6px;
}}
QTabBar::tab {{
    background: {C['panel']};
    color: {C['text2']};
    padding: 6px 16px;
    border: 1px solid {C['border']};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: #313244; color: {C['accent']}; font-weight: bold; }}
QTabBar::tab:hover    {{ background: #313244; color: {C['text']}; }}
QScrollBar:vertical   {{ background:{C['panel']}; width:8px; border-radius:4px; }}
QScrollBar::handle:vertical {{ background:{C['border']}; border-radius:4px; min-height:24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ background:{C['panel']}; height:8px; border-radius:4px; }}
QScrollBar::handle:horizontal {{ background:{C['border']}; border-radius:4px; }}
QRadioButton {{ color: {C['text']}; spacing: 6px; }}
QRadioButton::indicator {{ width:14px; height:14px; border:1px solid {C['border']}; border-radius:7px; background:{C['panel']}; }}
QRadioButton::indicator:checked {{ background:{C['accent']}; border-color:{C['accent']}; }}
QCheckBox {{ color: {C['text']}; }}
QCheckBox::indicator {{ width:14px; height:14px; border:1px solid {C['border']}; border-radius:3px; background:{C['panel']}; }}
QCheckBox::indicator:checked {{ background:{C['accent']}; border-color:{C['accent']}; }}
QLabel {{ color: {C['text']}; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# NodeWorker — background QThread that runs and streams a subprocess
# ─────────────────────────────────────────────────────────────────────────────

class NodeWorker(QThread):
    """
    Runs a shell command in a background thread, emits each output line,
    and supports graceful SIGINT abort (matching firmware configurator pattern).
    """
    line_out = Signal(str)
    finished = Signal(int)   # exit code

    def __init__(self, cmd: list[str], env: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._display_cmd = cmd
        self._cmd  = _wrap_ros_command(cmd)
        self._env  = env or os.environ.copy()
        self._env.setdefault("ROS_LOG_DIR", str(ROS_LOG_DIR))
        self._proc: Optional[subprocess.Popen] = None

    def run(self) -> None:
        self.line_out.emit(f"[RUN] $ {' '.join(self._display_cmd)}")
        try:
            ROS_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self._env,
                cwd=str(WORKSPACE_DIR),
            )
            for line in self._proc.stdout:
                self.line_out.emit(line.rstrip())
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as exc:
            self.line_out.emit(f"[ERROR] {exc}")
            rc = -1

        self.line_out.emit(
            "[DONE] Process exited." if rc == 0
            else f"[STOPPED] exit code {rc}"
        )
        self.finished.emit(rc)

    def abort(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.kill(self._proc.pid, signal.SIGINT)
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# ROS topic image preview widget (optional — only active when ROS2_OK & CV2_OK)
# ─────────────────────────────────────────────────────────────────────────────

class TopicPreviewLabel(QLabel):
    """
    A QLabel that subscribes to one ROS Image topic and refreshes itself
    at ~10 Hz via a QTimer.  Falls back to a placeholder when ROS is off.
    """

    def __init__(self, topic: str, ros_node, bridge, parent=None):
        super().__init__(parent)
        self._topic = topic
        self._ros_node = ros_node
        self._bridge = bridge
        self._latest_img: Optional[bytes] = None   # raw RGB numpy bytes
        self._subscription = None

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['border']};"
            " border-radius:4px; color:#585b70;"
        )
        self.setText(f"⌛  Waiting for\n{topic}")
        self.setMinimumSize(200, 150)

        if ROS2_OK and ros_node:
            self._subscription = ros_node.create_subscription(
                ROSImage, topic, self._ros_cb, qos_profile_sensor_data
            )
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh)
            self._timer.start(100)   # 10 Hz

    def _ros_cb(self, msg: ROSImage) -> None:
        if not (CV2_OK and self._bridge):
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self._latest_img = frame
        except Exception:
            pass

    def _refresh(self) -> None:
        if self._latest_img is None:
            return
        frame = self._latest_img
        h, w, ch = frame.shape
        lw = self.width()
        lh = self.height()
        if lw < 10 or lh < 10:
            return
        scale = min(lw / w, lh / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        frame_small = cv2.resize(frame, (nw, nh))
        self._img_gc_cache = frame_small  # KEEP ALIVE for PySide reference count
        qimg = QImage(frame_small.data, nw, nh, int(frame_small.strides[0]), QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))
        self.setText("")


# ─────────────────────────────────────────────────────────────────────────────
# Embedded Manual Red-Line Canvas (adapted from manual_annotator.py)
# ─────────────────────────────────────────────────────────────────────────────

class ManualCanvas(QGraphicsView):
    """
    An embedded QGraphicsView where the user can draw red lines over
    a loaded image (same logic as vision_work/tools/manual_annotator.py).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['border']};"
        )
        self._drawing = False
        self._last_pt: Optional[QPointF] = None
        self._mask_item = None
        self._base_img: Optional[np.ndarray] = None   # RGB numpy
        self._mask_arr: Optional[np.ndarray] = None   # RGBA numpy
        self._brush = 5
        # Stroke tracking for serialisation
        self._all_strokes: list[list[list[int]]] = []   # [ [ [x,y], ... ], ... ]
        self._current_stroke: list[list[int]] = []
        # Straight-line mode
        self._straight_line_mode = False
        self._sl_waypoints: list[list[int]] = []   # waypoints for current straight-line stroke

        self._bg_item = self._scene.addPixmap(QPixmap())
        self._bg_item.setZValue(0)
        self._mask_item = self._scene.addPixmap(QPixmap())
        self._mask_item.setZValue(1)

        self._placeholder = QLabel("Load an image above to start drawing red lines.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(f"color:{C['text2']}; font-size:13px;")
        self._placeholder_proxy = self._scene.addWidget(self._placeholder)

    # -- public API -----------------------------------------------------------
    def set_brush(self, px: int) -> None:
        self._brush = max(1, px)
        self._update_cursor()

    def set_straight_line_mode(self, enabled: bool) -> None:
        """Toggle straight-line waypoint mode. Right-click closes the current line."""
        self._straight_line_mode = enabled
        if not enabled:
            self._finish_sl_stroke()

    def get_strokes(self) -> list[list[list[int]]]:
        """Return all recorded strokes as a list of point lists for serialisation."""
        return self._all_strokes.copy()

    def load_image(self, rgb: np.ndarray) -> None:
        if self._placeholder_proxy.isVisible():
            self._placeholder_proxy.setVisible(False)
            
        self._base_img = rgb  # KEEP ALIVE
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888)
        
        self._bg_item.setPixmap(QPixmap.fromImage(qimg))
        self._mask_arr = np.zeros((h, w, 4), dtype=np.uint8)
        self._refresh_mask()
        self.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._update_cursor()

    def clear_mask(self) -> None:
        if self._mask_arr is not None:
            self._mask_arr.fill(0)
            self._refresh_mask()
        self._all_strokes = []
        self._current_stroke = []
        self._sl_waypoints = []

    def _refresh_mask(self) -> None:
        if self._mask_arr is None or self._mask_item is None:
            return
        h, w = self._mask_arr.shape[:2]
        # _mask_arr is kept alive by self._mask_arr
        qimg = QImage(self._mask_arr.data, w, h, int(self._mask_arr.strides[0]), QImage.Format_RGBA8888)
        self._mask_item.setPixmap(QPixmap.fromImage(qimg))

    def get_annotated_bgr(self) -> Optional[np.ndarray]:
        """Return a BGR copy of the image with red strokes blended in."""
        if self._base_img is None or self._mask_arr is None:
            return None
        out = self._base_img.copy()
        alpha = self._mask_arr[:, :, 3] > 0
        out[alpha] = [255, 0, 0]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # -- drawing internals ----------------------------------------------------
    def _update_cursor(self) -> None:
        sz = max(self._brush + 4, 16)
        pix = QPixmap(sz, sz)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setPen(QPen(QColor(255, 0, 0, 200), 1))
        c = sz / 2
        p.drawEllipse(QPointF(c, c), self._brush / 2, self._brush / 2)
        p.end()
        self.viewport().setCursor(QCursor(pix, int(c), int(c)))

    def _draw(self, a: QPointF, b: QPointF) -> None:
        if self._mask_arr is None:
            return
        x1, y1 = int(a.x()), int(a.y())
        x2, y2 = int(b.x()), int(b.y())
        cv2.line(self._mask_arr, (x1, y1), (x2, y2), (255, 0, 0, 255),
                 self._brush, lineType=cv2.LINE_AA)
        self._refresh_mask()
        # Track point in current stroke
        self._current_stroke.append([x2, y2])

    def _finish_freehand_stroke(self) -> None:
        if self._current_stroke:
            self._all_strokes.append(self._current_stroke)
            self._current_stroke = []

    def _finish_sl_stroke(self) -> None:
        """Commit and clear the current straight-line waypoint stroke."""
        if len(self._sl_waypoints) >= 2:
            self._all_strokes.append(self._sl_waypoints.copy())
        self._sl_waypoints = []

    def wheelEvent(self, event) -> None:
        """Add robust Canvas zooming functionality with the scroll wheel via modifiers."""
        if event.modifiers() & Qt.ControlModifier:
            zoom_in_factor = 1.15
            zoom_out_factor = 1.0 / zoom_in_factor
            
            if event.angleDelta().y() > 0:
                zoom_factor = zoom_in_factor
            else:
                zoom_factor = zoom_out_factor
                
            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, ev) -> None:
        if self._mask_arr is None:
            super().mousePressEvent(ev)
            return
        sp = self.mapToScene(ev.pos())
        h, w = self._mask_arr.shape[:2]
        in_bounds = 0 <= sp.x() < w and 0 <= sp.y() < h

        if self._straight_line_mode:
            if ev.button() == Qt.LeftButton and in_bounds:
                ix, iy = int(sp.x()), int(sp.y())
                # Snap to H/V if Shift held
                if ev.modifiers() & Qt.ShiftModifier and self._sl_waypoints:
                    px, py = self._sl_waypoints[-1]
                    if abs(ix - px) > abs(iy - py):
                        iy = py   # snap horizontal
                    else:
                        ix = px   # snap vertical
                if self._sl_waypoints:
                    prev = self._sl_waypoints[-1]
                    cv2.line(self._mask_arr,
                             (prev[0], prev[1]), (ix, iy),
                             (255, 0, 0, 255), self._brush, lineType=cv2.LINE_AA)
                    self._refresh_mask()
                self._sl_waypoints.append([ix, iy])
                ev.accept()
                return
            elif ev.button() == Qt.RightButton:
                self._finish_sl_stroke()
                ev.accept()
                return
        else:
            if ev.button() == Qt.LeftButton and in_bounds:
                self._drawing = True
                self._last_pt = sp
                self._draw(sp, sp)
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._drawing and self._last_pt is not None:
            sp = self.mapToScene(ev.pos())
            self._draw(self._last_pt, sp)
            self._last_pt = sp
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self._last_pt = None
            self._finish_freehand_stroke()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# ─────────────────────────────────────────────────────────────────────────────
# NodeButton — a toggle button that starts/stops a NodeWorker
# ─────────────────────────────────────────────────────────────────────────────

class NodeButton(QWidget):
    """
    A label + toggle button that shows ● (green) / ● (red) node status,
    start/stop the underlying NodeWorker, and pipes output to a QTextEdit.
    """
    def __init__(
        self,
        label: str,
        cmd_fn,          # callable() → list[str]
        log_widget: QTextEdit,
        accent_color: str = "#a6e3a1",
        on_started=None,
        status_check_cmd: Optional[list[str]] = None,
        external_stop_cmd: Optional[list[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._label    = label
        self._cmd_fn   = cmd_fn
        self._log      = log_widget
        self._accent   = accent_color
        self._on_started = on_started
        self._status_check_cmd = status_check_cmd
        self._external_stop_cmd = external_stop_cmd
        self._worker: Optional[NodeWorker] = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color:{C['border']}; font-size:14px;")
        self._status_dot.setFixedWidth(14)
        lay.addWidget(self._status_dot)

        self._lbl = QLabel(label)
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(self._lbl)

        # Create a dedicated log tab for this node.
        # NOTE: parent_tabs is always None here (built before _build_tabs runs).
        # Tab registration happens in VisionPipelineGUI._build_tabs() instead.
        self._node_log = QTextEdit()
        self._node_log.setReadOnly(True)
        self._node_log.setFont(QFont("Monospace", 9))

        self._btn = QPushButton("▶  Start")
        self._btn.setFixedWidth(90)
        self._btn.setStyleSheet(
            f"background:{C['panel']}; color:{C['text']};"
            " border:1px solid #585b70; border-radius:5px; padding:4px 8px;"
        )
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)

        self._status_timer = None
        if self._status_check_cmd:
            self._status_timer = QTimer(self)
            self._status_timer.timeout.connect(self._sync_external_state)
            self._status_timer.start(1500)
            self._sync_external_state()

    def is_running(self) -> bool:
        return (
            (self._worker is not None and self._worker.isRunning())
            or self._check_external_running()
        )

    def stop(self) -> None:
        if self._worker:
            self._worker.abort()
            if not hasattr(self, "_graveyard"):
                self._graveyard = []
            self._graveyard.append(self._worker)
            self._worker.finished.connect(lambda rc, w=self._worker: self._graveyard.remove(w) if w in getattr(self, "_graveyard", []) else None)
            self._worker = None
        elif self._check_external_running() and self._external_stop_cmd:
            subprocess.run(
                self._external_stop_cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(WORKSPACE_DIR),
            )
        self._set_stopped()

    def _toggle(self) -> None:
        if self.is_running():
            self.stop()
        else:
            self._start()

    def _start(self) -> None:
        if self._check_external_running():
            self._set_running()
            self._log.append(
                f'<span style="color:{self._accent}">[{self._label}] Already running.</span>'
            )
            return
        cmd = self._cmd_fn()
        if not cmd:
            return
        self._worker = NodeWorker(cmd)
        
        def _handle_log(s):
            formatted = f'<span style="color:{C["text2"]}">[{self._label}] {s}</span>'
            self._log.append(formatted) # Write to the main "All Nodes" log
            if hasattr(self, '_node_log'):
                self._node_log.append(formatted) # Write to the dedicated tab
                
        self._worker.line_out.connect(_handle_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._set_running()
        if self._on_started:
            self._on_started()

    def _on_finished(self, rc: int) -> None:
        self._worker = None
        if self._check_external_running():
            self._set_running()
        else:
            self._set_stopped()

    def _set_stopped(self) -> None:
        self._btn.setText("▶  Start")
        self._btn.setStyleSheet(
            f"background:{C['panel']}; color:{C['text']};"
            " border:1px solid #585b70; border-radius:5px; padding:4px 8px;"
        )
        self._status_dot.setStyleSheet(f"color:{C['border']}; font-size:14px;")

    def _set_running(self) -> None:
        self._btn.setText("■  Stop")
        self._btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']};"
            " border:none; border-radius:5px; padding:4px 8px; font-weight:bold;"
        )
        self._status_dot.setStyleSheet(
            f"color:{self._accent}; font-size:14px;"
        )

    def _check_external_running(self) -> bool:
        if not self._status_check_cmd:
            return False
        try:
            result = subprocess.run(
                self._status_check_cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(WORKSPACE_DIR),
            )
            return result.returncode == 0
        except Exception:
            return False

    def _sync_external_state(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._set_running()
        elif self._check_external_running():
            self._set_running()
        else:
            self._set_stopped()



# ─────────────────────────────────────────────────────────────────────────────
# CropROIView — interactive image panel for the Crop Image tab
# ─────────────────────────────────────────────────────────────────────────────

class CropROIView(QLabel):
    """
    Displays the live image from /vision/captured_image_raw and lets the user
    drag a yellow selection rectangle to define the crop ROI.

    Signals:
        roi_changed(tuple)   emitted when user releases mouse with (x,y,w,h) in
                             image pixel coordinates.
    """
    roi_changed  = Signal(tuple)   # (x, y, w, h) bounding box in image coords
    pixel_sampled = Signal("QColor")  # emitted when user picks a color with eyedropper

    def __init__(self, ros_node=None, bridge=None, parent=None):
        super().__init__(parent)
        self._ros_node = ros_node
        self._bridge   = bridge
        self._subscription = None

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background:{C['panel']}; border:none;"
        )
        self.setText("⌛  Waiting for /vision/captured_image_raw …")
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        # Latest frame stored as QPixmap (scaled) and original (for coordinate mapping)
        self._pix:       Optional[QPixmap]   = None   # displayed pixmap
        self._img_size:  Optional[tuple]     = None   # (orig_w, orig_h)

        # ROI Selection (Polygon mode)
        self._poly_pts: list[QPointF] = []            # list of clicked points in label coords
        self._current_mouse: Optional[QPointF] = None # for drawing the line to the cursor
        self._roi_px: Optional[tuple] = None          # (x,y,w,h) in image coords bounding box
        self._saved_roi: Optional[tuple] = None       # loaded from config

        # Make sure the widget can receive keyboard focus for Escape key
        self.setFocusPolicy(Qt.StrongFocus)
        
        # ROS subscription
        if ROS2_OK and CV2_OK and ros_node:
            self._subscription = ros_node.create_subscription(
                ROSImage, "/vision/captured_image_raw", self._ros_cb, qos_profile_sensor_data
            )
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._repaint)
            self._refresh_timer.start(100)

        self._latest_frame: Optional[np.ndarray] = None    # RGB numpy
        self._cv_frame:     Optional[np.ndarray] = None    # BGR numpy (for eyedropper)
        self._eyedropper_mode: bool = False

    # ── ROS callback ─────────────────────────────────────────────────

    def _ros_cb(self, msg: "ROSImage") -> None:
        if not (CV2_OK and self._bridge):
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self._latest_frame = frame
            self._img_size = (frame.shape[1], frame.shape[0])
            # Keep a BGR copy for eyedropper pixel sampling
            self._cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    def _repaint(self) -> None:
        if self._latest_frame is None:
            return
        frame = self._latest_frame
        h, w, ch = frame.shape
        lw, lh = self.width(), self.height()
        if lw < 4 or lh < 4:
            return
        scale = min(lw / w, lh / h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        small = cv2.resize(frame, (nw, nh))
        self._img_gc_cache = small  # KEEP ALIVE for PySide reference count
        qimg = QImage(small.data, nw, nh, int(small.strides[0]), QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(qimg)
        self.update()

    # ── Painting ──────────────────────────────────────────────────────

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        if self._pix is None:
            return

        painter = QPainter(self)

        # Draw the image centred
        pw, ph = self._pix.width(), self._pix.height()
        ox = (self.width()  - pw) // 2
        oy = (self.height() - ph) // 2
        painter.drawPixmap(ox, oy, self._pix)

        # Draw saved ROI (blue, dashed)
        if self._saved_roi and self._img_size:
            rx, ry, rw, rh = self._label_rect(self._saved_roi, ox, oy, pw, ph)
            pen = QPen(QColor("#89b4fa"), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(rx), int(ry), int(rw), int(rh))

        # Draw current polygon (yellow)
        if self._poly_pts:
            pen = QPen(QColor("#f9e2af"), 2, Qt.SolidLine)
            painter.setPen(pen)
            
            # Draw existing lines
            for i in range(1, len(self._poly_pts)):
                painter.drawLine(self._poly_pts[i-1], self._poly_pts[i])
                
            # Draw line to current mouse position if not closed
            if self._current_mouse is not None and not self._roi_px:
                painter.drawLine(self._poly_pts[-1], self._current_mouse)
                
            # Draw points
            painter.setBrush(QColor("#f9e2af"))
            for pt in self._poly_pts:
                painter.drawEllipse(pt, 3, 3)
                
            # If closed or calculated, draw bounding box preview (dashed)
            if self._roi_px is not None:
                # Draw closing line
                if len(self._poly_pts) > 2:
                    painter.drawLine(self._poly_pts[-1], self._poly_pts[0])
                    
                rx, ry, rw, rh = self._label_rect(self._roi_px, ox, oy, pw, ph)
                box_pen = QPen(QColor("#f9e2af"), 1, Qt.DashLine)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(box_pen)
                painter.drawRect(int(rx), int(ry), int(rw), int(rh))

        painter.end()

    def _label_rect(self, roi_px, ox, oy, pw, ph):
        """Convert image-coordinate roi tuple to label-coordinate box."""
        if self._img_size is None:
            return (0, 0, 0, 0)
        iw, ih = self._img_size
        sx, sy = pw / iw, ph / ih
        x, y, w, h = roi_px
        return (ox + x * sx, oy + y * sy, w * sx, h * sy)

    def _drag_rect_label(self) -> QRectF:
        x0, y0 = self._drag_start.x(), self._drag_start.y()
        x1, y1 = self._drag_end.x(),   self._drag_end.y()
        return QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    # ── Mouse events ──────────────────────────────────────────────────

    def _label_to_image(self, lx: float, ly: float) -> tuple[int, int]:
        """Map label pixel → image pixel."""
        if self._pix is None or self._img_size is None:
            return (0, 0)
        pw, ph = self._pix.width(), self._pix.height()
        ox = (self.width()  - pw) // 2
        oy = (self.height() - ph) // 2
        iw, ih = self._img_size
        ix = int((lx - ox) / pw * iw)
        iy = int((ly - oy) / ph * ih)
        return (max(0, min(ix, iw)), max(0, min(iy, ih)))

    def mousePressEvent(self, ev) -> None:
        self.setFocus() # ensure we get keyboard events

        # ── Eyedropper mode: sample pixel colour ───────────────────────────
        if getattr(self, "_eyedropper_mode", False) and ev.button() == Qt.LeftButton:
            if self._pix is not None and self._cv_frame is not None:
                ix, iy = self._label_to_image(ev.pos().x(), ev.pos().y())
                ih, iw = self._cv_frame.shape[:2]
                ix = max(0, min(ix, iw - 1))
                iy = max(0, min(iy, ih - 1))
                px = self._cv_frame[iy, ix]  # BGR
                color = QColor(int(px[2]), int(px[1]), int(px[0]))
                self._eyedropper_mode = False
                self.setCursor(Qt.ArrowCursor)
                self.pixel_sampled.emit(color)
            return

        # ── Normal polygon drawing ──────────────────────────────────────
        if ev.button() == Qt.LeftButton and self._pix is not None:
            # If we already have a calculated ROI, start a new one
            if self._roi_px is not None:
                self._poly_pts.clear()
                self._roi_px = None
                
            pt = QPointF(ev.pos())
            
            # Auto-close if clicked near the first point (and we have at least 3 points)
            if len(self._poly_pts) >= 3 and (pt - self._poly_pts[0]).manhattanLength() < 15:
                self._calculate_bounding_roi()
            else:
                self._poly_pts.append(pt)
                
            self.update()
            
        elif ev.button() == Qt.RightButton and self._poly_pts:
            # Right click finishes the polygon immediately
            if len(self._poly_pts) >= 2:
                self._calculate_bounding_roi()
            else:
                self._poly_pts.clear()
            self.update()

    def mouseMoveEvent(self, ev) -> None:
        if self._poly_pts and self._roi_px is None:
            self._current_mouse = QPointF(ev.pos())
            self.update()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_Escape:
            self.clear_roi()
        else:
            super().keyPressEvent(ev)
            
    def _calculate_bounding_roi(self) -> None:
        """Calculate the tightest bounding box around the polygon in image coordinates."""
        if len(self._poly_pts) < 2:
            return
            
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for pt in self._poly_pts:
            img_x, img_y = self._label_to_image(pt.x(), pt.y())
            min_x = min(min_x, img_x)
            min_y = min(min_y, img_y)
            max_x = max(max_x, img_x)
            max_y = max(max_y, img_y)
            
        w, h = max_x - min_x, max_y - min_y
        
        if w > 4 and h > 4:
            self._roi_px = (min_x, min_y, w, h)
            self._current_mouse = None
            self.roi_changed.emit(self._roi_px)
        else:
            self.clear_roi()

    # ── Public API ────────────────────────────────────────────────────

    def current_roi(self) -> Optional[tuple]:
        return self._roi_px

    def get_polygon_image_coords(self) -> list:
        """Return the current polygon vertices as [[x,y],...] in image pixel coords."""
        return [
            list(self._label_to_image(pt.x(), pt.y()))
            for pt in self._poly_pts
        ]

    def set_saved_roi(self, roi: tuple) -> None:
        """Display the previously saved ROI (loaded from config on startup)."""
        self._saved_roi = roi
        self._roi_px    = roi
        self.update()

    def clear_roi(self) -> None:
        self._poly_pts.clear()
        self._current_mouse = None
        self._roi_px     = None
        self._saved_roi  = None
        self.update()


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ColorPickerWidget — reusable colour swatch + eyedropper
# ─────────────────────────────────────────────────────────────────────────────

class ColorPickerWidget(QWidget):
    """A reusable widget that shows a colour swatch button + eyedropper.

    Usage::

        cpw = ColorPickerWidget(QColor(255, 0, 0))   # default red
        cpw.color_changed.connect(my_slot)
        layout.addWidget(cpw)
        r, g, b = cpw.color().red(), cpw.color().green(), cpw.color().blue()

    The ``eyedropper_view`` property must be set to a CropROIView instance for
    the eyedropper to work.  If not set, the eyedropper button is disabled.
    """
    color_changed = Signal("QColor")

    def __init__(self, default_color: "QColor" = None, parent=None):
        super().__init__(parent)
        if default_color is None:
            from PySide6.QtGui import QColor as _QC
            default_color = _QC(0, 0, 0)
        self._color = default_color
        self._dialog = None
        self.eyedropper_view = None   # set externally if eyedropper is needed

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._swatch = QPushButton()
        self._swatch.setFixedSize(28, 22)
        self._swatch.setToolTip("Click to choose colour")
        self._swatch.clicked.connect(self._pick)
        row.addWidget(self._swatch)

        self._eye_btn = QPushButton("\U0001f52c")
        self._eye_btn.setFixedSize(28, 22)
        self._eye_btn.setToolTip(
            "Eyedropper: pick colour from the crop view.\n"
            "Click this, then click a pixel on the Full Frame panel."
        )
        self._eye_btn.clicked.connect(self._activate_eyedropper)
        row.addWidget(self._eye_btn)

        self._refresh_swatch()

    def color(self) -> "QColor":
        return self._color

    def set_color(self, c: "QColor") -> None:
        self._color = c
        self._refresh_swatch()
        self.color_changed.emit(c)

    def _refresh_swatch(self) -> None:
        c = self._color
        hex_bg = c.name()
        luma = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        fg = "#000" if luma > 128 else "#fff"
        self._swatch.setStyleSheet(
            f"background:{hex_bg}; color:{fg};"
            " border:1px solid #585b70; border-radius:3px;"
        )
        self._swatch.setToolTip(
            f"Colour: {hex_bg} (RGB {c.red()},{c.green()},{c.blue()})\nClick to change."
        )

    def _pick(self) -> None:
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtCore import Qt
        if self._dialog is not None:
            self._dialog.raise_()
            self._dialog.activateWindow()
            return
        dlg = QColorDialog(self._color, self)
        dlg.setWindowTitle("Choose colour")
        dlg.setWindowModality(Qt.NonModal)
        dlg.setOption(QColorDialog.ShowAlphaChannel, False)

        def _accepted():
            c = dlg.currentColor()
            if c.isValid():
                self.set_color(c)
            self._dialog = None

        dlg.accepted.connect(_accepted)
        dlg.rejected.connect(lambda: setattr(self, "_dialog", None))
        dlg.finished.connect(lambda _: setattr(self, "_dialog", None))
        self._dialog = dlg
        dlg.show()

    def _activate_eyedropper(self) -> None:
        v = self.eyedropper_view
        if v is None:
            return
        v._eyedropper_mode = True
        v.setCursor(Qt.CrossCursor)
        # wire once; disconnect previous if any
        try:
            v.pixel_sampled.disconnect(self._on_eyedrop)
        except RuntimeError:
            pass
        v.pixel_sampled.connect(self._on_eyedrop)

    def _on_eyedrop(self, color: "QColor") -> None:
        self.set_color(color)
        try:
            self.eyedropper_view.pixel_sampled.disconnect(self._on_eyedrop)
        except RuntimeError:
            pass


class VisionPipelineGUI(QMainWindow):
    # Use a custom Qt Signal to thread-safely enforce manual canvas redraws
    manual_image_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6  ·  Vision Pipeline Launcher")
        self.resize(1100, 680)
        self._latest_cropped_rgb: Optional[np.ndarray] = None
        self._crop_set_params_client = None
        self._crop_clear_client = None
        self._crop_futures = []
        self._trigger_workers: list = []  # keeps NodeWorker threads alive until done

        # ROS 2 preview infrastructure (optional)
        self._ros_node = None
        self._bridge   = None
        if ROS2_OK:
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = rclpy.create_node("vision_pipeline_gui")
            if CV2_OK:
                self._bridge = CvBridge()
            # Spin ROS in background — guard against context teardown errors
            def _spin_once_safe():
                try:
                    if rclpy.ok():
                        rclpy.spin_once(self._ros_node, timeout_sec=0.005)
                except Exception:
                    pass
            self._ros_timer = QTimer(self)
            self._ros_timer.timeout.connect(_spin_once_safe)
            self._ros_timer.start(10)
            self._crop_set_params_client = self._ros_node.create_client(SetParameters, "/crop_image/set_parameters")
            self._crop_clear_client  = self._ros_node.create_client(Trigger, "/crop_image/clear_roi")
            self._crop_reload_client = self._ros_node.create_client(Trigger, "/crop_image/reload_roi")
            self._manual_set_params_client = self._ros_node.create_client(SetParameters, "/manual_line/set_parameters")
            self._manual_set_strokes_client = self._ros_node.create_client(Trigger, "/manual_line/set_strokes")
            self._manual_cropped_sub = None
            if CV2_OK:
                self._manual_cropped_sub = self._ros_node.create_subscription(
                    ROSImage,
                    "/vision/captured_image_color",
                    self._manual_topic_image_cb,
                    qos_profile_sensor_data,
                )

        self._build_ui()
        if not ROS2_OK and ROS2_IMPORT_ERROR:
            print(f"[VisionPipelineGUI] ROS import failed: {ROS2_IMPORT_ERROR}", file=sys.stderr)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Accent bar at very top
        accent_bar = QFrame()
        accent_bar.setFixedHeight(4)
        accent_bar.setStyleSheet(f"background:{C['accent']};")
        root.addWidget(accent_bar)

        # Initialize the global drawing canvas early so sidebar can reference it
        if CV2_OK:
            self._canvas = ManualCanvas()
            self.manual_image_ready.connect(self._canvas.load_image)
        else:
            self._canvas = QLabel("cv2 not available — Manual canvas disabled.")
            self._canvas.setAlignment(Qt.AlignCenter)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{C['panel']};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(20, 12, 20, 12)
        title = QLabel("🔭  PAROL6 Vision Pipeline")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet(f"color:{C['accent']};")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()
        badge_text = "ROS 2 ✅" if ROS2_OK else f"ROS 2 offline: {ROS2_IMPORT_ERROR or 'unknown error'}"
        ros_badge = QLabel(badge_text)
        ros_badge.setToolTip(ROS2_IMPORT_ERROR or badge_text)
        ros_badge.setStyleSheet(
            f"color:{'#a6e3a1' if ROS2_OK else '#f38ba8'}; font-size:11px;"
        )
        hdr_lay.addWidget(ros_badge)

        # Global Kill All Background Nodes Button
        hdr_lay.addSpacing(16)
        global_kill_btn = QPushButton("☠️ Kill All Background Nodes")
        global_kill_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:4px; padding:4px 12px;"
        )
        global_kill_btn.setToolTip("Terminate all orphaned ghost nodes (crop, capture, yolo, etc.) to fix shifting topics.")
        global_kill_btn.setCursor(Qt.PointingHandCursor)
        global_kill_btn.clicked.connect(self._kill_all)
        hdr_lay.addWidget(global_kill_btn)

        root.addWidget(hdr)

        # ── Main splitter: sidebar | tabs ─────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{C['border']};}}")
        root.addWidget(splitter)

        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        tabs = self._build_tabs()
        splitter.addWidget(tabs)

        splitter.setSizes([360, 1040])

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        wrap = QScrollArea()
        wrap.setWidgetResizable(True)
        wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrap.setStyleSheet(f"QScrollArea{{background:{C['bg']}; border:none;}}")
        wrap.setMinimumWidth(360)

        inner = QWidget()
        wrap.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # ── log widget (shared across all NodeButtons) ─────────────────
        # We assign an attribute `parent_tabs` later when the Log Tab is built
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(160)
        self._log.setFont(QFont("Monospace", 9))

        # ─── Stage 1: Camera ─────────────────────────────────────────────
        cam_grp = QGroupBox("Stage 1 — Camera")
        cg_lay  = QVBoxLayout(cam_grp)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Camera Backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItem("CPU (Stable/Default)", "cpu")
        self._backend_combo.addItem("CUDA (Nvidia GPU - Requires rebuild)", "cuda")
        self._backend_combo.addItem("OpenCL (AMD/Intel - Requires rebuild)", "opencl")
        backend_row.addWidget(self._backend_combo)
        cg_lay.addLayout(backend_row)

        self._btn_live_cam = NodeButton(
            "Live Kinect Camera",
            self._get_kinect_launch_cmd,
            self._log,
            C["blue"],
        )
        cg_lay.addWidget(self._btn_live_cam)

        kill_cam_btn = QPushButton("Kill Stale Camera")
        kill_cam_btn.clicked.connect(self._kill_camera_processes)
        kill_cam_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:5px; padding:5px;"
        )
        cg_lay.addWidget(kill_cam_btn)

        # Capture mode selector
        capture_mode_row = QHBoxLayout()
        capture_mode_row.addWidget(QLabel("Capture mode:"))
        self._capture_mode_combo = QComboBox()
        self._capture_mode_combo.addItem("keyboard (press 's')", "keyboard")
        self._capture_mode_combo.addItem("timed (auto)",        "timed")
        capture_mode_row.addWidget(self._capture_mode_combo)
        cg_lay.addLayout(capture_mode_row)

        self._btn_capture = NodeButton(
            "Capture Images Node",
            lambda: [
                "ros2", "run", "parol6_vision", "capture_images",
                "--ros-args",
                "-p", f"capture_mode:={self._capture_mode_combo.currentData()}",
                "-p", "output_topic:=/vision/captured_image_raw",
            ],
            self._log,
            C["green"],
            on_started=self._ensure_crop_node_running,
        )
        cg_lay.addWidget(self._btn_capture)

        trigger_btn = QPushButton("📷  Trigger Capture (keyboard: 's')")
        trigger_btn.clicked.connect(self._trigger_capture)
        trigger_btn.setStyleSheet(
            f"background:{C['yellow']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:5px; padding:5px;"
        )
        cg_lay.addWidget(trigger_btn)
        lay.addWidget(cam_grp)

        # ─── Stage 2: Processing Mode ─────────────────────────────────────
        mode_grp = QGroupBox("Stage 2 — Processing Mode")
        mg_lay   = QVBoxLayout(mode_grp)

        self._mode_buttons = QButtonGroup(self)
        modes = [
            ("🔮  YOLO Segment",       "yolo"),
            ("🎨  Color Mode",          "color"),
            ("✏️  Manual Red Line",     "manual"),
        ]
        for label, key in modes:
            rb = QRadioButton(label)
            rb.setProperty("mode_key", key)
            self._mode_buttons.addButton(rb)
            mg_lay.addWidget(rb)
        self._mode_buttons.buttons()[0].setChecked(True)

        # Switch to Manual Red Line tab when manual mode is selected
        def _on_mode_selected(btn):
            if btn.property('mode_key') == 'manual':
                if hasattr(self, '_main_tabs') and hasattr(self, '_manual_tab_index'):
                    self._main_tabs.setCurrentIndex(self._manual_tab_index)
                if CV2_OK and getattr(self, '_latest_cropped_rgb', None) is not None:
                    self.manual_image_ready.emit(self._latest_cropped_rgb.copy())

        self._mode_buttons.buttonClicked.connect(_on_mode_selected)

        # Capture-mode start button note: also launch crop_image node
        self._btn_crop_node = NodeButton(
            "Crop Image Node",
            lambda: ["ros2", "run", "parol6_vision", "crop_image"],
            self._log,
            "#74c7ec",
            # No status_check_cmd: avoid periodic blocking subprocess on the Qt thread.
            # Ownership (self._worker) tracks run state; explicit checks are done on demand.
            status_check_cmd=None,
            external_stop_cmd=[
                "bash", "-lc",
                "pkill -INT -f 'ros2 run parol6_vision crop_image'",
            ],
        )
        mg_lay.addWidget(self._btn_crop_node)

        self._btn_start_mode = QPushButton("▶  Start Selected Mode Node")
        self._btn_start_mode.setStyleSheet(
            f"background:{C['accent']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:6px;"
        )
        self._btn_start_mode.clicked.connect(self._start_mode_node)
        mg_lay.addWidget(self._btn_start_mode)

        self._btn_stop_mode = QPushButton("■  Stop Mode Node")
        self._btn_stop_mode.setEnabled(False)
        self._btn_stop_mode.clicked.connect(self._stop_mode_node)
        mg_lay.addWidget(self._btn_stop_mode)

        self._mode_worker: Optional[NodeWorker] = None
        lay.addWidget(mode_grp)

        # ─── Stage 3: Backend Pipeline ────────────────────────────────────
        pipe_grp = QGroupBox("Stage 3 — Backend Pipeline")
        pg_lay   = QVBoxLayout(pipe_grp)

        self._btn_optimizer = NodeButton(
            "Path Optimizer",
            lambda: ["ros2", "run", "parol6_vision", "path_optimizer"],
            self._log, "#fab387",
        )
        pg_lay.addWidget(self._btn_optimizer)

        self._btn_depth = NodeButton(
            "Depth Matcher",
            lambda: ["ros2", "run", "parol6_vision", "depth_matcher"],
            self._log, "#89dceb",
        )
        pg_lay.addWidget(self._btn_depth)

        self._btn_pathgen = NodeButton(
            "Path Generator",
            lambda: ["ros2", "run", "parol6_vision", "path_generator"],
            self._log, "#f9e2af",
        )
        pg_lay.addWidget(self._btn_pathgen)

        launch_all_btn = QPushButton("🚀  Launch Full Pipeline (stages 1-3)")
        launch_all_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:6px;"
        )
        launch_all_btn.clicked.connect(self._launch_full_pipeline)
        pg_lay.addWidget(launch_all_btn)

        stop_all_btn = QPushButton("☠️  Stop All Nodes")
        stop_all_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:6px;"
        )
        stop_all_btn.clicked.connect(self._stop_all)
        pg_lay.addWidget(stop_all_btn)

        lay.addWidget(pipe_grp)

        # ─── Stage 4: MoveIt ─────────────────────────────────────────────
        mi_grp = QGroupBox("Stage 4 — Send to MoveIt")
        mi_lay = QVBoxLayout(mi_grp)

        self._btn_moveit = NodeButton(
            "MoveIt Controller",
            lambda: ["ros2", "run", "parol6_vision", "moveit_controller"],
            self._log, C["accent"],
        )
        mi_lay.addWidget(self._btn_moveit)

        send_btn = QPushButton("📡  Send Path → MoveIt  (trigger service)")
        send_btn.setStyleSheet(
            f"background:{C['accent']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:6px;"
        )
        send_btn.clicked.connect(self._send_path_to_moveit)
        mi_lay.addWidget(send_btn)

        open_launch_tab_btn = QPushButton("🚀  Open ROS2 Launch Tab")
        open_launch_tab_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px;"
        )
        open_launch_tab_btn.clicked.connect(
            lambda: self._main_tabs.setCurrentIndex(
                self._main_tabs.indexOf(self._ros_launch_tab)
            ) if hasattr(self, '_ros_launch_tab') else None
        )
        mi_lay.addWidget(open_launch_tab_btn)

        lay.addWidget(mi_grp)

        # ─── Legacy Tools (from vision_work/launcher.py) ──────────────────
        legacy_grp = QGroupBox("Legacy Vision Tools")
        lg_lay     = QVBoxLayout(legacy_grp)

        legacy_tools = [
            ("🔍  Detect Objects (YOLO)",        "yolo_training/yolo_gui.py"),
            ("〰️  Segment Seam (ResUNet)",        "resunet_training/weld_seam_gui.py"),
            ("🖍️  Manual Path Annotator",         "tools/manual_annotator.py"),
            ("🔮  Pipeline Prototyper",            "tools/pipeline_prototyper.py"),
            ("📦  Script Sandbox",                "tools/script_sandbox.py"),
            ("🎨  Mask Painter",                  "tools/mask_painter.py"),
            ("🔍  YOLO Inspector",                "tools/yolo_inspector.py"),
            ("〰️  Seam Inspector",                "tools/seam_inspector.py"),
            ("🧪  Mask Pipeline Tester",          "tools/mask_pipeline_tester.py"),
            ("🎬  Pipeline Studio",               "tools/pipeline_studio.py"),
            ("✏️  Annotation Studio",             "tools/annotation_studio.py"),
            ("📦  Batch YOLO Exporter",           "tools/batch_yolo.py"),
        ]
        for label, rel_path in legacy_tools:
            btn = QPushButton(label)
            btn.setProperty("script_path", rel_path)
            btn.clicked.connect(self._launch_legacy)
            btn.setStyleSheet(
                f"background:{C['panel']}; color:{C['text2']}; text-align:left;"
                f" border:1px solid {C['border']}; border-radius:5px; padding:5px 10px;"
            )
            lg_lay.addWidget(btn)

        lay.addWidget(legacy_grp)
        lay.addStretch()

        return wrap

    # ── Tabs (right panel) ────────────────────────────────────────────────────

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self._main_tabs = tabs

        # ── Tab 1: Visual Outputs ─────────────────────────────────────────
        vis_tab = QTabWidget()

        previews = [
            ("/kinect2/sd/image_color_rect",           "📷 Live Camera"),
            ("/vision/captured_image_raw",              "📸 Captured Frame"),
            ("/vision/processing_mode/annotated_image", "🔍 Processing Output"),
            ("/vision/processing_mode/debug_image",     "🧠 Mode Debug (YOLO/Color)"),
            ("/path_optimizer/debug_image",             "📊 Path Optimizer Debug"),
        ]
        self._preview_panels = []
        self._captured_preview = None

        for idx, (topic, title) in enumerate(previews):
            frame = QWidget()
            frame.setStyleSheet(
                f"background:{C['bg']}; border:1px solid {C['border']};"
                " border-radius:4px;"
            )
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.setSpacing(2)
            title_lbl = QLabel(f"  {title}")
            title_lbl.setFixedHeight(22)
            title_lbl.setStyleSheet(
                f"color:{C['text2']}; font-size:10px; font-weight:bold;"
                f" background:{C['panel']}; padding:3px;"
            )
            fl.addWidget(title_lbl)
            if ROS2_OK and CV2_OK:
                prev = TopicPreviewLabel(topic, self._ros_node, self._bridge)
                prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                fl.addWidget(prev, 1)   # stretch=1 → image fills remaining space
            else:
                lbl = QLabel(f"⌛  {topic}\n(ROS2 / cv2 offline)")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(f"color:{C['text2']}; font-size:10px;")
                fl.addWidget(lbl, 1)
            self._preview_panels.append(frame)
            vis_tab.addTab(frame, title)

        tabs.addTab(vis_tab, "👁  Visual Outputs")

        # ── Tab 2: Manual Red Line ─────────────────────────────────────────
        manual_tab = QWidget()
        mt_lay = QVBoxLayout(manual_tab)
        mt_lay.setContentsMargins(8, 8, 8, 8)

        # Toolbar Row 1: File / Data Controls
        row1 = QHBoxLayout()
        load_btn = QPushButton("📂  Load Image")
        load_btn.clicked.connect(self._manual_load_image)
        row1.addWidget(load_btn)

        use_topic_btn = QPushButton("📥  Use Latest Cropped Frame")
        use_topic_btn.clicked.connect(self._manual_use_latest_cropped)
        row1.addWidget(use_topic_btn)

        save_btn = QPushButton("💾  Save Red-Line Image")
        save_btn.clicked.connect(self._manual_save)
        row1.addWidget(save_btn)

        ros_pub_btn = QPushButton("📡  Publish as ROS Image")
        ros_pub_btn.clicked.connect(self._manual_publish_ros)
        row1.addWidget(ros_pub_btn)
        
        row1.addStretch()
        mt_lay.addLayout(row1)

        # Toolbar Row 2: Drawing Tools
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Color:"))
        self._manual_color_picker = ColorPickerWidget(QColor(255, 0, 0), parent=manual_tab)
        row2.addWidget(self._manual_color_picker)
        row2.addSpacing(8)
        
        row2.addWidget(QLabel("Brush (px):"))
        self._manual_brush_spin = QSpinBox()
        self._manual_brush_spin.setRange(1, 150)
        self._manual_brush_spin.setValue(5)
        
        self._manual_brush_slider = QSlider(Qt.Horizontal)
        self._manual_brush_slider.setRange(1, 150)
        self._manual_brush_slider.setValue(5)
        self._manual_brush_slider.setFixedWidth(80)
        
        # Link them together
        self._manual_brush_spin.valueChanged.connect(self._manual_brush_slider.setValue)
        self._manual_brush_slider.valueChanged.connect(self._manual_brush_spin.setValue)
        
        self._manual_brush_spin.valueChanged.connect(
            lambda v: self._canvas.set_brush(v) if getattr(self, '_canvas', None) else None
        )
        row2.addWidget(self._manual_brush_slider)
        row2.addWidget(self._manual_brush_spin)
        row2.addSpacing(8)

        self._straight_line_check = QCheckBox("📐  Straight-line mode (shift locks H/V)")
        self._straight_line_check.setToolTip("When enabled, each click adds a waypoint.\nHold Shift to snap to horizontal or vertical.")
        self._straight_line_check.stateChanged.connect(self._on_straight_line_toggled)
        row2.addWidget(self._straight_line_check)
        row2.addSpacing(8)

        clear_btn = QPushButton("🗑  Clear")
        clear_btn.setStyleSheet(f"background:{C['panel']}; color:{C['text2']}; border:1px solid {C['border']}; border-radius:5px; padding:4px 8px;")
        clear_btn.clicked.connect(lambda: self._canvas.clear_mask() if CV2_OK else None)
        row2.addWidget(clear_btn)

        send_strokes_btn = QPushButton("📤  Send Strokes")
        send_strokes_btn.setToolTip("Serialize current strokes and send to manual_line node.\nConfig is saved to ~/.parol6/manual_line_config.json.")
        send_strokes_btn.setStyleSheet(f"background:{C['accent']}; color:{C['bg']}; font-weight:bold; border:none; border-radius:5px; padding:4px 8px;")
        send_strokes_btn.clicked.connect(self._manual_send_strokes)
        row2.addWidget(send_strokes_btn)

        reset_strokes_btn = QPushButton("🔄  Reset")
        reset_strokes_btn.setToolTip("Clear strokes from node and delete saved config.")
        reset_strokes_btn.setStyleSheet(f"background:{C['red']}; color:{C['bg']}; font-weight:bold; border:none; border-radius:5px; padding:4px 8px;")
        reset_strokes_btn.clicked.connect(self._manual_reset_strokes)
        row2.addWidget(reset_strokes_btn)
        
        row2.addStretch()
        mt_lay.addLayout(row2)

        if CV2_OK:
            canvas_hint = QLabel("Left-click drag to draw. Right-click to undo last point. Hold Shift with straight-line mode to lock H/V.")
        else:
            canvas_hint = QLabel("(cv2 offline — drawing disabled)")
        canvas_hint.setStyleSheet(f"color:{C['text2']}; font-size:10px;")
        mt_lay.addWidget(canvas_hint)

        mt_lay.addWidget(self._canvas, stretch=1)

        # Dedicated manual log panel
        manual_log_header = QHBoxLayout()
        manual_log_header.addWidget(QLabel("📋  Manual Log"))
        manual_log_header.addStretch()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.setStyleSheet(f"font-size:10px; padding:2px 6px; background:{C['panel']}; color:{C['text2']}; border:1px solid {C['border']}; border-radius:4px;")
        clear_log_btn.clicked.connect(lambda: self._manual_log.clear() if hasattr(self, '_manual_log') else None)
        manual_log_header.addWidget(clear_log_btn)
        mt_lay.addLayout(manual_log_header)

        self._manual_log = QTextEdit()
        self._manual_log.setReadOnly(True)
        self._manual_log.setFixedHeight(100)
        self._manual_log.setFont(QFont("Monospace", 9))
        self._manual_log.setStyleSheet(f"background:{C['bg']}; color:{C['text']}; border:1px solid {C['border']}; border-radius:4px;")
        mt_lay.addWidget(self._manual_log)

        tabs.addTab(manual_tab, "✏️  Manual Red Line")
        self._manual_tab_index = tabs.indexOf(manual_tab)

        # ── Tab 3: ROS Launch (firmware-configurator style ─ exact replica) ──
        ros_tab = QWidget()
        rt_lay = QVBoxLayout(ros_tab)
        rt_lay.setContentsMargins(16, 16, 16, 16)
        rt_lay.setSpacing(12)
        self._ros_launch_tab = ros_tab

        ros_title = QLabel("🚀 ROS2 / MoveIt Launcher")
        ros_title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        rt_lay.addWidget(ros_title)

        ros_hint = QLabel(
            "📋 <b>Method 1</b> Gazebo Only — load the 3D simulation world (no robot control).  "
            "<b>Method 2</b> Gazebo + MoveIt — full simulated robot you can plan and execute on.  "
            "<b>Method 3</b> Fake Hardware — RViz + fake joint states (no Teensy needed, for path planning).  "
            "<b>Method 4</b> Real Hardware (Current) — hardware bringup first, then MoveIt.  "
            "<b>Method 5</b> Real Hardware (Tested Single-Motor Legacy) — branch-locked bringup.  "
            "<span style='color:#fab387;'>⚠️ Only use real hardware methods after flashing firmware, homing, and testing limit switches.</span>  "
            "<b>☠️ Kill All</b> forcefully stops all running ROS 2 / Gazebo processes if something hangs."
        )
        ros_hint.setTextFormat(Qt.RichText)
        ros_hint.setWordWrap(True)
        ros_hint.setStyleSheet(
            "background:#1e1a2e; border:1px solid #cba6f7; border-radius:6px; "
            "color:#cdd6f4; font-size:11px; padding:6px 10px; margin-bottom:4px;"
        )
        rt_lay.addWidget(ros_hint)

        # ── Launch Mode group box ─────────────────────────────────────────────
        from PySide6.QtWidgets import QGroupBox as _QGB
        ctrls_grp = _QGB("Launch Mode")
        cl = QHBoxLayout(ctrls_grp)

        self._ros_method = QComboBox()
        self._ros_method.setMinimumWidth(300)
        self._ros_method.addItem("Method 1: Gazebo Only (Simulation World)",                    "launch_gazebo_only.sh")
        self._ros_method.addItem("Method 2: Gazebo AND MoveIt (Simulated)",                     "launch_moveit_with_gazebo.sh")
        self._ros_method.addItem("Method 3: MoveIt Fake (Standalone RViz)",                     "launch_moveit_fake.sh")
        self._ros_method.addItem("Method 4: MoveIt Real Hardware",                              "launch_moveit_real_hw.sh")
        self._ros_method.addItem("Method 5: MoveIt Real Hardware (Tested Single-Motor Legacy)", "launch_moveit_real_hw_tested_single_motor.sh")
        cl.addWidget(QLabel("Target:"))
        cl.addWidget(self._ros_method)

        self._ros_launch_btn = QPushButton("🚀 Launch")
        self._ros_launch_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._ros_launch_btn.clicked.connect(self._ros_launch_toggle)
        cl.addWidget(self._ros_launch_btn)

        ros_kill_btn = QPushButton("☠️ Kill All")
        ros_kill_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        ros_kill_btn.setToolTip("Forcefully kills all Gazebo, RViz, and MoveIt processes to clean up the environment.")
        ros_kill_btn.clicked.connect(self._kill_all)
        cl.addWidget(ros_kill_btn)

        cl.addSpacing(20)

        cl.addWidget(QLabel("Test Plan:"))
        self._test_shape_combo = QComboBox()
        self._test_shape_combo.addItems(["Straight", "Curve", "Circle", "ZigZag", "Live Camera (No Inject)"])
        cl.addWidget(self._test_shape_combo)

        self._test_btn = QPushButton("▶️ Run Auto-Test")
        self._test_btn.setStyleSheet(
            f"background:{C['yellow']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._test_btn.setToolTip("Starts the moveit_controller, injects the selected path shape, and executes it.")
        self._test_btn.clicked.connect(self._ros_run_auto_test)
        cl.addWidget(self._test_btn)

        cl.addStretch()
        rt_lay.addWidget(ctrls_grp)

        # ── Split log view ────────────────────────────────────────────────────
        logs_row = QHBoxLayout()
        logs_row.setContentsMargins(0, 0, 0, 0)

        rviz_grp = _QGB("ROS 2 / MoveIt Logs")
        rl = QVBoxLayout(rviz_grp)
        self._ros_log = QTextEdit()
        self._ros_log.setReadOnly(True)
        self._ros_log.setFont(QFont("Monospace", 9))
        self._ros_log.setLineWrapMode(QTextEdit.NoWrap)
        self._ros_log.setStyleSheet("background:#11111b; color:#a6adc8;")
        rl.addWidget(self._ros_log)
        logs_row.addWidget(rviz_grp)

        gazebo_grp = _QGB("Gazebo / Physics Logs")
        gl = QVBoxLayout(gazebo_grp)
        self._gazebo_log = QTextEdit()
        self._gazebo_log.setReadOnly(True)
        self._gazebo_log.setFont(QFont("Monospace", 9))
        self._gazebo_log.setLineWrapMode(QTextEdit.NoWrap)
        self._gazebo_log.setStyleSheet("background:#11111b; color:#89b4fa;")
        gl.addWidget(self._gazebo_log)
        logs_row.addWidget(gazebo_grp)

        rt_lay.addLayout(logs_row)

        self._ros_worker: Optional[NodeWorker] = None
        self._ros_test_worker: Optional[NodeWorker] = None
        self._launchers_dir = str(WORKSPACE_DIR / "scripts" / "launchers")
        tabs.addTab(ros_tab, "🚀  ROS Launch")

        # ── Tab 4: Crop Image ─────────────────────────────────────────────
        crop_tab = self._build_crop_tab()
        tabs.addTab(crop_tab, "✂️  Crop Image")

        # ── Tab 5: Console Log ────────────────────────────────────────────
        log_tab = QWidget()
        ll_lay = QVBoxLayout(log_tab)
        ll_lay.setContentsMargins(8, 8, 8, 8)
        
        # Upper area: clear button
        ctrl_ll = QHBoxLayout()
        clear_log_btn = QPushButton("🗑  Clear All Logs")
        
        def _clear_all_logs():
            self._log.clear()
            # Clear all node-specific logs
            for i in range(1, self._log_tabs_widget.count()):
                widget = self._log_tabs_widget.widget(i)
                if isinstance(widget, QTextEdit):
                    widget.clear()
                    
        clear_log_btn.clicked.connect(_clear_all_logs)
        clear_log_btn.setFixedWidth(120)
        ctrl_ll.addStretch()
        ctrl_ll.addWidget(clear_log_btn)
        ll_lay.addLayout(ctrl_ll)
        
        # Main area: The tab widget for logs
        self._log_tabs_widget = QTabWidget()
        
        # Disconnect any strict height constraints on the shared log
        self._log.setMaximumHeight(16777215) 
        self._log.setMinimumHeight(400)
        self._log_tabs_widget.addTab(self._log, "All Nodes")
        
        # Link the tabs widget so NodeButtons can add their tabs
        self._log.parent_tabs = self._log_tabs_widget
        
        # Add a dedicated log tab for Stage 2 (Processing Mode)
        self._mode_log = QTextEdit()
        self._mode_log.setReadOnly(True)
        self._mode_log.setFont(QFont("Monospace", 9))
        self._mode_log.setStyleSheet(self._log.styleSheet())
        self._log_tabs_widget.addTab(self._mode_log, "Stage 2 Mode")
        
        for attr in (
            '_btn_live_cam',
            '_btn_capture',
            '_btn_crop_node',
            '_btn_optimizer',
            '_btn_depth',
            '_btn_pathgen',
            '_btn_moveit',
        ):
            btn = getattr(self, attr, None)
            if btn and hasattr(btn, '_node_log') and self._log_tabs_widget.indexOf(btn._node_log) == -1:
                self._log_tabs_widget.addTab(btn._node_log, btn._label)
        
        ll_lay.addWidget(self._log_tabs_widget)
        tabs.addTab(log_tab, "📋  Console Logs")

        return tabs

    # ── Crop Image Tab ────────────────────────────────────────────────────────

    def _build_crop_tab(self) -> QWidget:
        """
        ✂️ Crop Image Tab
        ─────────────────
        Left panel: live full frame from /vision/captured_image_raw + ROI overlay.
        Right panel: live cropped output from /vision/captured_image_color.
        Bottom bar: status + Apply/Save/Reset buttons + config info.
        """
        tab = QWidget()
        root_lay = QVBoxLayout(tab)
        root_lay.setContentsMargins(8, 8, 8, 8)
        root_lay.setSpacing(6)

        # ── Top info bar ──────────────────────────────────────────────────
        info = QLabel(
            "📌 <b>Draw a polygon</b> on the full-frame image (left) — left-click to add vertices, "
            "right-click (or click near the first point) to close. "
            "<b>Mask mode</b> (recommended): keeps full resolution and zeros out pixels outside "
            "the polygon — depth coordinates stay valid for downstream nodes. "
            "<b>Crop mode</b>: cuts to the bounding box — changes image dimensions. "
            "Click <b>Apply & Save</b> to activate. <b>Reset</b> restores pass-through."
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        info.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['accent']};"
            f" border-radius:6px; color:{C['text2']}; font-size:11px; padding:8px;"
        )
        info.setMaximumHeight(80)
        root_lay.addWidget(info)

        # ── Dual preview ──────────────────────────────────────────────────
        preview_row = QHBoxLayout()

        # Left: full frame + ROI overlay
        left_frame = QWidget()
        left_frame.setStyleSheet(
            f"background:{C['bg']}; border:1px solid {C['border']}; border-radius:4px;"
        )
        lf_lay = QVBoxLayout(left_frame)
        lf_lay.setContentsMargins(0, 0, 0, 0)
        lf_lay.setSpacing(2)
        lhdr = QLabel("  📷  Full Frame — /vision/captured_image_raw")
        lhdr.setFixedHeight(22)
        lhdr.setStyleSheet(
            f"color:{C['text2']}; font-size:10px; font-weight:bold;"
            f" background:{C['panel']}; padding:3px;"
        )
        lf_lay.addWidget(lhdr)

        # CropROIView: shows ROS image with draggable ROI polygon
        self._crop_view = CropROIView(self._ros_node, self._bridge)
        self._crop_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lf_lay.addWidget(self._crop_view, 1)   # stretch → fills remaining height
        preview_row.addWidget(left_frame)

        # Right: cropped output
        right_frame = QWidget()
        right_frame.setStyleSheet(
            f"background:{C['bg']}; border:1px solid {C['border']}; border-radius:4px;"
        )
        rf_lay = QVBoxLayout(right_frame)
        rf_lay.setContentsMargins(0, 0, 0, 0)
        rf_lay.setSpacing(2)
        rhdr = QLabel("  ✂️  Masked/Cropped Output — /vision/captured_image_color")
        rhdr.setFixedHeight(22)
        rhdr.setStyleSheet(
            f"color:{C['text2']}; font-size:10px; font-weight:bold;"
            f" background:{C['panel']}; padding:3px;"
        )
        rf_lay.addWidget(rhdr)

        if ROS2_OK and CV2_OK:
            self._crop_output_preview = TopicPreviewLabel(
                "/vision/captured_image_color", self._ros_node, self._bridge
            )
        else:
            self._crop_output_preview = QLabel("⌛  /vision/captured_image_color\n(ROS2 / cv2 offline)")
            self._crop_output_preview.setAlignment(Qt.AlignCenter)
            self._crop_output_preview.setStyleSheet(
                f"background:{C['panel']}; color:{C['text2']}; font-size:10px;"
            )
        self._crop_output_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rf_lay.addWidget(self._crop_output_preview, 1)  # stretch → fills remaining height
        preview_row.addWidget(right_frame)
        preview_row.setStretch(0, 3)
        preview_row.setStretch(1, 2)

        root_lay.addLayout(preview_row)
        root_lay.setStretch(1, 1)

        # ── Bottom control bar ────────────────────────────────────────────
        ctrl_bar = QHBoxLayout()

        self._crop_status_lbl = QLabel("Status: No crop config (pass-through)")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")
        ctrl_bar.addWidget(self._crop_status_lbl)

        ctrl_bar.addStretch()

        # Mode selector
        ctrl_bar.addWidget(QLabel("Mode:"))
        self._crop_mode_combo = QComboBox()
        self._crop_mode_combo.addItem("🛡  Mask (preserve resolution)", "mask")
        self._crop_mode_combo.addItem("✂️  Crop (bounding box)", "crop")
        self._crop_mode_combo.setToolTip(
            "Mask: pixels outside polygon are zeroed — full resolution kept, "
            "depth coordinates valid.\n"
            "Crop: image is cut to the polygon's bounding box — resolution changes."
        )
        ctrl_bar.addWidget(self._crop_mode_combo)

        ctrl_bar.addSpacing(6)

        # ── Mask colour picker ────────────────────────────────────────
        # Default mask fill: black
        self._crop_mask_color = QColor(0, 0, 0)

        ctrl_bar.addWidget(QLabel("Mask fill:"))

        # Colour swatch — click to open QColorDialog
        self._color_swatch = QPushButton()
        self._color_swatch.setFixedSize(28, 22)
        self._color_swatch.setToolTip(
            "Click to choose the mask fill colour.\n"
            "Default: black (RGB 0,0,0)."
        )
        self._color_swatch.clicked.connect(self._on_pick_color)
        self._update_color_swatch()
        ctrl_bar.addWidget(self._color_swatch)

        # Eyedropper — sample a pixel from the left panel
        eyedrop_btn = QPushButton("🔬")
        eyedrop_btn.setFixedSize(28, 22)
        eyedrop_btn.setToolTip(
            "Pick fill colour from the image.\n"
            "Click, then click any pixel on the left panel."
        )
        eyedrop_btn.clicked.connect(self._on_eyedropper_activate)
        ctrl_bar.addWidget(eyedrop_btn)

        # Connect the crop view's pixel_sampled signal
        self._crop_view.pixel_sampled.connect(self._on_color_sampled)

        ctrl_bar.addSpacing(8)

        # Polygon vertex readout
        self._roi_readout = QLabel("Polygon: — ")
        self._roi_readout.setStyleSheet(
            f"color:{C['yellow']}; font-size:11px; font-weight:bold;"
        )
        ctrl_bar.addWidget(self._roi_readout)

        ctrl_bar.addSpacing(8)

        clear_roi_btn = QPushButton("🗑  Clear")
        clear_roi_btn.clicked.connect(self._crop_clear_roi)
        clear_roi_btn.setStyleSheet(
            f"background:{C['panel']}; color:{C['text2']}; border:1px solid {C['border']};"
            " border-radius:5px; padding:5px 10px;"
        )
        ctrl_bar.addWidget(clear_roi_btn)

        apply_btn = QPushButton("✅  Apply & Save")
        apply_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        apply_btn.clicked.connect(self._crop_apply_save)
        ctrl_bar.addWidget(apply_btn)

        reset_btn = QPushButton("↩  Reset (Pass-through)")
        reset_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        reset_btn.clicked.connect(self._crop_reset)
        ctrl_bar.addWidget(reset_btn)

        root_lay.addLayout(ctrl_bar)

        # Connect ROI change signal
        self._crop_view.roi_changed.connect(self._on_roi_changed)

        # Load and display existing config on startup
        self._crop_load_existing_config()

        return tab

    # ── Mask colour helpers ───────────────────────────────────────────────

    def _update_color_swatch(self) -> None:
        """Repaint the colour swatch button to show the current fill colour."""
        c = self._crop_mask_color
        hex_bg = c.name()          # e.g. '#000000'
        # Use white or black text depending on brightness
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        fg = "#000000" if lum > 128 else "#ffffff"
        self._color_swatch.setStyleSheet(
            f"background:{hex_bg}; color:{fg}; border:1px solid {C['border']};"
            " border-radius:3px;"
        )
        self._color_swatch.setToolTip(
            f"Mask fill colour: {hex_bg} (RGB {c.red()},{c.green()},{c.blue()}).\n"
            "Click to change."
        )

    def _on_pick_color(self) -> None:
        """Open a NON-MODAL QColorDialog so the panel stays visible and undimmed.

        Being non-modal means:
          • The image is not dimmed — you can clearly see the background colour.
          • You can drag the dialog anywhere to expose the workspace.
          • Use the 🔬 Eyedropper for pixel-accurate sampling straight from the image.
        """
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtCore import Qt

        if hasattr(self, "_color_dialog") and self._color_dialog is not None:
            # Bring existing dialog to front instead of opening a second one
            self._color_dialog.raise_()
            self._color_dialog.activateWindow()
            return

        dlg = QColorDialog(self._crop_mask_color, self)
        dlg.setWindowTitle("Choose mask fill colour")
        dlg.setWindowModality(Qt.NonModal)   # ← image stays fully visible
        dlg.setOption(QColorDialog.ShowAlphaChannel, False)

        # Wire up result signals
        def _accepted():
            color = dlg.currentColor()
            if color.isValid():
                self._crop_mask_color = color
                self._update_color_swatch()
                self._log.append(
                    f'<span style="color:{C["text2"]}">[Crop] Colour set to '
                    f'{color.name()} (RGB {color.red()},{color.green()},{color.blue()})</span>'
                )
            self._color_dialog = None

        def _rejected():
            self._color_dialog = None

        dlg.accepted.connect(_accepted)
        dlg.rejected.connect(_rejected)
        dlg.finished.connect(lambda _: setattr(self, "_color_dialog", None))

        self._color_dialog = dlg
        dlg.show()

    def _on_eyedropper_activate(self) -> None:
        """Toggle eyedropper mode: next click on the left panel samples a pixel."""
        self._crop_view._eyedropper_mode = True
        self._crop_view.setCursor(Qt.CrossCursor)
        self._log.append(
            f'<span style="color:{C["text2"]}">[Crop] Eyedropper active — '
            'click a pixel on the left panel to sample its colour.</span>'
        )

    def _on_color_sampled(self, color: "QColor") -> None:
        """Receive the colour sampled by the eyedropper and update the swatch."""
        self._crop_mask_color = color
        self._update_color_swatch()
        self._log.append(
            f'<span style="color:{C["text2"]}">[Crop] Sampled colour: '
            f'{color.name()} (RGB {color.red()},{color.green()},{color.blue()})</span>'
        )

    def _crop_load_existing_config(self) -> None:
        """Read ~/.parol6/crop_config.json and update status label + mode selector."""
        import json
        from pathlib import Path
        cfg_path = Path.home() / ".parol6" / "crop_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if not cfg.get("enabled"):
                    raise ValueError("disabled")

                mode = cfg.get("mode", "crop")
                # Set the combo to match saved mode
                idx = self._crop_mode_combo.findData(mode)
                if idx >= 0:
                    self._crop_mode_combo.setCurrentIndex(idx)

                mask_color = cfg.get("mask_color", [0, 0, 0])
                if len(mask_color) == 3:
                    self._crop_mask_color = QColor(mask_color[0], mask_color[1], mask_color[2])
                    self._update_color_swatch()

                if mode == "mask" and cfg.get("polygon"):
                    poly = cfg["polygon"]
                    roi = None
                    # Compute bounding box for display / set_saved_roi
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    roi = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                    self._crop_status_lbl.setText(
                        f"✅  Active: Mask mode — {len(poly)} polygon pts"
                    )
                    self._roi_readout.setText(f"Polygon: {len(poly)} pts")
                elif "x" in cfg:
                    roi = (cfg["x"], cfg["y"], cfg["width"], cfg["height"])
                    self._crop_status_lbl.setText(
                        f"✅  Active: Crop mode — ROI = {roi}"
                    )
                    self._roi_readout.setText(
                        f"Polygon: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}"
                    )
                else:
                    raise ValueError("no geometry")

                self._crop_status_lbl.setStyleSheet(
                    f"color:{C['green']}; font-size:11px; font-weight:bold;"
                )
                if roi:
                    self._crop_view.set_saved_roi(roi)
                return
            except Exception:
                pass
        self._crop_status_lbl.setText("Status: No crop config (pass-through)")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")

    def _on_roi_changed(self, roi: tuple) -> None:
        """Called when the user finishes drawing a polygon ROI."""
        x, y, w, h = roi
        n_pts = len(self._crop_view._poly_pts)
        self._roi_readout.setText(f"Polygon: {n_pts} pts  bbox=({x},{y},{w},{h})")
        self._crop_status_lbl.setText(
            f"📐  Pending: {n_pts}-pt polygon — click Apply & Save"
        )
        self._crop_status_lbl.setStyleSheet(
            f"color:{C['yellow']}; font-size:11px; font-weight:bold;"
        )

    def _ensure_crop_node_running(self) -> bool:
        if not hasattr(self, "_btn_crop_node"):
            return False
        if self._btn_crop_node.is_running():
            return True
        self._btn_crop_node._start()
        return self._btn_crop_node.is_running()

    def _crop_apply_save(self) -> None:
        """Save the current polygon/ROI to config and push to /crop_image.

        Mask mode  — writes polygon points, calls ~/reload_roi only.
                     Output has the same resolution as input; depth coords preserved.
        Crop mode  — writes bounding box, calls set_parameters via retry loop.
                     Output is a smaller image; depth coords shift.
        """
        import json
        from pathlib import Path

        roi = self._crop_view.current_roi()
        if roi is None:
            QMessageBox.warning(self, "No ROI",
                                "Draw a polygon on the image first.")
            return

        x, y, w, h = roi
        mode = self._crop_mode_combo.currentData()   # "mask" or "crop"
        polygon = self._crop_view.get_polygon_image_coords()  # [[x,y],...]

        cfg_path = Path.home() / ".parol6" / "crop_config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "mask":
            # ── Mask mode ────────────────────────────────────────────────
            if len(polygon) < 3:
                QMessageBox.warning(self, "Not Enough Points",
                    "Need at least 3 polygon vertices for mask mode.")
                return

            cfg = {"enabled": True, "mode": "mask", "polygon": polygon,
                   "mask_color": [self._crop_mask_color.red(),
                                  self._crop_mask_color.green(),
                                  self._crop_mask_color.blue()]}
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._log.append(
                f'<b style="color:{C["green"]}">[Crop] Mask config saved → {cfg_path}  '
                f'{len(polygon)} pts</b>'
            )
            self._crop_status_lbl.setText(
                f"⏳  Applying mask ({len(polygon)} pts) …"
            )
            self._crop_status_lbl.setStyleSheet(
                f"color:{C['yellow']}; font-size:11px; font-weight:bold;"
            )
            self._ensure_crop_node_running()

            if not ROS2_OK or self._crop_reload_client is None:
                self._log.append(
                    f'<span style="color:{C["red"]}">[Crop] ROS not available.</span>'
                )
                return

            # Retry until reload_roi service is ready (node may be starting up)
            _MAX = 8
            _MS  = 500

            def _reload_attempt(remaining: int) -> None:
                if self._crop_reload_client.service_is_ready():
                    _do_reload()
                elif remaining > 0:
                    self._log.append(
                        f'<span style="color:{C["text2"]}">[Crop] Waiting for '
                        f'/crop_image/reload_roi service ({remaining} left)…</span>'
                    )
                    QTimer.singleShot(_MS, lambda: _reload_attempt(remaining - 1))
                else:
                    self._log.append(
                        f'<span style="color:{C["red"]}">[Crop] reload_roi service '
                        'not available. Is the node running?</span>'
                    )

            def _do_reload() -> None:
                future = self._crop_reload_client.call_async(Trigger.Request())
                self._crop_futures.append(future)

                def _on_done(fut):
                    try:
                        resp = fut.result()
                        self._log.append(
                            f'<span style="color:{C["green"]}">[Crop] ✅ Mask applied: '
                            f'{resp.message}</span>'
                        )
                        self._crop_status_lbl.setText(
                            f"✅  Mask active: {len(polygon)} pts"
                        )
                        self._crop_status_lbl.setStyleSheet(
                            f"color:{C['green']}; font-size:11px; font-weight:bold;"
                        )
                        self._roi_readout.setText(f"Polygon: {len(polygon)} pts")
                    except Exception as exc:
                        self._log.append(
                            f'<span style="color:{C["red"]}">[Crop] reload_roi failed: {exc}</span>'
                        )
                    finally:
                        if fut in self._crop_futures:
                            self._crop_futures.remove(fut)

                future.add_done_callback(_on_done)

            _reload_attempt(_MAX)

        else:
            # ── Crop mode (legacy bbox) ───────────────────────────────────
            cfg = {"enabled": True, "mode": "crop",
                   "x": x, "y": y, "width": w, "height": h}
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._log.append(
                f'<b style="color:{C["green"]}">[Crop] Crop config saved → {cfg_path}  '
                f'ROI=({x},{y},{w},{h})</b>'
            )
            self._crop_status_lbl.setText(f"⏳  Applying crop ROI …")
            self._crop_status_lbl.setStyleSheet(
                f"color:{C['yellow']}; font-size:11px; font-weight:bold;"
            )
            self._ensure_crop_node_running()

            if not ROS2_OK or self._crop_set_params_client is None:
                self._log.append(
                    f'<span style="color:{C["red"]}">[Crop] ROS client unavailable.</span>'
                )
                return

            _MAX = 8
            _MS  = 500

            def _attempt(remaining: int) -> None:
                if self._crop_set_params_client.service_is_ready():
                    _do_call()
                elif remaining > 0:
                    self._log.append(
                        f'<span style="color:{C["text2"]}">[Crop] Waiting for '
                        f'/crop_image service ({remaining} left)…</span>'
                    )
                    QTimer.singleShot(_MS, lambda: _attempt(remaining - 1))
                else:
                    self._log.append(
                        f'<span style="color:{C["red"]}">[Crop] /crop_image service '
                        'not available. Is the node running?</span>'
                    )

            def _do_call() -> None:
                req = SetParameters.Request()
                req.parameters = [
                    RclpyParameter(
                        'roi', RclpyParameter.Type.INTEGER_ARRAY, [x, y, w, h]
                    ).to_parameter_msg()
                ]
                future = self._crop_set_params_client.call_async(req)
                self._crop_futures.append(future)

                def _on_done(fut):
                    try:
                        result = fut.result()
                        if result.results and result.results[0].successful:
                            self._log.append(
                                f'<span style="color:{C["green"]}">[Crop] ✅ ROI applied '
                                f'({x},{y},{w},{h})</span>'
                            )
                            _trigger_reload()
                            self._crop_status_lbl.setText(
                                f"✅  Crop active: ({x},{y}) {w}×{h}"
                            )
                            self._crop_status_lbl.setStyleSheet(
                                f"color:{C['green']}; font-size:11px; font-weight:bold;"
                            )
                            self._roi_readout.setText(
                                f"Polygon: x={x} y={y} w={w} h={h}"
                            )
                        else:
                            reason = (
                                result.results[0].reason if result.results else "unknown"
                            )
                            self._log.append(
                                f'<span style="color:{C["red"]}">[Crop] ⚠ Rejected: '
                                f'{reason}</span>'
                            )
                    except Exception as exc:
                        self._log.append(
                            f'<span style="color:{C["red"]}">[Crop] SetParameters failed: {exc}</span>'
                        )
                    finally:
                        if fut in self._crop_futures:
                            self._crop_futures.remove(fut)

                future.add_done_callback(_on_done)

            def _trigger_reload() -> None:
                if self._crop_reload_client is None:
                    return
                if not self._crop_reload_client.service_is_ready():
                    return
                future = self._crop_reload_client.call_async(Trigger.Request())
                self._crop_futures.append(future)

                def _on_r(fut):
                    try:
                        resp = fut.result()
                        self._log.append(
                            f'<span style="color:{C["text2"]}">[Crop] reload_roi: '
                            f'{resp.message}</span>'
                        )
                    except Exception:
                        pass
                    finally:
                        if fut in self._crop_futures:
                            self._crop_futures.remove(fut)

                future.add_done_callback(_on_r)

            _attempt(_MAX)

    def _crop_clear_roi(self) -> None:
        """Clear the drawn polygon on the canvas without saving."""
        self._crop_view.clear_roi()
        self._roi_readout.setText("Polygon: —")
        self._crop_status_lbl.setText("📐  Cleared — draw a new polygon")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")

    def _crop_reset(self) -> None:
        """Disable crop (pass-through) — call ~/clear_roi service and update config.

        Uses the same async retry loop as _crop_apply_save to avoid blocking the Qt
        main thread if the node was just started (service not immediately available).
        """
        if not ROS2_OK or self._crop_clear_client is None:
            self._log.append(
                f'<span style="color:{C["red"]}">[Crop] Clear-ROI ROS client unavailable.</span>'
            )
            return

        self._ensure_crop_node_running()

        _MAX = 8
        _MS  = 500

        def _clear_attempt(remaining: int) -> None:
            if self._crop_clear_client.service_is_ready():
                _do_clear()
            elif remaining > 0:
                self._log.append(
                    f'<span style="color:{C["text2"]}">[Crop] Waiting for '
                    f'/crop_image/clear_roi service ({remaining} left)…</span>'
                )
                QTimer.singleShot(_MS, lambda: _clear_attempt(remaining - 1))
            else:
                self._log.append(
                    f'<span style="color:{C["red"]}">[Crop] clear_roi service not available. '
                    'Is the node running?</span>'
                )

        def _do_clear() -> None:
            future = self._crop_clear_client.call_async(Trigger.Request())
            self._crop_futures.append(future)

            def _on_done(fut):
                try:
                    resp = fut.result()
                    self._log.append(
                        f'<span style="color:{C["text2"]}">[Crop] {resp.message}</span>'
                    )
                except Exception as exc:
                    self._log.append(
                        f'<span style="color:{C["red"]}">[Crop] Clear failed: {exc}</span>'
                    )
                finally:
                    if fut in self._crop_futures:
                        self._crop_futures.remove(fut)

            future.add_done_callback(_on_done)

        self._crop_view.clear_roi()
        self._roi_readout.setText("ROI: —")
        self._crop_status_lbl.setText("⏳  Resetting to pass-through…")
        self._crop_status_lbl.setStyleSheet(f"color:{C['yellow']}; font-size:11px; font-weight:bold;")
        self._log.append(
            f'<span style="color:{C["text2"]}">[Crop] Requesting pass-through reset…</span>'
        )
        _clear_attempt(_MAX)


    # ── Actions ───────────────────────────────────────────────────────────────

        # Note: _start_live_cam logic is now handled inline by NodeButton

    def _trigger_capture(self) -> None:
        """Send capture trigger.
        If in 'keyboard' mode, send 's\n' to the capture_images_node's stdin via the NodeWorker if it exists.
        Otherwise, publish to /vision/capture_trigger for the 'timed' mode override or fallback.
        """
        mode = self._capture_mode_combo.currentData()
        
        if mode == 'keyboard' and getattr(self._btn_capture, "_worker", None) is not None:
            # Send 's' + Newline directly to the subprocess stdin
            proc = self._btn_capture._worker._proc
            if proc and proc.stdin:
                try:
                    proc.stdin.write("s\n")
                    proc.stdin.flush()
                    self._log.append(f'<span style="color:{C["yellow"]}">[Trigger] Sent "s" to stdin</span>')
                    return
                except Exception as e:
                    self._log.append(f'<span style="color:{C["red"]}">[Trigger] Failed to write to stdin: {e}</span>')
        
        # Fallback to topic publish
        if not hasattr(self, '_trigger_workers'):
            self._trigger_workers = []
            
        worker = NodeWorker(
            ["ros2", "topic", "pub", "--once",
             "/vision/capture_trigger", "std_msgs/msg/Empty", "{}"],
        )
        worker.line_out.connect(
            lambda s: self._log.append(
                f'<span style="color:{C["yellow"]}">[Trigger] {s}</span>'
            )
        )
        # cleanup completed workers
        worker.finished.connect(lambda rc, w=worker: self._trigger_workers.remove(w) if w in self._trigger_workers else None)
        self._trigger_workers.append(worker)
        worker.start()

    def _selected_mode(self) -> str:
        for btn in self._mode_buttons.buttons():
            if btn.isChecked():
                return btn.property("mode_key")
        return "yolo"

    def _start_mode_node(self) -> None:
        mode = self._selected_mode()
        cmds = {
            "yolo":   ["ros2", "run", "parol6_vision", "yolo_segment"],
            "color":  ["ros2", "run", "parol6_vision", "color_mode"],
            "manual": ["ros2", "run", "parol6_vision", "manual_line"],
        }
        cmd = cmds.get(mode)
        if not cmd:
            return
        if self._mode_worker and self._mode_worker.isRunning():
            self._stop_mode_node()
        self._mode_worker = NodeWorker(cmd)
        
        def _log_mode(s: str):
            styled = f'<span style="color:#89dceb">[Mode] {s}</span>'
            self._log.append(styled)
            if hasattr(self, '_mode_log'):
                self._mode_log.append(styled)
                
        self._mode_worker.line_out.connect(_log_mode)
        self._mode_worker.finished.connect(self._on_mode_finished)
        self._mode_worker.start()
        self._btn_start_mode.setEnabled(False)
        self._btn_stop_mode.setEnabled(True)
        self._log.append(
            f'<b style="color:{C["green"]}">[Mode] Started: {mode}</b>'
        )

    def _stop_mode_node(self) -> None:
        if self._mode_worker:
            self._mode_worker.abort()
            if not hasattr(self, "_graveyard"):
                self._graveyard = []
            self._graveyard.append(self._mode_worker)
            self._mode_worker.finished.connect(lambda rc, w=self._mode_worker: self._graveyard.remove(w) if w in getattr(self, "_graveyard", []) else None)
            self._mode_worker = None
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    def _on_mode_finished(self, rc: int) -> None:
        self._mode_worker = None
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    # ── Manual Red Line actions ───────────────────────────────────────────────

    def _on_straight_line_toggled(self, state: int) -> None:
        """Toggle straight-line drawing mode on the canvas."""
        if hasattr(self, '_canvas') and CV2_OK:
            self._canvas.set_straight_line_mode(bool(state))

    def _manual_log_append(self, html: str) -> None:
        """Append to both the global log and the manual-specific log panel."""
        self._log.append(html)
        if hasattr(self, '_manual_log'):
            self._manual_log.append(html)

    def _manual_send_strokes(self) -> None:
        """Serialise canvas strokes and send them to the manual_line node via rclpy."""
        if not (ROS2_OK and CV2_OK):
            self._manual_log_append('<span style="color:#f38ba8">[Manual] ROS2 + cv2 required.</span>')
            return
        strokes = self._canvas.get_strokes()
        if not strokes:
            self._manual_log_append('<span style="color:#f9e2af">[Manual] No strokes to send — draw something first.</span>')
            return
        c = self._manual_color_picker.color()
        w_px = self._manual_brush_spin.value()
        import json
        payload = {'color': [c.blue(), c.green(), c.red()], 'width': w_px, 'strokes': strokes}
        json_str = json.dumps(payload)
        self._manual_log_append(f'<span style="color:#a6e3a1">[Manual] Sending {len(strokes)} stroke(s) → /manual_line/set_strokes…</span>')
        
        if not hasattr(self, '_manual_set_params_client') or self._manual_set_params_client is None:
            self._manual_log_append('<span style="color:#f38ba8">[Manual] ROS client not ready.</span>')
            return
        
        # Use rclpy SetParameters directly — avoids shell JSON quoting issues
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        pv = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=json_str)
        req = SetParameters.Request()
        req.parameters = [Parameter(name='strokes_json', value=pv)]
        
        future = self._manual_set_params_client.call_async(req)
        
        def _on_param_set(fut):
            try:
                result = fut.result()
                if result and result.results and result.results[0].successful:
                    # Now call the set_strokes service
                    svc_w = NodeWorker([
                        "ros2", "service", "call", "/manual_line/set_strokes",
                        "std_srvs/srv/Trigger", "{}",
                    ])
                    def _on_svc_done(rc):
                        if rc == 0:
                            self._manual_publish_ros()
                        else:
                            self._manual_log_append('<span style="color:#f38ba8">[Manual] set_strokes service call failed.</span>')
                    svc_w.line_out.connect(lambda s: self._manual_log_append(f'<span style="color:#a6e3a1">[Manual] {s}</span>'))
                    svc_w.finished.connect(_on_svc_done)
                    self._trigger_workers.append(svc_w)
                    svc_w.finished.connect(lambda r, ww=svc_w: self._trigger_workers.remove(ww) if ww in self._trigger_workers else None)
                    svc_w.start()
                else:
                    msg = result.results[0].reason if result and result.results else 'unknown'
                    self._manual_log_append(f'<span style="color:#f38ba8">[Manual] Failed to set strokes param: {msg}</span>')
            except Exception as e:
                self._manual_log_append(f'<span style="color:#f38ba8">[Manual] Param set error: {e}</span>')
        
        future.add_done_callback(_on_param_set)

    def _manual_reset_strokes(self) -> None:
        """Call ~/reset_strokes on the manual_line node and clear the canvas."""
        if CV2_OK:
            self._canvas.clear_mask()
        if not ROS2_OK:
            return
        svc_w = NodeWorker([
            "ros2", "service", "call", "/manual_line/reset_strokes",
            "std_srvs/srv/Trigger", "{}",
        ])
        svc_w.line_out.connect(
            lambda s: self._log.append(
                f'<span style="color:#f9e2af">[Manual] {s}</span>'
            )
        )
        self._trigger_workers.append(svc_w)
        svc_w.finished.connect(lambda r, w=svc_w: self._trigger_workers.remove(w) if w in getattr(self, "_trigger_workers", []) else None)
        svc_w.start()
        self._log.append('<span style="color:#f9e2af">[Manual] Reset sent → /manual_line/reset_strokes</span>')
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    def _launch_full_pipeline(self) -> None:
        self._ensure_crop_node_running()
        self._btn_capture.stop()
        QTimer.singleShot(200, self._btn_capture._toggle)
        QTimer.singleShot(600, self._start_mode_node)
        QTimer.singleShot(1200, self._btn_optimizer._toggle)
        QTimer.singleShot(1600, self._btn_depth._toggle)
        QTimer.singleShot(2000, self._btn_pathgen._toggle)

    def _stop_all(self) -> None:
        for btn in (self._btn_live_cam, self._btn_capture,
                    self._btn_crop_node,
                    self._btn_optimizer, self._btn_depth,
                    self._btn_pathgen, self._btn_moveit):
            btn.stop()
        self._stop_mode_node()
        if self._ros_worker:
            self._ros_worker.abort()
            if not hasattr(self, "_graveyard"):
                self._graveyard = []
            self._graveyard.append(self._ros_worker)
            self._ros_worker.finished.connect(lambda rc, w=self._ros_worker: self._graveyard.remove(w) if w in getattr(self, "_graveyard", []) else None)
            self._ros_worker = None

    def _send_path_to_moveit(self) -> None:
        """Call the moveit_controller trigger service."""
        worker = NodeWorker(
            ["ros2", "service", "call",
             "/moveit_controller/execute_welding_path",
             "std_srvs/srv/Trigger", "{}"],
        )
        worker.line_out.connect(
            lambda s: self._log.append(
                f'<span style="color:{C["accent"]}">[MoveIt] {s}</span>'
            )
        )
        worker.start()

    def _launch_vision_moveit(self) -> None:
        worker = NodeWorker(
            ["ros2", "launch", "parol6_vision", "vision_moveit.launch.py"],
        )
        worker.line_out.connect(self._log.append)
        worker.start()

    # ── ROS Launch Tab actions ─────────────────────────────────────────────

    def _ros_launch_toggle(self) -> None:
        if self._ros_worker and self._ros_worker.isRunning():
            self._ros_log.append("[LAUNCH] Stopping...")
            self._gazebo_log.append("[LAUNCH] Stopping...")
            self._ros_worker.abort()
            self._ros_worker = None
            self._ros_launch_btn.setText("\U0001f680 Launch")
            self._ros_launch_btn.setStyleSheet(
                f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
                " border:none; border-radius:6px; padding:5px 14px;"
            )
            self._ros_method.setEnabled(True)
            return

        script_name = self._ros_method.currentData()   # e.g. 'launch_moveit_fake.sh'
        script_path = os.path.join(self._launchers_dir, script_name)

        self._ros_log.clear()
        self._gazebo_log.clear()

        if not os.path.exists(script_path):
            self._ros_log.append(f"[LAUNCH] \u274c Error: Cannot find script {script_path}")
            return

        worker = NodeWorker([script_path])

        def _route_line(s: str):
            lo = s.lower()
            if any(k in lo for k in ("ign", "gazebo", "/usr/bin/ruby", "spawn")):
                self._gazebo_log.append(s)
            else:
                self._ros_log.append(s)

        worker.line_out.connect(_route_line)
        worker.finished.connect(self._on_ros_launch_finished)
        self._ros_worker = worker
        worker.start()

        self._ros_launch_btn.setText("\u25a0 Stop")
        self._ros_launch_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._ros_method.setEnabled(False)

    def _on_ros_launch_finished(self, rc: int) -> None:
        self._ros_worker = None
        self._ros_launch_btn.setText("\U0001f680 Launch")
        self._ros_launch_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._ros_method.setEnabled(True)

    def _ros_run_auto_test(self) -> None:
        if self._ros_test_worker and self._ros_test_worker.isRunning():
            self._ros_log.append("[TEST] Stopping currently running test...")
            self._ros_test_worker.abort()
            self._ros_test_worker = None
            self._test_btn.setText("\u25b6\ufe0f Run Auto-Test")
            self._test_btn.setStyleSheet(
                f"background:{C['yellow']}; color:{C['bg']}; font-weight:bold;"
                " border:none; border-radius:6px; padding:5px 14px;"
            )
            return

        script_path = os.path.join(self._launchers_dir, "launch_auto_test.sh")
        if not os.path.exists(script_path):
            self._ros_log.append(f"[TEST] \u274c Cannot find script {script_path}")
            return

        shape = self._test_shape_combo.currentText()
        self._ros_log.append(f"\n[TEST] Launching Auto-Test ({shape})...")
        self._ros_log.append("[TEST] Spawning moveit_controller and waiting for services...")

        worker = NodeWorker([script_path, shape])
        worker.line_out.connect(self._ros_log.append)
        worker.finished.connect(self._on_ros_test_finished)
        self._ros_test_worker = worker
        worker.start()

        self._test_btn.setText("\u25a0 Stop Test")
        self._test_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )

    def _on_ros_test_finished(self, rc: int = 0) -> None:
        self._ros_test_worker = None
        self._test_btn.setText("\u25b6\ufe0f Run Auto-Test")
        self._test_btn.setStyleSheet(
            f"background:{C['yellow']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )

    def _inject_test_path(self) -> None:
        """Publish a small synthetic straight-line path to /vision/welding_path.

        The YAML value for `ros2 topic pub` must be a single flat YAML mapping.
        """
        # Three waypoints along Y axis at fixed X=0.4m, Z=0.45m
        path_yaml = (
            "{"
            "header: {frame_id: base_link}, "
            "poses: ["
            "{header: {frame_id: base_link}, pose: {position: {x: 0.4, y: -0.1, z: 0.45}, orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}}}, "
            "{header: {frame_id: base_link}, pose: {position: {x: 0.4, y:  0.0, z: 0.45}, orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}}}, "
            "{header: {frame_id: base_link}, pose: {position: {x: 0.4, y:  0.1, z: 0.45}, orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}}}"
            "]}"
        )

        worker = NodeWorker(
            ["ros2", "topic", "pub", "--once",
             "/vision/welding_path", "nav_msgs/msg/Path", path_yaml],
        )
        worker.line_out.connect(
            lambda s: self._ros_log.append(
                f'<span style="color:{C["yellow"]}">[Inject] {s}</span>'
            )
        )
        worker.start()
        self._ros_log.append(
            f'<b style="color:{C["yellow"]}">[Inject] Synthetic 3-point path published → '
            '/vision/welding_path</b>'
        )

    def _kill_all(self) -> None:
        """Stop all managed workers then kill stray background ROS processes.

        Uses SIGINT first (allows clean ROS graph deregistration) then SIGTERM
        after a short delay. SIGKILL (-9) is intentionally avoided so that nodes
        can call rclpy.shutdown() and cleanly remove themselves from the graph.
        """
        self._stop_all()
        # SIGINT → gives nodes time to deregister; SIGTERM as fallback
        cmd = (
            "pkill -INT -f 'rviz2|move_group|ros2_control_node|ros2 run parol6' ; "
            "sleep 1 ; "
            "pkill -TERM -f 'rviz2|move_group|ros2_control_node|ros2 run parol6' 2>/dev/null || true"
        )
        subprocess.Popen(["bash", "-c", cmd])
        self._ros_log.append("[KILL] Sent SIGINT to all stray ROS 2 processes (SIGTERM fallback after 1 s).")

    def _kill_camera_processes(self) -> None:
        cmd = "pkill -INT -f 'kinect2_bridge_node|kinect2_bridge_gpu|kinect2_bridge_launch'"
        subprocess.run(["bash", "-c", cmd], check=False)
        self._btn_live_cam._set_stopped()
        self._log.append(
            f'<span style="color:{C["red"]}">[Live Kinect Camera] Requested stop for stale camera processes.</span>'
        )

    def _get_kinect_launch_cmd(self) -> list[str]:
        backend = self._backend_combo.currentData()
        
        if backend == "cpu":
            # For CPU, reg_method is CPU, and expensive filters must be disabled to hit 15Hz
            args = "depth_method:=cpu reg_method:=cpu bilateral_filter:=false edge_aware_filter:=false fps_limit:=15.0"
        elif backend == "opencl":
            # OpenCL reg_method is opencl_cpu
            args = "depth_method:=opencl reg_method:=opencl_cpu fps_limit:=15.0"
        else:
            # CUDA reg_method is cuda
            args = "depth_method:=cuda reg_method:=cuda fps_limit:=15.0"

        return [
            "bash", "-c",
            "source /opt/ros/humble/setup.bash && "
            "source /opt/kinect_ws/install/setup.bash 2>/dev/null || true && "
            f"ros2 launch {WORKSPACE_DIR}/kinect2_bridge_gpu.yaml {args}"
        ]

    # ── Manual Red Line actions ────────────────────────────────────────────

    def _manual_load_image(self) -> None:
        if not CV2_OK:
            QMessageBox.warning(self, "cv2 Missing", "OpenCV (cv2) is required.")
            return
        p, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if p:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                QMessageBox.critical(self, "Error", f"Cannot read: {p}")
                return
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self._canvas.load_image(rgb)

    def _manual_topic_image_cb(self, msg: "ROSImage") -> None:
        """Cache the latest cropped frame and auto-load into canvas if Manual mode is active."""
        if not (CV2_OK and self._bridge):
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self._latest_cropped_rgb = frame.copy()
            
            # Auto-load into manual canvas if the Manual mode is currently selected
            if hasattr(self, '_mode_buttons'):
                btn = self._mode_buttons.checkedButton()
                if btn and btn.property('mode_key') == 'manual':
                    if hasattr(self, '_canvas'):
                        # Thread-safely defer heavy scene manipulation to the Qt event loop
                        self.manual_image_ready.emit(self._latest_cropped_rgb.copy())
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _manual_use_latest_cropped(self) -> None:
        if not CV2_OK:
            QMessageBox.warning(self, "cv2 Missing", "OpenCV (cv2) is required.")
            return
        if getattr(self, '_latest_cropped_rgb', None) is None:
            QMessageBox.warning(
                self,
                "No Cropped Frame",
                "No image has been received yet on /vision/captured_image_color.",
            )
            return
        self.manual_image_ready.emit(self._latest_cropped_rgb.copy())
        self._log.append(
            f'<span style="color:{C["green"]}">[Manual] Loaded latest cropped frame from /vision/captured_image_color.</span>'
        )

    def _manual_save(self) -> None:
        if not CV2_OK:
            return
        img = self._canvas.get_annotated_bgr()
        if img is None:
            QMessageBox.warning(self, "Nothing to Save", "Load an image first.")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Red-Line Image", "red_line_annotated.png", "PNG (*.png)"
        )
        if p:
            cv2.imwrite(p, img)
            QMessageBox.information(self, "Saved", f"Saved to:\n{p}")

    def _manual_publish_ros(self) -> None:
        """Publish the annotated canvas image to /vision/processing_mode/annotated_image.

        Publisher is cached on first call (not recreated on every click) to avoid
        leaking publisher handles into the ROS graph.
        """
        if not (ROS2_OK and CV2_OK):
            QMessageBox.warning(self, "Unavailable", "ROS 2 + cv2 required.")
            return
        img = self._canvas.get_annotated_bgr()
        if img is None:
            QMessageBox.warning(self, "Nothing to Publish", "Draw red lines first.")
            return
        try:
            # Lazily create once; reuse on subsequent clicks
            if not hasattr(self, "_manual_annotated_pub") or self._manual_annotated_pub is None:
                self._manual_annotated_pub = self._ros_node.create_publisher(
                    ROSImage, "/vision/processing_mode/annotated_image", 1
                )
            msg = self._bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self._manual_annotated_pub.publish(msg)
            self._log.append(
                f'<b style="color:{C["green"]}">[Manual] Published to '
                '/vision/processing_mode/annotated_image ✅</b>'
            )
        except Exception as exc:
            self._log.append(f'[Manual] Publish error: {exc}')

    # ── Legacy Launcher ────────────────────────────────────────────────────

    def _launch_legacy(self) -> None:
        btn = self.sender()
        rel = btn.property("script_path")
        full = VISION_WORK / rel
        if not full.exists():
            QMessageBox.warning(
                self, "Not Found",
                f"Script not found:\n{full}\n\nCheck vision_work/ directory."
            )
            return
        subprocess.Popen([sys.executable, str(full)])
        self._log.append(f'[Legacy] Launched: {full.name}')

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, ev) -> None:
        self._stop_all()
        ev.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 11))
    win = VisionPipelineGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
