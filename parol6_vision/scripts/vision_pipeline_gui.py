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

import copy
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
    QFileDialog, QMessageBox, QCheckBox, QLineEdit, QDoubleSpinBox,
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
    from rclpy.qos import (
        qos_profile_sensor_data,
        QoSProfile,
        QoSDurabilityPolicy,
        QoSReliabilityPolicy,
    )
    from rclpy.parameter import Parameter as RclpyParameter
    from nav_msgs.msg import Path as ROSPath
    from sensor_msgs.msg import Image as ROSImage
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger
    from rcl_interfaces.srv import SetParameters
    from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
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
        # Prevent duplicates: iteratively extract match string and kill existing
        match_str = ""
        if len(self._display_cmd) >= 4 and self._display_cmd[0] == "ros2" and self._display_cmd[1] == "run":
            pkg, node = self._display_cmd[2], self._display_cmd[3]
            match_str = f"ros2 run {pkg} {node}"
        elif any("kinect2_bridge" in arg for arg in self._display_cmd):
            match_str = "kinect2_bridge"

        if match_str:
            import time
            pkill_cmd = f"pkill -TERM -f '{match_str}'"
            self.line_out.emit(f"<b style='color:#FF5555'>[KILL] Neutralizing existing '{match_str}' nodes...</b>")
            subprocess.run(pkill_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            time.sleep(0.4) # Wait for term to flush

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
                start_new_session=True,  # CRITICAL: Ensures we can kill the entire process tree later!
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
                # Send SIGINT to the process *group* so the shell and all children (e.g., ROS nodes) catch it
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


class RosLaunchWorker(QThread):
    """Replica of the firmware configurator launch worker for the ROS tab."""
    output_rviz = Signal(str)
    output_gazebo = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(int)

    def __init__(self, script_path: str, args: list[str], env_vars: Optional[dict[str, str]] = None, parent=None):
        super().__init__(parent)
        self._script_path = script_path
        self._args = args
        self._env_vars = env_vars or {}
        self._proc: Optional[subprocess.Popen] = None

    def run(self) -> None:
        cmd = [self._script_path] + self._args
        msg = f"[LAUNCH] $ {' '.join(cmd)}"
        self.output_rviz.emit(msg)
        self.output_gazebo.emit(msg)

        try:
            env = os.environ.copy()
            env.update(self._env_vars)
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
                cwd=str(WORKSPACE_DIR),
                start_new_session=True,
            )
            for line in self._proc.stdout:
                line = line.rstrip()
                lower = line.lower()
                if "ign" in lower or "gazebo" in lower or "/usr/bin/ruby" in lower or "spawn" in lower:
                    self.output_gazebo.emit(line)
                else:
                    self.output_rviz.emit(line)
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as exc:
            msg = f"[LAUNCH] ERROR: {exc}"
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
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass


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
        
        self._roi_mode = False
        self._roi_polygon: list[list[int]] = []

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
            
    def set_roi_mode(self, enabled: bool) -> None:
        """Toggle ROI drawing mode."""
        self._roi_mode = enabled
        if not enabled:
            if len(self._sl_waypoints) >= 3:
                self._roi_polygon = self._sl_waypoints.copy()
            self._sl_waypoints = []
            self._refresh_mask()

    def get_strokes(self) -> list[list[list[int]]]:
        """Return all recorded strokes as a list of point lists for serialisation."""
        return self._all_strokes.copy()
        
    def get_roi_polygon(self) -> list[list[int]]:
        return getattr(self, '_roi_polygon', []).copy()

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
        self._roi_polygon = []

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

        if self._straight_line_mode or self._roi_mode:
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
                    color = (255, 100, 0, 255) if self._roi_mode else (255, 0, 0, 255)
                    cv2.line(self._mask_arr,
                             (prev[0], prev[1]), (ix, iy),
                             color, self._brush, lineType=cv2.LINE_AA)
                    self._refresh_mask()
                self._sl_waypoints.append([ix, iy])
                ev.accept()
                return
            elif ev.button() == Qt.RightButton:
                if self._roi_mode:
                    if len(self._sl_waypoints) >= 3:
                        first = self._sl_waypoints[0]
                        last = self._sl_waypoints[-1]
                        cv2.line(self._mask_arr, (last[0], last[1]), (first[0], first[1]), (255, 100, 0, 255), self._brush, cv2.LINE_AA)
                        self._roi_polygon = self._sl_waypoints.copy()
                        self._refresh_mask()
                    self._sl_waypoints = []
                else:
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
            worker = self._worker
            self._worker = None
            try:
                worker.abort()
            except Exception:
                pass
            self._park_worker(worker)
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
            if self._label != "Live Kinect Camera":
                self._log.append(formatted) # Write to the main "All Nodes" log
            if hasattr(self, '_node_log'):
                self._node_log.append(formatted) # Write to the dedicated tab
                
        self._worker.line_out.connect(_handle_log)
        self._worker.finished.connect(lambda rc, w=self._worker: self._on_finished(w, rc))
        self._worker.start()
        self._set_running()
        if self._on_started:
            self._on_started()

    def _park_worker(self, worker: NodeWorker) -> None:
        host = self.window()
        if hasattr(host, "_park_worker"):
            host._park_worker(worker)
            return
        if not hasattr(self, "_graveyard"):
            self._graveyard = []
        if worker not in self._graveyard:
            self._graveyard.append(worker)
            worker.finished.connect(
                lambda rc, w=worker: self._graveyard.remove(w)
                if w in getattr(self, "_graveyard", []) else None
            )

    def _on_finished(self, worker: NodeWorker, rc: int) -> None:
        if self._worker is worker:
            self._worker = None
        if self._worker is not None and self._worker.isRunning():
            self._set_running()
            return
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
        self.resize(1300, 850)
        self._latest_cropped_rgb: Optional[np.ndarray] = None
        self._crop_set_params_client = None
        self._crop_clear_client = None
        self._moveit_execute_client = None
        self._path_holder_set_source_client = None
        self._inject_path_pub = None  # persistent publisher → /vision/inject_path
        self._crop_futures = []
        self._graveyard: list = []
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
            self._moveit_execute_client = self._ros_node.create_client(Trigger, "/moveit_controller/execute_welding_path")
            # path_holder source switching: set the parameter first, then call the service
            self._path_holder_set_params_client = self._ros_node.create_client(
                SetParameters, "/path_holder/set_parameters"
            )
            self._path_holder_set_source_client = self._ros_node.create_client(
                Trigger, "/path_holder/set_source"
            )
            self._manual_set_params_client = self._ros_node.create_client(SetParameters, "/manual_line/set_parameters")
            self._manual_set_strokes_client = self._ros_node.create_client(Trigger, "/manual_line/set_strokes")
            self._aligner_set_params_client = self._ros_node.create_client(SetParameters, "/manual_line_aligner/set_parameters")
            self._aligner_teach_client = self._ros_node.create_client(Trigger, "/manual_line_aligner/teach_reference")
            self._aligner_set_strokes_client = self._ros_node.create_client(Trigger, "/manual_line_aligner/set_strokes")
            self._manual_teach_client = self._ros_node.create_client(Trigger, "/manual_line_aligner/teach_reference")
            self._manual_cropped_sub = None
            # Persistent publisher → inject_path_node (no subprocess, no DDS race)
            self._inject_path_pub = self._ros_node.create_publisher(
                ROSPath, "/vision/inject_path", 10
            )
            if CV2_OK:
                self._manual_cropped_sub = self._ros_node.create_subscription(
                    ROSImage,
                    "/vision/captured_image_color",
                    self._manual_topic_image_cb,
                    qos_profile_sensor_data,
                )

            # Real-time welding path monitor — updates the Simple dashboard status
            # when an actual path arrives from path_holder (TRANSIENT_LOCAL).
            _path_qos = rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._weld_path_sub = self._ros_node.create_subscription(
                ROSPath,
                "/vision/welding_path",
                self._on_weld_path_received,
                _path_qos,
            )
            # Also subscribe to depth topic to confirm capture
            self._depth_confirm_sub = self._ros_node.create_subscription(
                ROSImage,
                "/vision/captured_image_depth",
                self._on_depth_confirmed,
                rclpy.qos.QoSProfile(
                    depth=1,
                    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
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

        sidebar_tabs = QTabWidget()
        sidebar_tabs.addTab(self._build_simple_sidebar(), "🎯 Simple Workflow")
        sidebar_tabs.addTab(self._build_sidebar(), "🔧 Advanced Nodes")
        splitter.addWidget(sidebar_tabs)
        # Start the real node-liveness poller after the sidebar status box is wired up
        QTimer.singleShot(500, self._start_simple_status_poller)

        tabs = self._build_tabs()
        splitter.addWidget(tabs)

        splitter.setSizes([360, 1040])

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_simple_sidebar(self) -> QWidget:
        """A streamlined 3-button dashboard for production."""
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(16, 24, 16, 16)
        lay.setSpacing(24)

        title = QLabel("Production Workflow")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet(f"color:{C['accent']};")
        lay.addWidget(title)

        desc = QLabel(
            "Quickly run the end-to-end vision pipeline without managing "
            "individual micro-nodes. Ensure the robot and camera are connected."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{C['text2']};")
        lay.addWidget(desc)

        # ── Status feedback box (defined early so button lambdas can reference it) ─
        # This is a forward-declared helper; the widget is added at the end.
        _status_lbl_holder: list = []  # mutable container so inner lambdas can update it

        def _set_status(msg: str, color: str = "#a6e3a1"):
            if _status_lbl_holder:
                lbl = _status_lbl_holder[0]
                lbl.setText(msg)
                lbl.setStyleSheet(
                    f"font-size:12px; font-weight:bold; color:{color};"
                    " padding:4px; border:none;"
                )

        # 0. ROS / MoveIt launch (mini panel)
        grp0 = QGroupBox("Step 0: ROS / MoveIt")
        glay0 = QVBoxLayout(grp0)
        glay0.setSpacing(6)

        mode_r0 = QHBoxLayout()
        mode_r0.addWidget(QLabel("Method:"))
        self._simple_ros_method = QComboBox()
        self._simple_ros_method.addItem("M2 — Gazebo + MoveIt (Simulated)",            "launch_moveit_with_gazebo.sh")
        self._simple_ros_method.addItem("M3 — Fake HW (Standalone RViz)",               "launch_moveit_fake.sh")
        self._simple_ros_method.addItem("M5 — Real HW (Tested Single-Motor Legacy)",    "launch_moveit_real_hw_tested_single_motor.sh")
        self._simple_ros_method.setCurrentIndex(1)  # default to M3 (safest)
        mode_r0.addWidget(self._simple_ros_method, 1)
        glay0.addLayout(mode_r0)

        r0_btns = QHBoxLayout()
        r0_launch_btn = QPushButton("RUN — Launch MoveIt")
        r0_launch_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:5px; padding:5px 10px;"
        )
        def _simple_ros_launch():
            _set_status(">> Launching MoveIt...", "#f9e2af")
            # Mirror _ros_launch_toggle but using the mini sidebar dropdown
            script = self._simple_ros_method.currentData()
            for i in range(self._ros_method.count()):
                if self._ros_method.itemData(i) == script:
                    self._ros_method.setCurrentIndex(i)
                    break
            self._ros_launch_toggle()
            # Real confirmation will arrive via node liveness poller (no fake timer)
        r0_launch_btn.clicked.connect(_simple_ros_launch)
        r0_btns.addWidget(r0_launch_btn)

        r0_kill_btn = QPushButton("STOP — Kill All")
        r0_kill_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:5px; padding:5px 10px;"
        )
        r0_kill_btn.clicked.connect(self._ros_kill_all_nodes)
        r0_btns.addWidget(r0_kill_btn)
        glay0.addLayout(r0_btns)

        # Link to open the full ROS tab
        ros_link = QLabel("<a href='#ros_tab' style='color:#cba6f7;'>→ Open Full ROS Launch Tab</a>")
        ros_link.setTextFormat(Qt.RichText)
        ros_link.setStyleSheet("font-size:10px;")
        def _goto_ros_tab():
            # Jump to the ROS launch tab in the main panel
            if hasattr(self, '_main_tabs') and hasattr(self, '_ros_launch_tab'):
                idx = self._main_tabs.indexOf(self._ros_launch_tab)
                if idx >= 0:
                    self._main_tabs.setCurrentIndex(idx)
        ros_link.linkActivated.connect(lambda _: _goto_ros_tab())
        glay0.addWidget(ros_link)
        lay.addWidget(grp0)

        # 1. Start Pipeline
        grp1 = QGroupBox("Step 1: Initialization")
        glay1 = QVBoxLayout(grp1)
        btn1 = QPushButton("START — Full Pipeline")
        btn1.setFixedHeight(45)
        btn1.setStyleSheet(f"background:#89b4fa; color:#11111b; font-weight:bold; font-size:14px; border-radius:6px;")
        def _start_pipeline():
            _set_status(">> Starting pipeline nodes...", "#89b4fa")
            self._ros_log.append("<b style='color:#89b4fa'>[Simple Run] Starting Pipeline...</b>")
            self._btn_live_cam._start()
            self._btn_capture._start()
            self._btn_crop_node._start()
            self._btn_optimizer._start()
            self._btn_depth._start()
            self._btn_pathgen._start()
            # Real confirmation arrives via node liveness poller (no fake timer)
        btn1.clicked.connect(_start_pipeline)
        glay1.addWidget(btn1)
        lay.addWidget(grp1)

        # 2. Capture & Map
        grp2 = QGroupBox("Step 2: Scanning")
        glay2 = QVBoxLayout(grp2)
        glay2.setSpacing(10)  # Add space to prevent overlap
        glay2.setContentsMargins(10, 15, 10, 10)

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._simple_mode_combo = QComboBox()
        self._simple_mode_combo.addItem("Manual Red Line", "manual")
        self._simple_mode_combo.addItem("YOLO Detection", "yolo")
        self._simple_mode_combo.addItem("Color Threshold", "color")
        mode_row.addWidget(self._simple_mode_combo, 1)
        glay2.addLayout(mode_row)

        btn2 = QPushButton("SCAN — Capture && Process")
        btn2.setFixedHeight(45)
        btn2.setStyleSheet("background:#a6e3a1; color:#11111b; font-weight:bold; font-size:14px; border-radius:6px;")
        def _capture_and_map():
            _set_status("[CAP] Capturing frame...", "#89dceb")
            self._ros_log.append("<b style='color:#a6e3a1'>[Simple Run] Requesting Frame Capture...</b>")
            if not self._ros_node:
                self._ros_log.append("ROS node offline.")
                _set_status("❌ ROS node offline.", "#f38ba8")
                return
            pub = self._ros_node.create_publisher(Empty, "/vision/capture_trigger", 10)
            pub.publish(Empty())
            # Real confirmation arrives via _on_depth_confirmed ROS callback (no fake timer)
        btn2.clicked.connect(_capture_and_map)
        glay2.addWidget(btn2)

        # Dynamic hint based on selected mode
        _mode_hints = {
            "manual": "<i>Capture, then switch to the <b>Manual Red Line</b> tab, draw your weld path, and click <b>📤 Send Strokes</b>.</i>",
            "yolo":   "<i>Capture — YOLO will auto-detect objects and publish weld lines. Check the <b>YOLO Debug</b> preview.</i>",
            "color":  "<i>Capture — the Color Threshold node will segment the red weld line automatically from the image.</i>",
        }
        info2 = QLabel(_mode_hints["manual"])
        info2.setWordWrap(True)
        info2.setStyleSheet("font-size:10px; color:#6c7086;")
        glay2.addWidget(info2)

        def _update_hint(idx):
            mode_key = self._simple_mode_combo.itemData(idx)
            info2.setText(_mode_hints.get(mode_key, ""))
        self._simple_mode_combo.currentIndexChanged.connect(_update_hint)

        lay.addWidget(grp2)

        # Step 3. Execute
        grp3 = QGroupBox("Step 3: Execution")
        glay3 = QVBoxLayout(grp3)
        
        # Reachability Safety Toggle
        safety_cb = QCheckBox("Enforce Reachable Workspace (Clamp)")
        safety_cb.setToolTip("If enabled, waypoints outside the radial/box limits will be shifted to the nearest reachable point.")
        def _set_safety(state):
            val = (state == 2) # 2 is checked in PySide6
            self.get_logger().info(f"Setting enforce_reachable_test_path to {val}")
            self._set_ros_param("moveit_controller", "enforce_reachable_test_path", val)
        safety_cb.stateChanged.connect(_set_safety)
        glay3.addWidget(safety_cb)

        btn3 = QPushButton("WELD — Execute Path")
        btn3.setFixedHeight(45)
        btn3.setStyleSheet(f"background:#f38ba8; color:#11111b; font-weight:bold; font-size:14px; border-radius:6px;")
        def _execute():
            _set_status(">> Executing weld path...", "#fab387")
            self._btn_moveit._start() # Run moveit node if not running
            QTimer.singleShot(1500, self._send_path_to_moveit) # trigger after a delay
        btn3.clicked.connect(_execute)
        glay3.addWidget(btn3)
        lay.addWidget(grp3)

        # ── Status feedback box ────────────────────────────────────────────
        status_box = QFrame()
        status_box.setFrameShape(QFrame.StyledPanel)
        status_box.setStyleSheet(
            "border:1px solid #45475a; border-radius:8px; "
            f"background:{C['panel']}; padding:2px;"
        )
        sb_lay = QVBoxLayout(status_box)
        sb_lay.setContentsMargins(10, 8, 10, 8)
        sb_lay.setSpacing(2)

        status_title = QLabel("Pipeline Status")
        status_title.setStyleSheet("font-size:10px; font-weight:bold; color:#6c7086; border:none;")
        sb_lay.addWidget(status_title)

        self._simple_status_label = QLabel("⏸  Idle — press Step 0 to begin.")
        self._simple_status_label.setWordWrap(True)
        self._simple_status_label.setTextFormat(Qt.RichText)
        self._simple_status_label.setStyleSheet(
            "font-size:12px; font-weight:bold; color:#a6e3a1;"
            " padding:4px; border:none;"
        )
        sb_lay.addWidget(self._simple_status_label)
        lay.addWidget(status_box)

        def _set_status(msg: str, color: str = "#a6e3a1"):
            self._simple_status_label.setText(msg)
            self._simple_status_label.setStyleSheet(
                f"font-size:12px; font-weight:bold; color:{color};"
                " padding:4px; border:none;"
            )
        # Expose so other methods can push status updates
        self._simple_set_status = _set_status

        lay.addStretch()
        return tab

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
            ("🎯  Auto-Align Mode",     "align"),
        ]
        for label, key in modes:
            rb = QRadioButton(label)
            rb.setProperty("mode_key", key)
            self._mode_buttons.addButton(rb)
            mg_lay.addWidget(rb)
        self._mode_buttons.buttons()[0].setChecked(True)

        # Switch to Manual Red Line or Auto-Align tab when those modes are selected
        def _on_mode_selected(btn):
            key = btn.property('mode_key')
            if key == 'manual':
                if hasattr(self, '_main_tabs') and hasattr(self, '_manual_tab_index'):
                    self._main_tabs.setCurrentIndex(self._manual_tab_index)
                if CV2_OK and getattr(self, '_latest_cropped_rgb', None) is not None:
                    self.manual_image_ready.emit(self._latest_cropped_rgb.copy())
            elif key == 'align':
                if hasattr(self, '_main_tabs') and hasattr(self, '_align_tab_index'):
                    self._main_tabs.setCurrentIndex(self._align_tab_index)

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

        self._btn_path_holder = NodeButton(
            "Path Holder",
            lambda: ["ros2", "run", "parol6_vision", "path_holder"],
            self._log, "#cba6f7",
        )
        pg_lay.addWidget(self._btn_path_holder)

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
        
        save_prof_btn = QPushButton("💾  Save Profile As...")
        save_prof_btn.clicked.connect(self._manual_save_profile)
        save_prof_btn.setStyleSheet(f"background:{C['panel']}; border:1px solid {C['border']}; border-radius:4px; padding:4px;")
        row1.addWidget(save_prof_btn)

        load_prof_btn = QPushButton("📂  Load Profile...")
        load_prof_btn.clicked.connect(self._manual_load_profile)
        load_prof_btn.setStyleSheet(f"background:{C['panel']}; border:1px solid {C['border']}; border-radius:4px; padding:4px;")
        row1.addWidget(load_prof_btn)

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

        # ── Tab 2b: Auto-Align (manual_line_aligner) ───────────────────────────
        align_tab = QWidget()
        at_lay = QVBoxLayout(align_tab)
        at_lay.setContentsMargins(8, 8, 8, 8)

        align_hint = QLabel(
            "<b>🎯 Auto-Align Workflow:</b>  "
            "1) Load the latest cropped frame.  "
            "2) Check <b>📐 Straight-line</b> and draw your <b>weld strokes</b> (red).  "
            "3) Check <b>🔲 Draw ROI Boundary</b> and click a polygon around the part (orange), right-click to close.  "
            "4) Click <b>📤 Teach &amp; Send</b> to teach the node — it will auto-align on every future frame."
        )
        align_hint.setTextFormat(Qt.RichText)
        align_hint.setWordWrap(True)
        align_hint.setStyleSheet(
            f"background:#1e1a2e; border:1px solid #cba6f7; border-radius:6px;"
            f" color:{C['text']}; font-size:11px; padding:6px 10px; margin-bottom:4px;"
        )
        at_lay.addWidget(align_hint)

        # Toolbar Row A1
        a_row1 = QHBoxLayout()
        a_load_btn = QPushButton("📥  Use Latest Cropped Frame")
        a_load_btn.clicked.connect(self._align_use_latest_cropped)
        a_row1.addWidget(a_load_btn)
        a_row1.addStretch()
        at_lay.addLayout(a_row1)

        # Toolbar Row A2: Drawing Tools
        a_row2 = QHBoxLayout()
        a_row2.addWidget(QLabel("Brush (px):"))
        self._align_brush_spin = QSpinBox()
        self._align_brush_spin.setRange(1, 80)
        self._align_brush_spin.setValue(5)
        a_row2.addWidget(self._align_brush_spin)
        a_row2.addSpacing(8)

        self._align_straight_check = QCheckBox("📐  Straight-line strokes")
        self._align_straight_check.stateChanged.connect(
            lambda s: self._align_canvas.set_straight_line_mode(s == Qt.Checked)
            if getattr(self, '_align_canvas', None) else None
        )
        a_row2.addWidget(self._align_straight_check)
        a_row2.addSpacing(8)

        self._align_roi_check = QCheckBox("🔲  Draw ROI Boundary (orange)")
        self._align_roi_check.setToolTip("Click polygon vertices around the part, right-click to close.")
        self._align_roi_check.stateChanged.connect(self._on_align_roi_toggled)
        a_row2.addWidget(self._align_roi_check)
        a_row2.addSpacing(8)

        a_clear_btn = QPushButton("🗑  Clear")
        a_clear_btn.clicked.connect(
            lambda: self._align_canvas.clear_mask() if getattr(self, '_align_canvas', None) and CV2_OK else None
        )
        a_row2.addWidget(a_clear_btn)

        teach_btn = QPushButton("📤  Teach & Send")
        teach_btn.setToolTip("Send strokes + ROI to manual_line_aligner (teach_reference service).")
        teach_btn.setStyleSheet(f"background:#a6e3a1; color:{C['bg']}; font-weight:bold; border:none; border-radius:5px; padding:4px 8px;")
        teach_btn.clicked.connect(self._align_teach_send)
        a_row2.addWidget(teach_btn)

        a_reset_btn = QPushButton("🔄  Reset")
        a_reset_btn.setStyleSheet(f"background:{C['red']}; color:{C['bg']}; font-weight:bold; border:none; border-radius:5px; padding:4px 8px;")
        a_reset_btn.clicked.connect(self._align_reset)
        a_row2.addWidget(a_reset_btn)
        a_row2.addStretch()
        at_lay.addLayout(a_row2)

        align_canvas_hint = QLabel("Strokes = red weld path.  ROI boundary = orange polygon (right-click to close).")
        align_canvas_hint.setStyleSheet(f"color:{C['text2']}; font-size:10px;")
        at_lay.addWidget(align_canvas_hint)

        # Separate canvas for Auto-Align (doesn't share state with Manual)
        self._align_canvas = ManualCanvas()
        self._align_brush_spin.valueChanged.connect(
            lambda v: self._align_canvas.set_brush(v) if getattr(self, '_align_canvas', None) else None
        )
        at_lay.addWidget(self._align_canvas, stretch=1)

        self._align_log = QTextEdit()
        self._align_log.setReadOnly(True)
        self._align_log.setFixedHeight(90)
        self._align_log.setFont(QFont("Monospace", 9))
        self._align_log.setStyleSheet(f"background:{C['bg']}; color:{C['text']}; border:1px solid {C['border']}; border-radius:4px;")
        at_lay.addWidget(self._align_log)

        tabs.addTab(align_tab, "🎯  Auto-Align")
        self._align_tab_index = tabs.indexOf(align_tab)

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
        self._ros_method.addItem("Method 2: Gazebo AND MoveIt (Simulated)",                     "launch_moveit_with_gazebo.sh")
        self._ros_method.addItem("Method 3: MoveIt Fake (Standalone RViz)",                     "launch_moveit_fake.sh")
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
        ros_kill_btn.clicked.connect(self._ros_kill_all_nodes)
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

        self._ros_worker: Optional[RosLaunchWorker] = None
        self._ros_test_worker: Optional[RosLaunchWorker] = None
        self._ros_launch_env: dict[str, str] = {}
        self._launchers_dir = str(WORKSPACE_DIR / "scripts" / "launchers")
        tabs.addTab(ros_tab, "🚀  ROS Launch")

        # ── Tab 4: Crop Image ─────────────────────────────────────────────
        crop_tab = self._build_crop_tab()
        tabs.addTab(crop_tab, "✂️  Crop Image")

        # ── Tab 4b: Camera Calibration ─────────────────────────────────────
        cal_tab = self._build_calibration_tab()
        tabs.addTab(cal_tab, "📷  Cam Calibrate")

        # ── Tab 5: Settings ───────────────────────────────────────────────
        if ROS2_OK:
            settings_tab = self._build_settings_tab()
            tabs.addTab(settings_tab, "⚙️  ROS Parameters")

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

    # ── Settings Tab ──────────────────────────────────────────────────────────

    def _build_settings_tab(self) -> QWidget:
        """
        ⚙️ Settings Tab
        ─────────────────
        Dynamic ROS parameter adjustment for live nodes.
        """
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        def _add_param(parent_lay, name, default, node_name, label_text, min_val, max_val, step=1, double=False):
            row = QHBoxLayout()
            label = QLabel(f"<b>{label_text}</b>  <span style='color:{C['text2']}'>({name})</span>")
            label.setMinimumWidth(250)
            row.addWidget(label)
            
            spin = QDoubleSpinBox() if double else QSpinBox()
            if double:
                spin.setDecimals(3)
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setMinimumWidth(120)
            row.addWidget(spin)
            
            apply_btn = QPushButton("Apply")
            
            def _apply():
                val = spin.value()
                req = SetParameters.Request()
                p = Parameter()
                p.name = name
                pv = ParameterValue()
                if double:
                    pv.type = ParameterType.PARAMETER_DOUBLE
                    pv.double_value = float(val)
                else:
                    pv.type = ParameterType.PARAMETER_INTEGER
                    pv.integer_value = int(val)
                p.value = pv
                req.parameters = [p]
                
                cli = self._ros_node.create_client(SetParameters, f'/{node_name}/set_parameters')
                if not cli.wait_for_service(timeout_sec=1.0):
                    self._ros_log.append(f"<b style='color:#ff5555'>[Settings] Service /{node_name}/set_parameters offline</b>")
                    return
                # Call async and forget — if it's up, it'll apply
                cli.call_async(req)
                self._ros_log.append(f"<b style='color:#55ff55'>[Settings] Applied {name}={val} to {node_name}</b>")
                
            apply_btn.clicked.connect(_apply)
            row.addWidget(apply_btn)
            row.addStretch()
            parent_lay.addLayout(row)

        pg_grp = QGroupBox("Path Generator (/path_generator)")
        pg_lay = QVBoxLayout(pg_grp)
        _add_param(pg_lay, "max_waypoints", 80, "path_generator", "Max Waypoints Cap", 10, 1000, 10)
        _add_param(pg_lay, "waypoint_spacing", 0.005, "path_generator", "Waypoint Spacing (m)", 0.001, 0.05, 0.001, True)
        lay.addWidget(pg_grp)

        mc_grp = QGroupBox("MoveIt Controller (/moveit_controller)")
        mc_lay = QVBoxLayout(mc_grp)
        _add_param(mc_lay, "approach_distance", 0.15, "moveit_controller", "Approach Distance (m)", 0.01, 0.5, 0.01, True)
        _add_param(mc_lay, "weld_velocity", 0.01, "moveit_controller", "Weld Velocity (m/s)", 0.001, 0.1, 0.001, True)
        _add_param(mc_lay, "joint_waypoint_fallback_count", 8, "moveit_controller", "Fallback Samples", 2, 50, 1)
        
        # Reachability Safety Toggle
        safety_cb = QCheckBox("Enforce Reachable Workspace (Clamp)")
        safety_cb.setToolTip("If enabled, waypoints outside the radial/box limits will be shifted to the nearest reachable point.")
        def _set_safety(state):
            val = safety_cb.isChecked()
            self._set_ros_param("moveit_controller", "enforce_reachable_test_path", val)
        safety_cb.stateChanged.connect(_set_safety)
        mc_lay.addWidget(safety_cb)

        lay.addWidget(mc_grp)

        # ── Path Offset (welding correction) ──────────────────────────────────
        offset_grp = QGroupBox("🎯 Path Offset — Welding Correction (/moveit_controller)")
        offset_grp.setStyleSheet(f"QGroupBox {{ border:1px solid #fab387; border-radius:6px; margin-top:6px; }}")
        offset_lay = QVBoxLayout(offset_grp)

        offset_hint = QLabel(
            "Apply a <b>static XYZ offset</b> (mm) to every waypoint before execution. "
            "Useful when the detected path needs fine correction for weld bead placement "
            "without re-running vision. "
            "<span style='color:#fab387;'>⚠️ Values are in <b>mm</b> in the GUI but stored as <b>meters</b> in ROS.</span>"
        )
        offset_hint.setWordWrap(True)
        offset_hint.setTextFormat(Qt.RichText)
        offset_hint.setStyleSheet(f"font-size:11px; color:{C['text2']}; border:none;")
        offset_lay.addWidget(offset_hint)

        spin_row = QHBoxLayout()
        offset_spins = {}
        for axis, color in [("X", "#89b4fa"), ("Y", "#a6e3a1"), ("Z", "#fab387")]:
            col = QVBoxLayout()
            lbl = QLabel(f"<b style='color:{color}'>{axis} offset (mm)</b>")
            lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(lbl)
            sp = QDoubleSpinBox()
            sp.setRange(-50.0, 50.0)
            sp.setSingleStep(0.5)
            sp.setDecimals(1)
            sp.setValue(0.0)
            sp.setMinimumWidth(90)
            sp.setAlignment(Qt.AlignCenter)
            col.addWidget(sp)
            spin_row.addLayout(col)
            offset_spins[axis] = sp
        offset_lay.addLayout(spin_row)

        def _apply_offset():
            req = SetParameters.Request()
            for axis, param_name in [("X", "path_offset_x"), ("Y", "path_offset_y"), ("Z", "path_offset_z")]:
                val_m = offset_spins[axis].value() / 1000.0  # mm → m
                p = Parameter()
                p.name = param_name
                pv = ParameterValue()
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = val_m
                p.value = pv
                req.parameters.append(p)
            cli = self._ros_node.create_client(SetParameters, '/moveit_controller/set_parameters')
            if not cli.wait_for_service(timeout_sec=1.0):
                self._ros_log.append("<b style='color:#ff5555'>[Offset] moveit_controller offline</b>")
                return
            cli.call_async(req)
            dx = offset_spins["X"].value()
            dy = offset_spins["Y"].value()
            dz = offset_spins["Z"].value()
            self._ros_log.append(
                f"<b style='color:#fab387'>[Offset] Applied: X={dx:+.1f}mm  Y={dy:+.1f}mm  Z={dz:+.1f}mm</b>"
            )

        apply_all_btn = QPushButton("✅  Apply All 3 Offsets")
        apply_all_btn.setStyleSheet(
            f"background:#fab387; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        apply_all_btn.clicked.connect(_apply_offset)
        reset_offset_btn = QPushButton("↩  Reset to Zero")
        reset_offset_btn.clicked.connect(lambda: [sp.setValue(0.0) for sp in offset_spins.values()] or _apply_offset())
        btn_row = QHBoxLayout()
        btn_row.addWidget(apply_all_btn)
        btn_row.addWidget(reset_offset_btn)
        btn_row.addStretch()
        offset_lay.addLayout(btn_row)
        lay.addWidget(offset_grp)

        lay.addStretch()
        return tab




    # ── Camera Calibration Tab ────────────────────────────────────────────────

    def _build_calibration_tab(self) -> QWidget:
        """
        📷 Camera Calibration Tab
        ─────────────────────────
        Two-panel tab:
          Top    : Current calibration status (loaded from ~/.parol6/camera_tf.yaml)
                   with Enforce (dynamic override) / Stop buttons.
          Bottom : ArUco auto-calibration launcher — runs aruco_detector + eye_to_hand_calibrator,
                   shows progress, and saves the new result when done.
        """
        from PySide6.QtWidgets import QProgressBar, QScrollArea

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # ── Title bar ─────────────────────────────────────────────────────────
        title = QLabel("📷  Camera Auto-Calibration (ArUco)")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        lay.addWidget(title)

        hint = QLabel(
            "📌 Fix the camera frame (<b>base_link → kinect2_link</b>) without editing any node. "
            "<b>Step 1</b>: <i>Enforce</i> the already-calibrated ArUco result immediately (no restart). "
            "<b>Step 2</b>: Re-run ArUco calibration to refresh the frame if the camera moved."
        )
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"background:{C['panel']}; border:1px solid #cba6f7; border-radius:6px;"
            f" color:{C['text2']}; font-size:11px; padding:8px 10px;"
        )
        lay.addWidget(hint)

        # ─────────────────────────────────────────────────────────────────────
        # GROUP 1 — CURRENT FRAME
        # ─────────────────────────────────────────────────────────────────────
        frame_grp = QGroupBox("💾 Current Camera Frame  (≈ ~/.parol6/camera_tf.yaml)")
        frame_lay = QVBoxLayout(frame_grp)

        self._cal_source_lbl = QLabel("Source: loading...")
        self._cal_source_lbl.setStyleSheet(f"color:{C['text2']}; font-size:10px; font-style:italic;")
        frame_lay.addWidget(self._cal_source_lbl)

        # Values grid: x y z  qx qy qz qw  child_frame
        vals_row = QHBoxLayout()
        self._cal_val_labels: dict[str, QLabel] = {}
        for key, colour in [('x','#89b4fa'),('y','#89b4fa'),('z','#89b4fa'),
                             ('qx','#a6e3a1'),('qy','#a6e3a1'),('qz','#a6e3a1'),('qw','#a6e3a1')]:
            col = QVBoxLayout()
            col.setSpacing(2)
            nm = QLabel(f"<b style='color:{colour}'>{key}</b>")
            nm.setAlignment(Qt.AlignCenter)
            col.addWidget(nm)
            vl = QLabel("—")
            vl.setAlignment(Qt.AlignCenter)
            vl.setStyleSheet(f"color:{colour}; font-family:Monospace; font-size:12px;")
            col.addWidget(vl)
            vals_row.addLayout(col)
            self._cal_val_labels[key] = vl

        child_col = QVBoxLayout(); child_col.setSpacing(2)
        child_col.addWidget(QLabel("<b>child frame</b>", alignment=Qt.AlignCenter))
        self._cal_child_lbl = QLabel("—")
        self._cal_child_lbl.setAlignment(Qt.AlignCenter)
        self._cal_child_lbl.setStyleSheet("color:#fab387; font-family:Monospace; font-size:12px;")
        child_col.addWidget(self._cal_child_lbl)
        vals_row.addLayout(child_col)
        frame_lay.addLayout(vals_row)

        # Enforce / stop buttons
        enforce_row = QHBoxLayout()
        self._cal_enforce_btn = QPushButton("▶️  Enforce This Frame (Live Dynamic Override)")
        self._cal_enforce_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._cal_enforce_btn.setToolTip(
            "Starts camera_tf_enforcer node — publishes the calibrated camera frame\n"
            "at 100 Hz, overriding the static TF from live_pipeline WITHOUT any restart."
        )
        self._cal_enforce_btn.clicked.connect(self._cal_enforce_start)
        enforce_row.addWidget(self._cal_enforce_btn)

        self._cal_stop_enforce_btn = QPushButton("⏹  Stop Override")
        self._cal_stop_enforce_btn.setEnabled(False)
        self._cal_stop_enforce_btn.clicked.connect(self._cal_enforce_stop)
        enforce_row.addWidget(self._cal_stop_enforce_btn)

        self._cal_enforce_status = QLabel("⚠️  Not active — pipeline uses static TF from launch file")
        self._cal_enforce_status.setStyleSheet(f"color:{C['yellow']}; font-size:11px;")
        enforce_row.addWidget(self._cal_enforce_status)
        enforce_row.addStretch()
        frame_lay.addLayout(enforce_row)
        lay.addWidget(frame_grp)

        self._cal_enforcer_worker = None

        # ─────────────────────────────────────────────────────────────────────
        # GROUP 2 — MANUAL FRAME ENTRY
        # ─────────────────────────────────────────────────────────────────────
        manual_grp = QGroupBox("✏️  Manual Frame Entry — paste ArUco result directly")
        manual_grp.setStyleSheet("QGroupBox { border:1px solid #fab387; border-radius:6px; margin-top:6px; }")
        man_lay = QVBoxLayout(manual_grp)

        man_hint = QLabel(
            "Enter the transform your teammate's calibration produced (<b>or any known value</b>). "
            "This writes <code>~/.parol6/camera_tf.yaml</code> immediately and then enforces it live."
        )
        man_hint.setWordWrap(True)
        man_hint.setTextFormat(Qt.RichText)
        man_hint.setStyleSheet(f"font-size:11px; color:{C['text2']}; border:none;")
        man_lay.addWidget(man_hint)

        # Child frame selector
        frame_sel_row = QHBoxLayout()
        frame_sel_row.addWidget(QLabel("base_link  →  child frame:"))
        self._man_child_combo = QComboBox()
        self._man_child_combo.addItem("kinect2_link  (ArUco calibration result)", "kinect2_link")
        self._man_child_combo.addItem("kinect2       (original launch-file default)", "kinect2")
        self._man_child_combo.setMinimumWidth(280)
        frame_sel_row.addWidget(self._man_child_combo)
        frame_sel_row.addStretch()
        man_lay.addLayout(frame_sel_row)

        # Translation row
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("<b style='color:#89b4fa'>Translation (metres):</b>", objectName="tl"))
        t_row.itemAt(0).widget().setTextFormat(Qt.RichText)
        self._man_x = QDoubleSpinBox(); self._man_x.setRange(-10, 10); self._man_x.setDecimals(4); self._man_x.setFixedWidth(90)
        self._man_y = QDoubleSpinBox(); self._man_y.setRange(-10, 10); self._man_y.setDecimals(4); self._man_y.setFixedWidth(90)
        self._man_z = QDoubleSpinBox(); self._man_z.setRange(-10, 10); self._man_z.setDecimals(4); self._man_z.setFixedWidth(90)
        # Pre-fill with ArUco-calibrated values from calibration_setup.launch.py
        self._man_x.setValue(0.5550); self._man_y.setValue(0.1777); self._man_z.setValue(1.0016)
        for lbl, w in [("X:", self._man_x), ("Y:", self._man_y), ("Z:", self._man_z)]:
            t_row.addWidget(QLabel(lbl)); t_row.addWidget(w)
        t_row.addStretch()
        man_lay.addLayout(t_row)

        # Quaternion row
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("<b style='color:#a6e3a1'>Quaternion (qx qy qz qw):</b>"))
        q_row.itemAt(0).widget().setTextFormat(Qt.RichText)
        self._man_qx = QDoubleSpinBox(); self._man_qx.setRange(-1, 1); self._man_qx.setDecimals(4); self._man_qx.setFixedWidth(90)
        self._man_qy = QDoubleSpinBox(); self._man_qy.setRange(-1, 1); self._man_qy.setDecimals(4); self._man_qy.setFixedWidth(90)
        self._man_qz = QDoubleSpinBox(); self._man_qz.setRange(-1, 1); self._man_qz.setDecimals(4); self._man_qz.setFixedWidth(90)
        self._man_qw = QDoubleSpinBox(); self._man_qw.setRange(-1, 1); self._man_qw.setDecimals(4); self._man_qw.setFixedWidth(90)
        # Pre-fill with ArUco-calibrated quaternion
        self._man_qx.setValue(0.7078); self._man_qy.setValue(0.7058)
        self._man_qz.setValue(0.0269); self._man_qw.setValue(0.0123)
        for lbl, w in [("qx:", self._man_qx), ("qy:", self._man_qy), ("qz:", self._man_qz), ("qw:", self._man_qw)]:
            q_row.addWidget(QLabel(lbl)); q_row.addWidget(w)
        q_row.addStretch()
        man_lay.addLayout(q_row)

        # Buttons row
        man_btn_row = QHBoxLayout()
        man_save_btn = QPushButton("💾  Save & Enforce Now")
        man_save_btn.setStyleSheet(
            f"background:#fab387; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        man_save_btn.setToolTip(
            "Writes the values above to ~/.parol6/camera_tf.yaml\n"
            "then starts camera_tf_enforcer to apply them live."
        )
        man_save_btn.clicked.connect(self._man_save_and_enforce)
        man_btn_row.addWidget(man_save_btn)

        man_load_btn = QPushButton("🔄  Load from Current File")
        man_load_btn.setToolTip("Populate the fields above from the currently saved camera_tf.yaml.")
        man_load_btn.clicked.connect(self._man_load_from_yaml)
        man_btn_row.addWidget(man_load_btn)

        man_btn_row.addStretch()
        self._man_status_lbl = QLabel("")
        self._man_status_lbl.setStyleSheet(f"color:{C['green']}; font-size:11px;")
        man_btn_row.addWidget(self._man_status_lbl)
        man_lay.addLayout(man_btn_row)

        lay.addWidget(manual_grp)

        # ─────────────────────────────────────────────────────────────────────
        # GROUP 3 — ARUCO AUTO-CALIBRATION
        # ─────────────────────────────────────────────────────────────────────
        aruco_grp = QGroupBox("🎯  ArUco Auto-Calibration (re-run to refresh camera frame)")

        aruco_grp.setStyleSheet("QGroupBox { border:1px solid #89b4fa; border-radius:6px; margin-top:6px; }")
        aruco_lay = QVBoxLayout(aruco_grp)

        # Marker & samples config row
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Marker ID:"))
        self._aruco_id_spin = QSpinBox()
        self._aruco_id_spin.setRange(0, 999); self._aruco_id_spin.setValue(6); self._aruco_id_spin.setFixedWidth(60)
        cfg_row.addWidget(self._aruco_id_spin)

        cfg_row.addSpacing(10); cfg_row.addWidget(QLabel("Size (mm):"))
        self._aruco_size_spin = QDoubleSpinBox()
        self._aruco_size_spin.setRange(1.0, 500.0); self._aruco_size_spin.setDecimals(2)
        self._aruco_size_spin.setValue(45.75); self._aruco_size_spin.setFixedWidth(80)
        cfg_row.addWidget(self._aruco_size_spin)

        cfg_row.addSpacing(10); cfg_row.addWidget(QLabel("Samples:"))
        self._aruco_samples_spin = QSpinBox()
        self._aruco_samples_spin.setRange(5, 200); self._aruco_samples_spin.setValue(20); self._aruco_samples_spin.setFixedWidth(60)
        cfg_row.addWidget(self._aruco_samples_spin)
        cfg_row.addStretch()
        aruco_lay.addLayout(cfg_row)

        # Marker physical position row
        mpos_row = QHBoxLayout()
        mpos_row.addWidget(QLabel("Known marker pos in base_link (m):"))
        self._aruco_mx = QDoubleSpinBox(); self._aruco_mx.setRange(-5,5); self._aruco_mx.setDecimals(4); self._aruco_mx.setValue(0.623); self._aruco_mx.setFixedWidth(80)
        self._aruco_my = QDoubleSpinBox(); self._aruco_my.setRange(-5,5); self._aruco_my.setDecimals(4); self._aruco_my.setValue(0.080); self._aruco_my.setFixedWidth(80)
        self._aruco_mz = QDoubleSpinBox(); self._aruco_mz.setRange(-5,5); self._aruco_mz.setDecimals(4); self._aruco_mz.setValue(0.234); self._aruco_mz.setFixedWidth(80)
        for ax, w in [('X', self._aruco_mx), ('Y', self._aruco_my), ('Z', self._aruco_mz)]:
            mpos_row.addWidget(QLabel(f"{ax}:"))
            mpos_row.addWidget(w)
        mpos_row.addStretch()
        aruco_lay.addLayout(mpos_row)

        # Progress bar
        self._aruco_progress = QProgressBar()
        self._aruco_progress.setRange(0, 20); self._aruco_progress.setValue(0)
        self._aruco_progress.setFormat("%v / %m samples")
        self._aruco_progress.setStyleSheet(
            f"QProgressBar {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:4px; "
            f"color:{C['text']}; text-align:center; }}"
            f"QProgressBar::chunk {{ background:#89b4fa; border-radius:4px; }}"
        )
        aruco_lay.addWidget(self._aruco_progress)

        # Run / Stop / Save buttons
        btn_row = QHBoxLayout()
        self._aruco_run_btn = QPushButton("▶️  Run ArUco Calibration")
        self._aruco_run_btn.setStyleSheet(
            f"background:#89b4fa; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._aruco_run_btn.clicked.connect(self._aruco_start)
        btn_row.addWidget(self._aruco_run_btn)

        self._aruco_stop_btn = QPushButton("⏹  Stop")
        self._aruco_stop_btn.setEnabled(False)
        self._aruco_stop_btn.clicked.connect(self._aruco_stop)
        btn_row.addWidget(self._aruco_stop_btn)

        self._aruco_enforce_btn = QPushButton("✅  Save & Enforce New Frame")
        self._aruco_enforce_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._aruco_enforce_btn.setEnabled(False)
        self._aruco_enforce_btn.clicked.connect(self._aruco_save_and_enforce)
        btn_row.addWidget(self._aruco_enforce_btn)
        btn_row.addStretch()
        aruco_lay.addLayout(btn_row)

        # Log area
        self._aruco_log = QTextEdit()
        self._aruco_log.setReadOnly(True)
        self._aruco_log.setFixedHeight(160)
        self._aruco_log.setFont(QFont("Monospace", 9))
        self._aruco_log.setStyleSheet("background:#11111b; color:#a6adc8;")
        aruco_lay.addWidget(self._aruco_log)

        # ── Results panel: Camera→Marker + Final Frame ────────────────────
        res_grp = QGroupBox("📐  Calibration Results")
        res_grp.setStyleSheet("QGroupBox { border:1px solid #45475a; border-radius:5px; margin-top:4px; }")
        res_lay = QHBoxLayout(res_grp)
        res_lay.setSpacing(16)

        def _make_result_col(title: str, colour: str):
            """Create a labelled column widget; returns (QFrame, dict[key→QLabel])."""
            col = QVBoxLayout()
            hdr = QLabel(title)
            hdr.setStyleSheet(f"font-weight:bold; color:{colour}; font-size:11px;")
            col.addWidget(hdr)
            labels = {}
            for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'roll', 'pitch', 'yaw'):
                row = QHBoxLayout()
                nm = QLabel(f"{k}:")
                nm.setStyleSheet(f"color:{C['text2']}; font-size:10px;")
                nm.setFixedWidth(38)
                vl = QLabel("—")
                vl.setStyleSheet(f"color:{colour}; font-family:Monospace; font-size:11px;")
                row.addWidget(nm); row.addWidget(vl); row.addStretch()
                col.addLayout(row)
                labels[k] = vl
            col.addStretch()
            return col, labels

        # Left: Camera → Marker (kinect2_ir_optical_frame → detected_marker_frame)
        col_cm, self._res_cam_marker = _make_result_col(
            "📷 Camera → Marker  (raw ArUco)", "#89b4fa"
        )
        res_lay.addLayout(col_cm)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{C['border']};")
        res_lay.addWidget(sep)

        # Right: base_link → kinect2_link (final calibrated frame)
        col_bc, self._res_base_cam = _make_result_col(
            "🎯 Final Frame: base_link → camera", "#a6e3a1"
        )
        res_lay.addLayout(col_bc)

        aruco_lay.addWidget(res_grp)
        self._aruco_results_grp = res_grp

        lay.addWidget(aruco_grp)
        lay.addStretch()

        self._aruco_workers: list = []
        QTimer.singleShot(200, self._cal_load_yaml)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        return scroll


    # ── Camera frame helpers ──────────────────────────────────────────────────

    def _cal_load_yaml(self) -> None:
        """Read ~/.parol6/camera_tf.yaml and update the Current Frame panel."""
        import yaml as _yaml
        from pathlib import Path as _Path
        _path = _Path.home() / '.parol6' / 'camera_tf.yaml'
        try:
            with open(_path) as f:
                cfg = _yaml.safe_load(f) or {}
            at  = cfg.get('calibrated_at', '?')
            by  = cfg.get('calibrated_by', '?')
            self._cal_source_lbl.setText(f"Source: {_path}  ·  {at}  ·  {by}")
            self._cal_source_lbl.setStyleSheet(f"color:{C['green']}; font-size:10px;")
            for key in ('x','y','z','qx','qy','qz','qw'):
                if key in cfg:
                    self._cal_val_labels[key].setText(f"{float(cfg[key]):.4f}")
                else:
                    self._cal_val_labels[key].setText("—")
            self._cal_child_lbl.setText(str(cfg.get('child_frame_id', '?')))
        except FileNotFoundError:
            self._cal_source_lbl.setText(f"⚠️  {_path} not found — pipeline uses hardcoded defaults")
            self._cal_source_lbl.setStyleSheet(f"color:{C['yellow']}; font-size:10px;")
            for lbl in self._cal_val_labels.values():
                lbl.setText("—")
        except Exception as e:
            self._cal_source_lbl.setText(f"Error loading yaml: {e}")
            self._cal_source_lbl.setStyleSheet(f"color:{C['red']}; font-size:10px;")

    def _cal_enforce_start(self) -> None:
        """Launch camera_tf_enforcer as a background process."""
        if self._cal_enforcer_worker is not None:
            return
        cmd = ["ros2", "run", "parol6_vision", "camera_tf_enforcer"]
        # NodeWorker(cmd) — passes raw list; NodeWorker wraps it internally
        worker = NodeWorker(cmd)
        worker.line_out.connect(
            lambda t: self._log.append(f"<span style='color:#89b4fa'>[CamTF] {t}</span>")
        )
        worker.finished.connect(self._cal_enforce_stopped)
        worker.start()
        self._cal_enforcer_worker = worker
        self._cal_enforce_btn.setEnabled(False)
        self._cal_stop_enforce_btn.setEnabled(True)
        self._cal_enforce_status.setText("✅  ACTIVE — dynamic TF override running at 100 Hz")
        self._cal_enforce_status.setStyleSheet(f"color:{C['green']}; font-size:11px; font-weight:bold;")

    def _cal_enforce_stopped(self) -> None:
        self._cal_enforcer_worker = None
        self._cal_enforce_btn.setEnabled(True)
        self._cal_stop_enforce_btn.setEnabled(False)
        self._cal_enforce_status.setText("⚠️  Not active — pipeline uses static TF from launch file")
        self._cal_enforce_status.setStyleSheet(f"color:{C['yellow']}; font-size:11px;")

    def _cal_enforce_stop(self) -> None:
        if self._cal_enforcer_worker is not None:
            self._cal_enforcer_worker.abort()  # correct method name
        self._cal_enforce_stopped()

    # ── Manual frame entry helpers ────────────────────────────────────────────

    def _man_save_and_enforce(self) -> None:
        """Write spinbox values to camera_tf.yaml, reload display, start enforcer."""
        import yaml as _yaml
        import datetime
        from pathlib import Path as _Path

        child = self._man_child_combo.currentData()
        cfg = {
            'frame_id':       'base_link',
            'child_frame_id': child,
            'x':  round(self._man_x.value(),  4),
            'y':  round(self._man_y.value(),  4),
            'z':  round(self._man_z.value(),  4),
            'qx': round(self._man_qx.value(), 4),
            'qy': round(self._man_qy.value(), 4),
            'qz': round(self._man_qz.value(), 4),
            'qw': round(self._man_qw.value(), 4),
            'calibrated_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'calibrated_by': 'manual entry (GUI)',
        }
        out = _Path.home() / '.parol6' / 'camera_tf.yaml'
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out, 'w') as f:
                _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            self._man_status_lbl.setText(f"✅  Saved")
            self._man_status_lbl.setStyleSheet(f"color:{C['green']}; font-size:11px;")
        except Exception as e:
            self._man_status_lbl.setText(f"❌  Save failed: {e}")
            self._man_status_lbl.setStyleSheet(f"color:{C['red']}; font-size:11px;")
            return
        self._cal_load_yaml()
        self._cal_enforce_stop()
        QTimer.singleShot(500, self._cal_enforce_start)

    def _man_load_from_yaml(self) -> None:
        """Populate the manual-entry spinboxes from the currently saved yaml."""
        import yaml as _yaml
        from pathlib import Path as _Path
        _path = _Path.home() / '.parol6' / 'camera_tf.yaml'
        try:
            with open(_path) as f:
                cfg = _yaml.safe_load(f) or {}
            self._man_x.setValue(float(cfg.get('x',  0)))
            self._man_y.setValue(float(cfg.get('y',  0)))
            self._man_z.setValue(float(cfg.get('z',  0)))
            self._man_qx.setValue(float(cfg.get('qx', 0)))
            self._man_qy.setValue(float(cfg.get('qy', 0)))
            self._man_qz.setValue(float(cfg.get('qz', 0)))
            self._man_qw.setValue(float(cfg.get('qw', 1)))
            child = cfg.get('child_frame_id', 'kinect2_link')
            for i in range(self._man_child_combo.count()):
                if self._man_child_combo.itemData(i) == child:
                    self._man_child_combo.setCurrentIndex(i)
                    break
            self._man_status_lbl.setText("✅  Loaded")
            self._man_status_lbl.setStyleSheet(f"color:{C['green']}; font-size:11px;")
        except FileNotFoundError:
            self._man_status_lbl.setText("⚠️  File not found")
            self._man_status_lbl.setStyleSheet(f"color:{C['yellow']}; font-size:11px;")
        except Exception as e:
            self._man_status_lbl.setText(f"❌  {e}")
            self._man_status_lbl.setStyleSheet(f"color:{C['red']}; font-size:11px;")

    # ── ArUco calibration helpers ─────────────────────────────────────────────

    def _aruco_log_append(self, txt: str) -> None:

        if not hasattr(self, '_aruco_log'):
            return
        self._aruco_log.append(txt)
        import re

        # ── Progress tracking ─────────────────────────────────────────────
        m = re.search(r'Collected (\d+)/(\d+)', txt)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            self._aruco_progress.setMaximum(total)
            self._aruco_progress.setValue(n)

        # ── Parse machine-readable result lines ───────────────────────────
        def _parse_kv(line: str) -> dict:
            out = {}
            for token in line.split():
                if '=' in token:
                    k, v = token.split('=', 1)
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            return out

        def _fill_result(labels: dict, kv: dict) -> None:
            for k, lbl in labels.items():
                if k in kv:
                    v = kv[k]
                    lbl.setText(f"{v:.4f}" if isinstance(v, float) else str(v))

        if '[CAL_CAM_MARKER]' in txt:
            payload = txt.split('[CAL_CAM_MARKER]', 1)[1]
            _fill_result(self._res_cam_marker, _parse_kv(payload))

        if '[CAL_BASE_CAM]' in txt:
            payload = txt.split('[CAL_BASE_CAM]', 1)[1]
            _fill_result(self._res_base_cam, _parse_kv(payload))

        # ── Completion / yaml saved ───────────────────────────────────────
        if 'Saved to' in txt or 'camera_tf.yaml' in txt:
            self._aruco_enforce_btn.setEnabled(True)
            self._aruco_progress.setValue(self._aruco_progress.maximum())
            QTimer.singleShot(1000, self._cal_load_yaml)

    def _aruco_start(self) -> None:
        """Launch aruco_detector + eye_to_hand_calibrator."""
        marker_id   = self._aruco_id_spin.value()
        marker_size = self._aruco_size_spin.value() / 1000.0
        n_samples   = self._aruco_samples_spin.value()
        mx, my, mz  = self._aruco_mx.value(), self._aruco_my.value(), self._aruco_mz.value()

        self._aruco_log.clear()
        self._aruco_progress.setValue(0)
        self._aruco_progress.setMaximum(n_samples)
        self._aruco_enforce_btn.setEnabled(False)
        self._aruco_aborted = False

        aruco_cmd = [
            "ros2", "run", "parol6_vision", "aruco_detector",
            "--ros-args",
            "-p", "image_topic:=/kinect2/sd/image_color_rect",
            "-p", "camera_info_topic:=/kinect2/sd/camera_info",
            "-p", f"marker_id:={marker_id}",
            "-p", f"marker_size:={marker_size:.5f}",
            "-p", "camera_optical_frame:=kinect2_ir_optical_frame",
            "-p", "marker_frame:=detected_marker_frame",
            "-p", "marker_dict:=DICT_ARUCO_ORIGINAL",
        ]
        self._aruco_log_append(
            f"<b style='color:#89b4fa'>[GUI] ArUco calibration starting…</b>"
        )
        try:
            # NodeWorker(raw_cmd): NodeWorker.__init__ calls _wrap_ros_command internally
            aruco_worker = NodeWorker(aruco_cmd)
            aruco_worker.line_out.connect(
                lambda t: self._aruco_log_append(f"<span style='color:#a6adc8'>[aruco_detector] {t}</span>")
            )
            aruco_worker.start()

            cal_cmd = [
                "ros2", "run", "parol6_vision", "eye_to_hand_calibrator",
                "--ros-args",
                "-p", f"marker_x:={mx:.4f}",
                "-p", f"marker_y:={my:.4f}",
                "-p", f"marker_z:={mz:.4f}",
                "-p", f"samples_to_collect:={n_samples}",
            ]
            cal_worker = NodeWorker(cal_cmd)
            cal_worker.line_out.connect(self._aruco_log_append)
            cal_worker.finished.connect(self._aruco_calibration_done)
            QTimer.singleShot(2000, cal_worker.start)

            self._aruco_workers = [aruco_worker, cal_worker]
            self._aruco_run_btn.setEnabled(False)
            self._aruco_stop_btn.setEnabled(True)
            self._aruco_log_append(
                f"<b style='color:#89b4fa'>[GUI] ArUco calibration started — "
                f"marker #{marker_id} ({marker_size*1000:.1f} mm), "
                f"{n_samples} samples, marker @ ({mx:.3f}, {my:.3f}, {mz:.3f}) m<br>"
                f"aruco_detector launched; eye_to_hand_calibrator will start in 2 s…</b>"
            )
        except Exception as _exc:
            self._aruco_log_append(
                f"<span style='color:#f38ba8'>[GUI] ❌ Failed to launch: {_exc}</span>"
            )
            self._aruco_run_btn.setEnabled(True)
            self._aruco_stop_btn.setEnabled(False)

    def _aruco_calibration_done(self) -> None:
        self._aruco_stop_btn.setEnabled(False)
        self._aruco_run_btn.setEnabled(True)
        if getattr(self, '_aruco_aborted', False):
            return  # Already logged during stop

        self._aruco_log_append(
            "<b style='color:#a6e3a1'>[GUI] ✅ Calibration finished. "
            "Click \"Save & Enforce\" to apply the new camera frame live.</b>"
        )
        if self._aruco_workers:
            self._aruco_workers[0].abort()   # stop aruco_detector

    def _aruco_stop(self) -> None:
        self._aruco_aborted = True
        for w in self._aruco_workers:
            try: w.abort()  # correct method name
            except Exception: pass
        self._aruco_workers = []
        self._aruco_run_btn.setEnabled(True)
        self._aruco_stop_btn.setEnabled(False)
        self._aruco_log_append("<span style='color:#f38ba8'>[GUI] Calibration stopped.</span>")

    def _aruco_save_and_enforce(self) -> None:
        """Reload display from yaml and kick-start the enforcer node."""
        self._cal_load_yaml()
        self._aruco_log_append("<b style='color:#a6e3a1'>[GUI] Saved. Starting enforcer...</b>")
        self._cal_enforce_stop()
        QTimer.singleShot(500, self._cal_enforce_start)

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

    def _start_transient_worker(self, worker: NodeWorker) -> None:
        """Keep short-lived workers alive until their thread fully exits."""
        if not hasattr(self, "_trigger_workers"):
            self._trigger_workers = []
        self._trigger_workers.append(worker)
        worker.finished.connect(
            lambda rc, w=worker: self._trigger_workers.remove(w)
            if w in self._trigger_workers else None
        )
        worker.start()

    def _park_worker(self, worker) -> None:
        """Keep aborted workers alive until their QThread fully exits."""
        if worker is None:
            return
        if worker not in self._graveyard:
            self._graveyard.append(worker)
            worker.finished.connect(
                lambda *_, w=worker: self._graveyard.remove(w)
                if w in self._graveyard else None
            )

    def _abort_and_park_worker(self, worker, wait_ms: int = 0):
        if worker is None:
            return None
        try:
            worker.abort()
        except Exception:
            pass
        self._park_worker(worker)
        if wait_ms > 0:
            try:
                worker.wait(wait_ms)
            except Exception:
                pass
        return None

    def _switch_path_holder_source(self, source: str) -> None:
        """
        Tell path_holder to switch its active source.
        Sets the parameter first, then calls ~/set_source.
        Returns immediately; result is logged asynchronously.
        """
        if not (ROS2_OK and self._ros_node):
            return
        try:
            from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
            pv = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=source)
            req = SetParameters.Request()
            req.parameters = [Parameter(name='active_source', value=pv)]
            fut_param = self._path_holder_set_params_client.call_async(req)

            def _after_param(f):
                # Now call ~/set_source to apply
                if self._path_holder_set_source_client.service_is_ready():
                    fut_svc = self._path_holder_set_source_client.call_async(Trigger.Request())
                    def _on_switch(sf):
                        try:
                            r = sf.result()
                            color = C['green'] if r.success else C['red']
                            html = f'<span style="color:{color}">[PathHolder] {r.message}</span>'
                            self._log.append(html)
                            if hasattr(self, '_ros_log'):
                                self._ros_log.append(html)
                        except Exception:
                            pass
                    fut_svc.add_done_callback(_on_switch)
            fut_param.add_done_callback(_after_param)
        except Exception as exc:
            self._log.append(f'<span style="color:{C["red"]}">[PathHolder] Source switch error: {exc}</span>')

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
            "align":  ["ros2", "run", "parol6_vision", "manual_line_aligner"],
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
        self._mode_worker.finished.connect(
            lambda rc, w=self._mode_worker: self._on_mode_finished(w, rc)
        )
        self._mode_worker.start()
        self._btn_start_mode.setEnabled(False)
        self._btn_stop_mode.setEnabled(True)
        self._log.append(
            f'<b style="color:{C["green"]}">[Mode] Started: {mode}</b>'
        )

    def _stop_mode_node(self) -> None:
        self._mode_worker = self._abort_and_park_worker(self._mode_worker)
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    def _on_mode_finished(self, worker: NodeWorker, rc: int) -> None:
        if self._mode_worker is worker:
            self._mode_worker = None
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    # ── Manual Red Line actions ───────────────────────────────────────────────

    def _on_roi_toggled(self, state: int) -> None:
        """Legacy handler — kept for compatibility but ROI is now in Auto-Align tab."""
        pass

    def _on_straight_line_toggled(self, state: int) -> None:
        """Toggle straight-line drawing mode on the canvas."""
        if hasattr(self, '_canvas') and CV2_OK:
            self._canvas.set_straight_line_mode(bool(state))

    # ── Auto-Align tab actions ─────────────────────────────────────────────────

    def _on_align_roi_toggled(self, state: int) -> None:
        if state == Qt.Checked:
            if hasattr(self, '_align_straight_check'): self._align_straight_check.setChecked(False)
            if getattr(self, '_align_canvas', None):
                self._align_canvas.set_roi_mode(True)
        else:
            if getattr(self, '_align_canvas', None):
                self._align_canvas.set_roi_mode(False)

    def _align_use_latest_cropped(self) -> None:
        if not CV2_OK:
            return
        rgb = getattr(self, '_latest_cropped_rgb', None)
        if rgb is not None and getattr(self, '_align_canvas', None):
            self._align_canvas.load_image(rgb.copy())
        else:
            if hasattr(self, '_align_log'):
                self._align_log.append('<span style="color:#f9e2af">[Align] No cropped frame yet — trigger a capture first.</span>')

    def _align_teach_send(self) -> None:
        """Serialise align canvas strokes+ROI and call manual_line_aligner/teach_reference."""
        if not (ROS2_OK and CV2_OK):
            self._align_log.append('<span style="color:#f38ba8">[Align] ROS2 + cv2 required.</span>')
            return
        strokes = getattr(self, '_align_canvas', None) and self._align_canvas.get_strokes()
        roi_polygon = getattr(self, '_align_canvas', None) and self._align_canvas.get_roi_polygon()
        if not strokes:
            self._align_log.append('<span style="color:#f9e2af">[Align] Draw weld strokes first.</span>')
            return
        if not roi_polygon:
            self._align_log.append('<span style="color:#f9e2af">[Align] Draw an ROI boundary first (check the ROI checkbox).</span>')
            return
        import json
        payload = {'color': [0, 0, 255], 'width': self._align_brush_spin.value(),
                   'strokes': strokes, 'roi_polygon': roi_polygon}
        json_str = json.dumps(payload)
        self._align_log.append(f'<span style="color:#a6e3a1">[Align] Teaching {len(strokes)} stroke(s) + {len(roi_polygon)}-pt ROI…</span>')

        if not hasattr(self, '_aligner_set_params_client') or self._aligner_set_params_client is None:
            self._align_log.append('<span style="color:#f38ba8">[Align] Aligner ROS client not ready.</span>')
            return

        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        pv = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=json_str)
        req = SetParameters.Request()
        req.parameters = [Parameter(name='strokes_json', value=pv)]
        future = self._aligner_set_params_client.call_async(req)

        def _on_param_set(fut):
            try:
                result = fut.result()
                if result and result.results and result.results[0].successful:
                    svc_w = NodeWorker([
                        "ros2", "service", "call", "/manual_line_aligner/teach_reference",
                        "std_srvs/srv/Trigger", "{}",
                    ])
                    svc_w.line_out.connect(lambda s: self._align_log.append(f'<span style="color:#a6e3a1">[Align] {s}</span>'))
                    self._trigger_workers.append(svc_w)
                    svc_w.finished.connect(lambda r, w=svc_w: self._trigger_workers.remove(w) if w in self._trigger_workers else None)
                    svc_w.start()
                else:
                    msg = result.results[0].reason if result and result.results else 'unknown'
                    self._align_log.append(f'<span style="color:#f38ba8">[Align] Param set failed: {msg}</span>')
            except Exception as e:
                self._align_log.append(f'<span style="color:#f38ba8">[Align] Error: {e}</span>')
        future.add_done_callback(_on_param_set)

    def _align_reset(self) -> None:
        if getattr(self, '_align_canvas', None) and CV2_OK:
            self._align_canvas.clear_mask()
        if not ROS2_OK:
            return
        svc_w = NodeWorker([
            "ros2", "service", "call", "/manual_line_aligner/reset_strokes",
            "std_srvs/srv/Trigger", "{}",
        ])
        svc_w.line_out.connect(lambda s: self._align_log.append(f'<span style="color:#f9e2af">[Align] {s}</span>'))
        self._trigger_workers.append(svc_w)
        svc_w.finished.connect(lambda r, w=svc_w: self._trigger_workers.remove(w) if w in getattr(self, "_trigger_workers", []) else None)
        svc_w.start()

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
        roi_polygon = getattr(self._canvas, 'get_roi_polygon', lambda: [])()
        
        if not strokes:
            self._manual_log_append('<span style="color:#f9e2af">[Manual] No strokes to send — draw something first.</span>')
            return
        c = self._manual_color_picker.color()
        w_px = self._manual_brush_spin.value()
        import json
        payload = {'color': [c.blue(), c.green(), c.red()], 'width': w_px, 'strokes': strokes}
        if roi_polygon:
            payload['roi_polygon'] = roi_polygon
            
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
            lambda s: self._manual_log_append(
                f'<span style="color:#f9e2af">[Manual] {s}</span>'
            )
        )
        self._trigger_workers.append(svc_w)
        svc_w.finished.connect(lambda r, w=svc_w: self._trigger_workers.remove(w) if w in getattr(self, "_trigger_workers", []) else None)
        svc_w.start()
        self._manual_log_append('<span style="color:#f9e2af">[Manual] Reset sent → /manual_line/reset_strokes</span>')
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
        QTimer.singleShot(2400, self._btn_path_holder._toggle)

    def _stop_all(self) -> None:
        for btn in (self._btn_live_cam, self._btn_capture,
                    self._btn_crop_node,
                    self._btn_optimizer, self._btn_depth,
                    self._btn_pathgen, self._btn_path_holder,
                    self._btn_moveit):
            btn.stop()
        self._stop_mode_node()
        self._ros_worker = self._abort_and_park_worker(self._ros_worker)
        self._ros_test_worker = self._abort_and_park_worker(self._ros_test_worker)

    def _send_path_to_moveit(self) -> None:
        """
        Two-step sequence so moveit_controller always receives the path before execute:
          1. Call path_holder/set_source  →  forces path_holder to republish the
             cached path as a fresh TRANSIENT_LOCAL message, waking any late-joining
             moveit_controller subscriber.
          2. After 1 s (DDS delivery), call execute_welding_path trigger service.
        """
        def _log(msg: str, color: str = C["accent"]) -> None:
            html = f'<span style="color:{color}">[MoveIt] {msg}</span>'
            self._log.append(html)
            if hasattr(self, "_ros_log"):
                self._ros_log.append(html)

        if not ROS2_OK or self._ros_node is None:
            _log("ROS 2 client unavailable in GUI.", C["red"])
            return

        if self._moveit_execute_client is None:
            _log("MoveIt execute client not ready.", C["red"])
            return

        if not self._moveit_execute_client.wait_for_service(timeout_sec=1.0):
            _log("/moveit_controller/execute_welding_path is not available.", C["red"])
            return

        # ── Step 1: force path_holder to republish cached path ──────────────
        def _do_execute():
            """Called 1 s after path_holder refresh to trigger MoveIt."""
            _log("Triggering MoveIt execution…", C["yellow"])
            exec_future = self._moveit_execute_client.call_async(Trigger.Request())
            self._crop_futures.append(exec_future)

            def _on_exec_done(exec_fut):
                try:
                    exec_result = exec_fut.result()
                    if exec_result and exec_result.success:
                        _log(exec_result.message or "Execution started.", C["green"])
                    else:
                        reason = exec_result.message if exec_result else "unknown failure"
                        _log(f"Execution failed: {reason}", C["red"])
                except Exception as exc:
                    _log(f"Execution service error: {exc}", C["red"])
            exec_future.add_done_callback(_on_exec_done)

        path_holder_ready = (
            hasattr(self, "_path_holder_set_source_client")
            and self._path_holder_set_source_client is not None
            and self._path_holder_set_source_client.service_is_ready()
        )

        if path_holder_ready:
            _log("Refreshing path_holder cache → moveit_controller…", C["yellow"])
            refresh_future = self._path_holder_set_source_client.call_async(Trigger.Request())
            self._crop_futures.append(refresh_future)

            def _on_refresh_done(fut):
                try:
                    res = fut.result()
                    if res and res.success:
                        _log(f"Path refreshed: {res.message}", C["text2"])
                    else:
                        reason = res.message if res else "no response"
                        _log(f"path_holder refresh: {reason} (continuing anyway)", C["yellow"])
                except Exception as exc:
                    _log(f"path_holder refresh error: {exc} (continuing anyway)", C["yellow"])
                # Always proceed to execute after refresh attempt
                from PySide6.QtCore import QTimer
                QTimer.singleShot(1000, _do_execute)

            refresh_future.add_done_callback(_on_refresh_done)
        else:
            # path_holder not available — try execute directly
            _log("path_holder not available, trying direct execute…", C["yellow"])
            _do_execute()



    def _launch_vision_moveit(self) -> None:
        worker = NodeWorker(
            ["ros2", "launch", "parol6_vision", "vision_moveit.launch.py"],
        )
        worker.line_out.connect(self._log.append)
        self._start_transient_worker(worker)

    # ── ROS Launch Tab actions ─────────────────────────────────────────────

    def _ros_launch_toggle(self) -> None:
        if self._ros_worker:
            self._ros_log.append("[LAUNCH] Stopping process...")
            self._gazebo_log.append("[LAUNCH] Stopping process...")
            self._ros_worker = self._abort_and_park_worker(self._ros_worker)
            self._set_ros_button_state(False)
            return

        script_name = self._ros_method.currentData()
        script_path = os.path.join(self._launchers_dir, script_name)

        self._ros_log.clear()
        self._gazebo_log.clear()

        if not os.path.exists(script_path):
            self._ros_log.append(f"[LAUNCH] \u274c Error: Cannot find script {script_path}")
            return

        self._ros_launch_env.clear()
        self._ros_worker = RosLaunchWorker(script_path, [], env_vars=self._ros_launch_env)
        self._ros_worker.output_rviz.connect(self._ros_log.append)
        self._ros_worker.output_gazebo.connect(self._gazebo_log.append)
        self._ros_worker.finished_ok.connect(lambda w=self._ros_worker: self._on_ros_launch_finished(w))
        self._ros_worker.finished_err.connect(lambda rc, w=self._ros_worker: self._on_ros_launch_finished(w, rc))
        self._ros_worker.start()
        self._set_ros_button_state(True)

    def _set_ros_button_state(self, is_running: bool) -> None:
        if is_running:
            self._ros_launch_btn.setText("🛑 Stop")
            self._ros_launch_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
            self._ros_method.setEnabled(False)
        else:
            self._ros_launch_btn.setText("🚀 Launch")
            self._ros_launch_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")
            self._ros_method.setEnabled(True)

    def _on_ros_launch_finished(self, worker=None, rc: Optional[int] = None) -> None:
        if worker is not None and self._ros_worker is not worker:
            return
        self._ros_worker = None
        self._set_ros_button_state(False)

    def _ros_kill_all_nodes(self) -> None:
        self._ros_log.append("[LAUNCH] ⚠️ Sending KILL signal to all Gazebo/RViz/MoveIt processes...")
        cmd = "pkill -9 -f 'ros2|rviz2|ign|gazebo|ruby|move_group|parameter_bridge|robot_state_publisher|launch_'"
        if os.path.exists("/.dockerenv"):
            full_cmd = ["bash", "-c", cmd]
        else:
            full_cmd = ["docker", "exec", "parol6_dev", "bash", "-c", cmd]
        try:
            subprocess.run(full_cmd, check=False)
            self._ros_log.append("[LAUNCH] ✅ Kill command executed. Zombie processes terminated.")
        except Exception as exc:
            self._ros_log.append(f"[LAUNCH] ❌ Error executing kill: {exc}")

    def _ros_run_auto_test(self) -> None:
        if self._ros_test_worker:
            self._ros_log.append("[TEST] Stopping currently running test...")
            self._ros_test_worker = self._abort_and_park_worker(self._ros_test_worker)
            self._test_btn.setText("▶️ Run Auto-Test")
            self._test_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")
            return

        script_path = os.path.join(self._launchers_dir, "launch_auto_test.sh")
        if not os.path.exists(script_path):
            self._ros_log.append(f"[TEST] \u274c Cannot find script {script_path}")
            return

        shape = self._test_shape_combo.currentText()
        self._ros_log.append(f"\n[TEST] Launching comprehensive Auto-Test ({shape})...")
        self._ros_log.append("[TEST] Spawning moveit_controller and waiting for services...")

        self._ros_launch_env.clear()
        self._ros_test_worker = RosLaunchWorker(script_path, [shape], env_vars=self._ros_launch_env)
        self._ros_test_worker.output_rviz.connect(self._ros_log.append)
        self._ros_test_worker.finished_ok.connect(lambda w=self._ros_test_worker: self._on_ros_test_finished(w))
        self._ros_test_worker.finished_err.connect(lambda rc, w=self._ros_test_worker: self._on_ros_test_finished(w, rc))
        self._ros_test_worker.start()

        self._test_btn.setText("🛑 Stop Test")
        self._test_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")

    def _on_ros_test_finished(self, worker=None, rc: Optional[int] = None) -> None:
        if worker is not None and self._ros_test_worker is not worker:
            return
        self._ros_test_worker = None
        self._test_btn.setText("▶️ Run Auto-Test")
        self._test_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;")

    def _inject_test_path(self) -> None:
        """Send the conservative reachable test path through inject_path_node.

        Uses the GUI's persistent ROS publisher to /vision/inject_path.
        inject_path_node re-latches it to /vision/welding_path/injected and
        path_holder picks it up when source is 'injected'.
        No subprocess, no DDS discovery race.
        """
        if not (ROS2_OK and self._ros_node and self._inject_path_pub):
            self._log.append(
                f'<span style="color:{C["red"]}">[Inject] ROS unavailable.</span>'
            )
            return

        from geometry_msgs.msg import Pose, Point, Quaternion
        path = ROSPath()
        path.header.frame_id = 'base_link'
        path.header.stamp = self._ros_node.get_clock().now().to_msg()

        # FK-confirmed home: x=0.20, z=0.33 — 5 lateral waypoints
        waypoints = [
            (-0.08, 0.33), (-0.04, 0.33), (0.00, 0.33), (0.04, 0.33), (0.08, 0.33)
        ]
        for y, z in waypoints:
            from geometry_msgs.msg import PoseStamped
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = 0.20
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.x = 0.707
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = -0.707
            ps.pose.orientation.w = 0.0
            path.poses.append(ps)

        self._inject_path_pub.publish(path)
        self._log.append(
            f'<b style="color:{C["yellow"]}">[Inject] Test path ({len(path.poses)} poses) → '
            '/vision/inject_path → inject_path_node → path_holder</b>'
        )
        if hasattr(self, '_ros_log'):
            self._ros_log.append(
                f'<b style="color:{C["yellow"]}">[Inject] Published {len(path.poses)}-waypoint '
                'test path. Switching path_holder source → injected.</b>'
            )
        # Switch path_holder to serve the injected path
        self._switch_path_holder_source('injected')


    def _kill_all(self) -> None:
        """Stop managed workers, then signal stray ROS processes to exit."""
        self._stop_all()
        cmd = (
            "pkill -INT -f 'rviz2|move_group|ros2_control_node|ros2 run parol6' ; "
            "sleep 1 ; "
            "pkill -TERM -f 'rviz2|move_group|ros2_control_node|ros2 run parol6' "
            "2>/dev/null || true"
        )
        subprocess.Popen(["bash", "-c", cmd])
        self._ros_log.append(
            "[KILL] Sent SIGINT to stray ROS 2 processes "
            "(SIGTERM fallback after 1 s)."
        )

    def _ros_kill_all(self) -> None:
        """Compatibility wrapper for older signal/slot hookups."""
        self._kill_all()

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
            "if [ -f /opt/kinect_ws/install/setup.bash ]; then "
            "source /opt/kinect_ws/install/setup.bash; "
            "fi && "
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

    def _on_weld_path_received(self, msg) -> None:
        """Real callback: fires when an actual welding path arrives from path_holder."""
        n = len(msg.poses)
        if n == 0:
            return
        fn = getattr(self, '_simple_set_status', None)
        if fn:
            fn(f"[PATH] Ready! ({n} waypoints) - Proceed to Step 3", "#a6e3a1")

    def _on_depth_confirmed(self, msg) -> None:
        """Real callback: fires when captured depth image arrives, confirming capture."""
        fn = getattr(self, '_simple_set_status', None)
        if fn:
            fn("[SCAN] Depth confirmed - Generating path...", "#f9e2af")

    def _start_simple_status_poller(self) -> None:
        """Start a QTimer that polls real NodeWorker process states every 2 seconds.
        
        Updates the status box with a truthful summary of which pipeline nodes
        are actually alive according to the OS process table — no fake timer tricks.
        """
        def _is_alive(worker) -> bool:
            if worker is None:
                return False
            proc = getattr(worker, '_proc', None)
            if proc is None:
                # May not have started ever
                thread = getattr(worker, '_worker', None)
                if thread:
                    proc = getattr(thread, '_proc', None)
            return proc is not None and proc.poll() is None

        def _poll_nodes():
            fn = getattr(self, '_simple_set_status', None)
            if not fn:
                return

            cam_up    = _is_alive(getattr(self, '_btn_live_cam', None))
            cap_up    = _is_alive(getattr(self, '_btn_capture', None))
            crop_up   = _is_alive(getattr(self, '_btn_crop_node', None))
            opt_up    = _is_alive(getattr(self, '_btn_optimizer', None))
            depth_up  = _is_alive(getattr(self, '_btn_depth', None))
            gen_up    = _is_alive(getattr(self, '_btn_pathgen', None))
            moveit_up = _is_alive(getattr(self, '_btn_moveit', None))

            # Build a compact node health row
            nodes = [
                ("Cam", cam_up), ("Cap", cap_up), ("Crop", crop_up),
                ("Opt", opt_up), ("Depth", depth_up), ("Gen", gen_up),
                ("MoveIt", moveit_up),
            ]
            up   = [name for name, alive in nodes if alive]
            down = [name for name, alive in nodes if not alive]

            if not any(alive for _, alive in nodes):
                return

            if all(alive for _, alive in nodes):
                fn("[OK] All nodes online - System Ready.", "#a6e3a1")
            else:
                up_str   = " ".join(up)   or "--"
                down_str = " ".join(down) or "--"
                fn(f"(WARN) Online: {up_str} | Offline: {down_str}", "#f9e2af")

        self._simple_poller = QTimer(self)
        self._simple_poller.timeout.connect(_poll_nodes)
        self._simple_poller.start(2000)  # every 2 s

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

    def _manual_save_profile(self) -> None:
        """Save the current drawn strokes as a JSON profile via an OS dialog."""
        strokes = getattr(self, '_canvas', None) and self._canvas.get_strokes()
        if not strokes:
            QMessageBox.warning(self, "No Strokes", "Please draw strokes before saving a profile.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Stroke Profile", "", "JSON Files (*.json)"
        )
        if not out_path:
            return

        c = getattr(self, '_manual_color_picker', None) 
        if c: c = c.get_color()
        else: c = QColor(255, 0, 0)
        
        w_px = getattr(self, '_manual_brush_spin', None)
        w_px = w_px.value() if w_px else 5
        
        payload = {
            'color': [c.blue(), c.green(), c.red()],
            'width': w_px,
            'strokes': strokes
        }
        
        import json
        try:
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            self._manual_log_append(f'<span style="color:#a6e3a1">[Manual] Saved {len(strokes)} strokes to {out_path}</span>')
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save profile:\n{e}")

    def _manual_load_profile(self) -> None:
        """Load strokes from a JSON profile via an OS dialog."""
        in_path, _ = QFileDialog.getOpenFileName(
            self, "Load Stroke Profile", "", "JSON Files (*.json)"
        )
        if not in_path:
            return

        import json
        try:
            with open(in_path, 'r') as f:
                payload = json.load(f)
            
            strokes = payload.get('strokes', [])
            if hasattr(self, '_canvas'):
                self._canvas.load_strokes(strokes)
            self._manual_log_append(f'<span style="color:#a6e3a1">[Manual] Loaded {len(strokes)} strokes from {in_path}</span>')
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load profile:\n{e}")

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
        for worker in list(getattr(self, "_trigger_workers", [])):
            try:
                worker.abort()
                worker.wait(1500)
            except Exception:
                pass
        for worker in list(getattr(self, "_graveyard", [])):
            try:
                worker.wait(1500)
            except Exception:
                pass
        for worker in (
            getattr(self, "_mode_worker", None),
            getattr(self, "_ros_worker", None),
            getattr(self, "_ros_test_worker", None),
        ):
            if worker is not None:
                try:
                    worker.wait(1500)
                except Exception:
                    pass
        if hasattr(self, "_ros_timer"):
            try:
                self._ros_timer.stop()
            except Exception:
                pass
        ev.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 11))
    win = VisionPipelineGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
