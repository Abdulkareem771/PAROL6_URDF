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
import os
import sys
import signal
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QLabel, QPushButton, QRadioButton,
    QButtonGroup, QTextEdit, QScrollArea, QFrame, QTabWidget,
    QSizePolicy, QGridLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSpinBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QMessageBox, QCheckBox, QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QFont, QColor, QPalette, QPixmap, QImage, QIcon, QPainter,
    QPen, QCursor,
)

# ─── Try importing ROS / cv_bridge for live topic previews ───────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as ROSImage
    from cv_bridge import CvBridge
    ROS2_OK = True
except ImportError:
    ROS2_OK = False

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE_DIR = Path("/home/osama/Desktop/PAROL6_URDF")
VISION_WORK   = WORKSPACE_DIR / "vision_work"
VISION_PKG    = "parol6_vision"

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
        self._cmd  = cmd
        self._env  = env or os.environ.copy()
        self._proc: Optional[subprocess.Popen] = None

    def run(self) -> None:
        self.line_out.emit(f"[RUN] $ {' '.join(self._cmd)}")
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self._env,
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

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['border']};"
            " border-radius:4px; color:#585b70;"
        )
        self.setText(f"⌛  Waiting for\n{topic}")
        self.setMinimumSize(200, 150)

        if ROS2_OK and ros_node:
            ros_node.create_subscription(
                ROSImage, topic, self._ros_cb, 1
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
        qimg = QImage(frame_small.data, nw, nh, ch * nw, QImage.Format_RGB888)
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

        placeholder = QLabel("Load an image above to start drawing red lines.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(f"color:{C['text2']}; font-size:13px;")
        self._scene.addWidget(placeholder)

    # -- public API -----------------------------------------------------------
    def set_brush(self, px: int) -> None:
        self._brush = max(1, px)
        self._update_cursor()

    def load_image(self, rgb: np.ndarray) -> None:
        self._scene.clear()
        self._base_img = rgb
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._scene.addPixmap(pix).setZValue(0)
        self._mask_arr = np.zeros((h, w, 4), dtype=np.uint8)
        self._mask_item = self._scene.addPixmap(QPixmap(w, h))
        self._mask_item.setZValue(1)
        self.setSceneRect(QRectF(0, 0, w, h))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._update_cursor()

    def clear_mask(self) -> None:
        if self._mask_arr is not None:
            self._mask_arr.fill(0)
            self._refresh_mask()

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

    def _refresh_mask(self) -> None:
        if self._mask_arr is None or self._mask_item is None:
            return
        h, w = self._mask_arr.shape[:2]
        qimg = QImage(self._mask_arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._mask_item.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._mask_arr is not None:
            sp = self.mapToScene(ev.pos())
            h, w = self._mask_arr.shape[:2]
            if 0 <= sp.x() < w and 0 <= sp.y() < h:
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
        parent=None,
    ):
        super().__init__(parent)
        self._label    = label
        self._cmd_fn   = cmd_fn
        self._log      = log_widget
        self._accent   = accent_color
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

        # Create a dedicated log tab for this node
        self._node_log = QTextEdit()
        self._node_log.setReadOnly(True)
        self._node_log.setFont(QFont("Monospace", 9))
        self._log_tabs = getattr(self._log, "parent_tabs", None)
        if self._log_tabs is not None:
            self._log_tabs.addTab(self._node_log, label)

        self._btn = QPushButton("▶  Start")
        self._btn.setFixedWidth(90)
        self._btn.setStyleSheet(
            f"background:{C['panel']}; color:{C['text']};"
            " border:1px solid #585b70; border-radius:5px; padding:4px 8px;"
        )
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def stop(self) -> None:
        if self._worker:
            self._worker.abort()
            self._worker = None
        self._set_stopped()

    def _toggle(self) -> None:
        if self.is_running():
            self.stop()
        else:
            self._start()

    def _start(self) -> None:
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
        self._btn.setText("■  Stop")
        self._btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']};"
            " border:none; border-radius:5px; padding:4px 8px; font-weight:bold;"
        )
        self._status_dot.setStyleSheet(
            f"color:{self._accent}; font-size:14px;"
        )

    def _on_finished(self, rc: int) -> None:
        self._worker = None
        self._set_stopped()

    def _set_stopped(self) -> None:
        self._btn.setText("▶  Start")
        self._btn.setStyleSheet(
            f"background:{C['panel']}; color:{C['text']};"
            " border:1px solid #585b70; border-radius:5px; padding:4px 8px;"
        )
        self._status_dot.setStyleSheet(f"color:{C['border']}; font-size:14px;")



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
    roi_changed = Signal(tuple)

    def __init__(self, ros_node=None, bridge=None, parent=None):
        super().__init__(parent)
        self._ros_node = ros_node
        self._bridge   = bridge

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
            ros_node.create_subscription(
                ROSImage, "/vision/captured_image_raw", self._ros_cb, 1
            )
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._repaint)
            self._refresh_timer.start(100)

        self._latest_frame: Optional[np.ndarray] = None    # RGB numpy

    # ── ROS callback ─────────────────────────────────────────────────

    def _ros_cb(self, msg: "ROSImage") -> None:
        if not (CV2_OK and self._bridge):
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "rgb8")
            self._latest_frame = frame
            self._img_size = (frame.shape[1], frame.shape[0])
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
        qimg = QImage(small.data, nw, nh, ch * nw, QImage.Format_RGB888)
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

class VisionPipelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6  ·  Vision Pipeline Launcher")
        self.resize(1400, 860)

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

        self._build_ui()

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
        ros_badge = QLabel(
            "ROS 2 ✅" if ROS2_OK else "ROS 2 offline"
        )
        ros_badge.setStyleSheet(
            f"color:{'#a6e3a1' if ROS2_OK else '#f38ba8'}; font-size:11px;"
        )
        hdr_lay.addWidget(ros_badge)
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

        splitter.setSizes([340, 1060])

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        wrap = QScrollArea()
        wrap.setWidgetResizable(True)
        wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrap.setStyleSheet(f"QScrollArea{{background:{C['bg']}; border:none;}}")

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

        self._btn_live_cam = QPushButton("Live Kinect Camera")
        self._btn_live_cam.setStyleSheet(
            f"background:{C['blue']}; color:{C['bg']}; font-weight:bold; border-radius:6px; padding:6px;"
        )
        self._btn_live_cam.clicked.connect(self._start_live_cam)
        cg_lay.addWidget(self._btn_live_cam)

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
            self._log, C["green"],
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

        # Capture-mode start button note: also launch crop_image node
        self._btn_crop_node = NodeButton(
            "Crop Image Node",
            lambda: ["ros2", "run", "parol6_vision", "crop_image"],
            self._log, "#74c7ec",
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

        # Method selector (matches firmware configurator pattern)
        mi_lay.addWidget(QLabel("Launch Method:"))
        self._moveit_method = QComboBox()
        self._moveit_method.addItem("Method 3: Fake Hardware (Standalone RViz)", "fake")
        self._moveit_method.addItem("Method 4: Real Hardware + MoveIt", "real")
        mi_lay.addWidget(self._moveit_method)

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

        launch_vision_moveit_btn = QPushButton("🌐  Launch vision_moveit.launch.py")
        launch_vision_moveit_btn.setStyleSheet(
            f"background:#313244; color:{C['text']}; border:1px solid {C['border']};"
            " border-radius:6px; padding:5px;"
        )
        launch_vision_moveit_btn.clicked.connect(self._launch_vision_moveit)
        mi_lay.addWidget(launch_vision_moveit_btn)

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

        # ── Tab 1: Visual Outputs ─────────────────────────────────────────
        vis_tab = QWidget()
        vt_lay  = QGridLayout(vis_tab)
        vt_lay.setSpacing(4)

        previews = [
            ("/kinect2/qhd/image_color_rect",           "📷 Live Camera"),
            ("/vision/captured_image_color",            "📸 Captured Frame"),
            ("/vision/processing_mode/annotated_image", "🔍 Processing Output"),
            ("/path_optimizer/debug_image",             "📊 Path Optimizer Debug"),
        ]
        self._preview_panels = []
        for idx, (topic, title) in enumerate(previews):
            row, col = divmod(idx, 2)
            frame = QWidget()
            frame.setStyleSheet(
                f"background:{C['bg']}; border:1px solid {C['border']};"
                " border-radius:4px;"
            )
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.setSpacing(2)
            title_lbl = QLabel(f"  {title}")
            title_lbl.setStyleSheet(
                f"color:{C['text2']}; font-size:10px; font-weight:bold;"
                f" background:{C['panel']}; padding:3px;"
            )
            fl.addWidget(title_lbl)
            if ROS2_OK and CV2_OK:
                prev = TopicPreviewLabel(topic, self._ros_node, self._bridge)
                fl.addWidget(prev)
            else:
                lbl = QLabel(f"⌛  {topic}\n(ROS2 / cv2 offline)")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(f"color:{C['text2']}; font-size:10px;")
                fl.addWidget(lbl)
            self._preview_panels.append(frame)
            vt_lay.addWidget(frame, row, col)

        tabs.addTab(vis_tab, "👁  Visual Outputs")

        # ── Tab 2: Manual Red Line ─────────────────────────────────────────
        manual_tab = QWidget()
        mt_lay = QVBoxLayout(manual_tab)
        mt_lay.setContentsMargins(8, 8, 8, 8)

        ctrl_row = QHBoxLayout()
        load_btn = QPushButton("📂  Load Image")
        load_btn.clicked.connect(self._manual_load_image)
        ctrl_row.addWidget(load_btn)

        ctrl_row.addWidget(QLabel("Brush (px):"))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 150)
        self._brush_spin.setValue(5)
        self._brush_spin.valueChanged.connect(
            lambda v: self._canvas.set_brush(v) if CV2_OK else None
        )
        ctrl_row.addWidget(self._brush_spin)

        clear_btn = QPushButton("🗑  Clear")
        clear_btn.clicked.connect(lambda: self._canvas.clear_mask() if CV2_OK else None)
        ctrl_row.addWidget(clear_btn)

        save_btn = QPushButton("💾  Save Red-Line Image")
        save_btn.clicked.connect(self._manual_save)
        ctrl_row.addWidget(save_btn)

        ros_pub_btn = QPushButton("📡  Publish as ROS Image")
        ros_pub_btn.clicked.connect(self._manual_publish_ros)
        ctrl_row.addWidget(ros_pub_btn)

        ctrl_row.addStretch()
        mt_lay.addLayout(ctrl_row)

        if CV2_OK:
            self._canvas = ManualCanvas()
        else:
            self._canvas = QLabel("cv2 not available — Manual canvas disabled.")
            self._canvas.setAlignment(Qt.AlignCenter)
        mt_lay.addWidget(self._canvas)

        tabs.addTab(manual_tab, "✏️  Manual Red Line")

        # ── Tab 3: ROS Launch (firmware-configurator style) ────────────────
        ros_tab  = QWidget()
        rt_lay   = QVBoxLayout(ros_tab)
        rt_lay.setContentsMargins(12, 12, 12, 12)

        hint = QLabel(
            "📋 <b>Method 3</b> Fake Hardware — RViz + fake joints (path planning).  "
            "<b>Method 4</b> Real Hardware — live MoveIt execution.  "
            "▶️ <b>Run Auto-Test</b> injects a synthetic path and executes it to "
            "verify the full pipeline end-to-end.  "
            "<b>Inject Path</b> calls the moveit_controller service manually.  "
        )
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['accent']};"
            f" border-radius:6px; color:{C['text2']}; font-size:11px; padding:8px;"
        )
        rt_lay.addWidget(hint)

        ctrls_row = QHBoxLayout()

        self._ros_method = QComboBox()
        self._ros_method.addItem("Live Pipeline (no bag)",               "live")
        self._ros_method.addItem("Method 3: MoveIt Fake Hardware",        "fake")
        self._ros_method.addItem("Method 4: Real Hardware (MoveIt)",      "real")
        self._ros_method.addItem("Vision + MoveIt (vision_moveit.launch.py)", "vision_moveit")
        ctrls_row.addWidget(QLabel("Method:"))
        ctrls_row.addWidget(self._ros_method)

        self._ros_launch_btn = QPushButton("🚀  Launch")
        self._ros_launch_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        self._ros_launch_btn.clicked.connect(self._ros_launch_toggle)
        ctrls_row.addWidget(self._ros_launch_btn)

        kill_btn = QPushButton("☠️  Kill All")
        kill_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        kill_btn.clicked.connect(self._kill_all)
        ctrls_row.addWidget(kill_btn)

        ctrls_row.addSpacing(16)

        inject_btn = QPushButton("💉  Inject Test Path")
        inject_btn.setStyleSheet(
            f"background:{C['yellow']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        inject_btn.clicked.connect(self._inject_test_path)
        inject_btn.setToolTip(
            "Publishes a synthetic straight-line path on /vision/welding_path "
            "so the moveit_controller can execute it without the full camera pipeline."
        )
        ctrls_row.addWidget(inject_btn)

        send_moveit_btn = QPushButton("📡  Send Path → MoveIt")
        send_moveit_btn.setStyleSheet(
            f"background:{C['accent']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )
        send_moveit_btn.clicked.connect(self._send_path_to_moveit)
        ctrls_row.addWidget(send_moveit_btn)

        ctrls_row.addStretch()
        rt_lay.addLayout(ctrls_row)

        self._ros_log = QTextEdit()
        self._ros_log.setReadOnly(True)
        self._ros_log.setFont(QFont("Monospace", 9))
        rt_lay.addWidget(self._ros_log)

        self._ros_worker: Optional[NodeWorker] = None
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
            "📌 <b>Draw a rectangle</b> on the full-frame image (left) to select the crop region. "
            "Click <b>Apply & Save</b> to activate — every future captured image will be cropped "
            "automatically using the saved config. <b>Reset</b> restores pass-through."
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        info.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['accent']};"
            f" border-radius:6px; color:{C['text2']}; font-size:11px; padding:8px;"
        )
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
        lhdr.setStyleSheet(
            f"color:{C['text2']}; font-size:10px; font-weight:bold;"
            f" background:{C['panel']}; padding:3px;"
        )
        lf_lay.addWidget(lhdr)

        # CropROIView: shows ROS image with draggable ROI rectangle
        self._crop_view = CropROIView(self._ros_node, self._bridge)
        lf_lay.addWidget(self._crop_view)
        preview_row.addWidget(left_frame)

        # Right: cropped output
        right_frame = QWidget()
        right_frame.setStyleSheet(
            f"background:{C['bg']}; border:1px solid {C['border']}; border-radius:4px;"
        )
        rf_lay = QVBoxLayout(right_frame)
        rf_lay.setContentsMargins(0, 0, 0, 0)
        rf_lay.setSpacing(2)
        rhdr = QLabel("  ✂️  Cropped Output — /vision/captured_image_color")
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
        rf_lay.addWidget(self._crop_output_preview)
        preview_row.addWidget(right_frame)

        root_lay.addLayout(preview_row)

        # ── Bottom control bar ────────────────────────────────────────────
        ctrl_bar = QHBoxLayout()

        self._crop_status_lbl = QLabel("Status: No crop config (pass-through)")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")
        ctrl_bar.addWidget(self._crop_status_lbl)

        ctrl_bar.addStretch()

        # ROI readout
        self._roi_readout = QLabel("ROI: — ")
        self._roi_readout.setStyleSheet(
            f"color:{C['yellow']}; font-size:11px; font-weight:bold;"
        )
        ctrl_bar.addWidget(self._roi_readout)

        ctrl_bar.addSpacing(8)

        clear_roi_btn = QPushButton("🗑  Clear ROI")
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

    def _crop_load_existing_config(self) -> None:
        """Read ~/.parol6/crop_config.json and update status label."""
        import json
        from pathlib import Path
        cfg_path = Path.home() / ".parol6" / "crop_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if cfg.get("enabled") and "x" in cfg:
                    roi = (cfg["x"], cfg["y"], cfg["width"], cfg["height"])
                    self._crop_status_lbl.setText(
                        f"✅  Active config: ROI = {roi}"
                    )
                    self._crop_status_lbl.setStyleSheet(
                        f"color:{C['green']}; font-size:11px; font-weight:bold;"
                    )
                    self._roi_readout.setText(
                        f"ROI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}"
                    )
                    self._crop_view.set_saved_roi(roi)
                    return
            except Exception:
                pass
        self._crop_status_lbl.setText("Status: No crop config (pass-through)")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")

    def _on_roi_changed(self, roi: tuple) -> None:
        """Called when the user finishes drawing an ROI rectangle."""
        x, y, w, h = roi
        self._roi_readout.setText(f"ROI: x={x} y={y} w={w} h={h}")
        self._crop_status_lbl.setText(
            f"📐  Pending: ROI = ({x}, {y}, {w}, {h}) — click Apply & Save"
        )
        self._crop_status_lbl.setStyleSheet(
            f"color:{C['yellow']}; font-size:11px; font-weight:bold;"
        )

    def _crop_apply_save(self) -> None:
        """Save the current ROI to config and call the crop_image node's param."""
        import json
        from pathlib import Path

        roi = self._crop_view.current_roi()
        if roi is None:
            QMessageBox.warning(self, "No ROI", "Draw a rectangle on the image first.")
            return

        x, y, w, h = roi
        cfg_path = Path.home() / ".parol6" / "crop_config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(cfg_path, "w") as f:
            json.dump({"enabled": True, "x": x, "y": y, "width": w, "height": h}, f, indent=2)

        # Push ROI to running crop_image node via ros2 param set + reload_roi service
        param_worker = NodeWorker(
            ["ros2", "param", "set", "/crop_image", "roi", f"[{x},{y},{w},{h}]"],
        )
        param_worker.line_out.connect(self._log.append)
        param_worker.start()

        self._crop_status_lbl.setText(
            f"✅  Saved & Active: ROI = ({x}, {y}, {w}, {h})"
        )
        self._crop_status_lbl.setStyleSheet(
            f"color:{C['green']}; font-size:11px; font-weight:bold;"
        )
        self._log.append(
            f'<b style="color:{C["green"]}">[Crop] Config saved → {cfg_path}  ROI=({x},{y},{w},{h})</b>'
        )

    def _crop_clear_roi(self) -> None:
        """Clear the drawn ROI on the canvas without saving."""
        self._crop_view.clear_roi()
        self._roi_readout.setText("ROI: —")
        self._crop_status_lbl.setText("📐  ROI cleared — draw a new region")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")

    def _crop_reset(self) -> None:
        """Disable crop (pass-through) — call ~/clear_roi service and update config."""
        worker = NodeWorker(
            ["ros2", "service", "call", "/crop_image/clear_roi", "std_srvs/srv/Trigger", "{}"]
        )
        worker.line_out.connect(self._log.append)
        worker.start()

        self._crop_view.clear_roi()
        self._roi_readout.setText("ROI: —")
        self._crop_status_lbl.setText("↩  Reset: pass-through active (no crop).")
        self._crop_status_lbl.setStyleSheet(f"color:{C['text2']}; font-size:11px;")
        self._log.append(
            f'<span style="color:{C["text2"]}">[Crop] Reset → pass-through.</span>'
        )


    # ── Actions ───────────────────────────────────────────────────────────────

    def _start_live_cam(self) -> None:
        """Start Kinect camera. Worker stored to prevent thread GC crash."""
        self._live_cam_worker = NodeWorker(
            [
                "bash", "-c",
                "source /opt/ros/humble/setup.bash && "
                "source /opt/kinect_ws/install/setup.bash 2>/dev/null || true && "
                "ros2 launch kinect2_bridge kinect2_bridge_launch.yaml"
            ]
        )
        self._live_cam_worker.line_out.connect(self._log.append)
        self._live_cam_worker.start()

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
            "manual": None,   # handled by Manual tab
        }
        cmd = cmds.get(mode)
        if mode == "manual":
            self._log.append(
                '<span style="color:#f9e2af">[INFO] Use the ✏️ Manual Red Line tab '
                "to draw the path, then publish via '📡 Publish as ROS Image'.</span>"
            )
            return
        if not cmd:
            return
        if self._mode_worker and self._mode_worker.isRunning():
            self._stop_mode_node()
        self._mode_worker = NodeWorker(cmd)
        self._mode_worker.line_out.connect(
            lambda s: self._log.append(
                f'<span style="color:#89dceb">[Mode] {s}</span>'
            )
        )
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
            self._mode_worker = None
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    def _on_mode_finished(self, rc: int) -> None:
        self._mode_worker = None
        self._btn_start_mode.setEnabled(True)
        self._btn_stop_mode.setEnabled(False)

    def _launch_full_pipeline(self) -> None:
        self._btn_capture.stop()
        QTimer.singleShot(200, self._btn_capture._toggle)
        QTimer.singleShot(600, self._start_mode_node)
        QTimer.singleShot(1200, self._btn_optimizer._toggle)
        QTimer.singleShot(1600, self._btn_depth._toggle)
        QTimer.singleShot(2000, self._btn_pathgen._toggle)

    def _stop_all(self) -> None:
        for btn in (self._btn_live_cam, self._btn_capture,
                    self._btn_optimizer, self._btn_depth,
                    self._btn_pathgen, self._btn_moveit):
            btn.stop()
        self._stop_mode_node()
        if self._ros_worker:
            self._ros_worker.abort()
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
            self._ros_worker.abort()
            self._ros_worker = None
            self._ros_launch_btn.setText("🚀  Launch")
            self._ros_launch_btn.setStyleSheet(
                f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
                " border:none; border-radius:6px; padding:5px 14px;"
            )
            return

        method = self._ros_method.currentData()
        method_cmds = {
            "live":  ["ros2", "launch", "parol6_vision", "live_pipeline.launch.py"],
            "fake":  ["ros2", "launch", "parol6_vision", "vision_pipeline.launch.py"],
            "real":  ["ros2", "launch", "parol6_vision", "vision_moveit.launch.py",
                      "use_bag:=false"],
            "vision_moveit": ["ros2", "launch", "parol6_vision",
                               "vision_moveit.launch.py"],
        }
        cmd = method_cmds.get(method, method_cmds["fake"])

        self._ros_log.clear()
        self._ros_worker = NodeWorker(cmd)
        self._ros_worker.line_out.connect(self._ros_log.append)
        self._ros_worker.finished.connect(self._on_ros_launch_finished)
        self._ros_worker.start()

        self._ros_launch_btn.setText("■  Stop")
        self._ros_launch_btn.setStyleSheet(
            f"background:{C['red']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )

    def _on_ros_launch_finished(self, rc: int) -> None:
        self._ros_worker = None
        self._ros_launch_btn.setText("🚀  Launch")
        self._ros_launch_btn.setStyleSheet(
            f"background:{C['green']}; color:{C['bg']}; font-weight:bold;"
            " border:none; border-radius:6px; padding:5px 14px;"
        )

    def _inject_test_path(self) -> None:
        """Publish a small synthetic straight-line path to /vision/welding_path."""
        yaml_path = """
poses:
  - header:
      frame_id: base_link
    pose:
      position: {x: 0.4, y: -0.1, z: 0.45}
      orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}
  - header:
      frame_id: base_link
    pose:
      position: {x: 0.4, y: 0.0, z: 0.45}
      orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}
  - header:
      frame_id: base_link
    pose:
      position: {x: 0.4, y: 0.1, z: 0.45}
      orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}
""".strip()

        worker = NodeWorker(
            ["ros2", "topic", "pub", "--once",
             "/vision/welding_path", "nav_msgs/msg/Path",
             f"{{header: {{frame_id: base_link}}, {yaml_path}}}"],
        )
        worker.line_out.connect(
            lambda s: self._ros_log.append(
                f'<span style="color:{C["yellow"]}">[Inject] {s}</span>'
            )
        )
        worker.start()
        self._ros_log.append(
            f'<b style="color:{C["yellow"]}">[Inject] Synthetic path published → '
            '/vision/welding_path</b>'
        )

    def _kill_all(self) -> None:
        self._stop_all()
        cmd = "pkill -9 -f 'rviz2|move_group|ros2_control_node|ros2 run parol6'"
        subprocess.run(["bash", "-c", cmd], check=False)
        self._ros_log.append("[KILL] All ROS 2 processes terminated.")

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
        """Publish the annotated canvas image to /vision/processing_mode/annotated_image."""
        if not (ROS2_OK and CV2_OK):
            QMessageBox.warning(self, "Unavailable", "ROS 2 + cv2 required.")
            return
        img = self._canvas.get_annotated_bgr()
        if img is None:
            QMessageBox.warning(self, "Nothing to Publish", "Draw red lines first.")
            return
        try:
            msg = self._bridge.cv2_to_imgmsg(img, encoding="bgr8")
            pub = self._ros_node.create_publisher(
                ROSImage, "/vision/processing_mode/annotated_image", 1
            )
            pub.publish(msg)
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
