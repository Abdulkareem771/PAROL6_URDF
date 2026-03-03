#!/usr/bin/env python3
"""
capture_gui.py — PAROL6 Vision Capture Testing GUI
====================================================

Standalone script — no installation needed. Run with:

    python3 /workspace/src/parol6_vision/scripts/capture_gui.py

(ROS2 must be sourced in the same shell first.)

Workflow
--------
1. Preview the live camera stream (QHD topic).
2. Click [📷 Capture Frame] — grabs one frame from the live camera
   and publishes it to /capture_gui/frozen_frame.  The red_line_detector
   (running in capture_mode=True) is remapped to listen to that topic,
   so it processes ONLY that one frozen image per click.
3. The detection overlay (/red_line_detector/debug_image) appears in
   the Detection Result tab, with per-line stats in the table below.
4. Click [🔁 Re-detect] to re-run detection on the same frozen frame
   (useful when tuning HSV params without moving the camera).
5. Click [🔍 Match Depth] to run one depth-matching cycle → publishes
   /vision/weld_lines_3d.
6. Use the Node Controls panel to Start/Stop the detector and
   depth_matcher from inside the GUI (no extra terminal needed).

Requirements
------------
    PyQt6       (apt install python3-pyqt6  OR  pip3 install PyQt6)
    rclpy, cv_bridge, OpenCV (already in Docker environment)
    parol6_vision built and sourced
"""

import sys
import os
import time
import subprocess
import threading
import datetime
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from parol6_msgs.msg import WeldLineArray, WeldLine3DArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import cv2

# ── Qt auto-detect: tries PyQt6 → PyQt5 → PySide6 ──────────────────
# Normalises the small API differences so the rest of the code is
# identical regardless of which binding is installed.

def _import_qt():
    """Return (QtWidgets, QtCore, QtGui, Signal, is_pyqt) for whichever Qt is available."""

    # ── PyQt6 ────────────────────────────────────────────────────────
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui
        # PyQt6 uses pyqtSignal; enums are namespaced (Qt.AlignmentFlag.X)
        return QtWidgets, QtCore, QtGui, QtCore.pyqtSignal, True, 'PyQt6'
    except ImportError:
        pass

    # ── PyQt5 ────────────────────────────────────────────────────────
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui
        # PyQt5 uses pyqtSignal; enums are on Qt directly (Qt.AlignCenter)
        # Patch enums to match PyQt6 namespacing so code below works unchanged
        Qt = QtCore.Qt
        Qt.AlignmentFlag    = Qt
        Qt.Orientation      = Qt
        Qt.AspectRatioMode  = Qt
        Qt.TransformationMode = Qt
        QtWidgets.QHeaderView.ResizeMode   = QtWidgets.QHeaderView
        QtWidgets.QAbstractItemView.EditTrigger = QtWidgets.QAbstractItemView
        QtGui.QImage.Format = QtGui.QImage
        # exec_ → exec compatibility
        QtWidgets.QApplication.exec = QtWidgets.QApplication.exec_
        return QtWidgets, QtCore, QtGui, QtCore.pyqtSignal, True, 'PyQt5'
    except ImportError:
        pass

    # ── PySide6 ──────────────────────────────────────────────────────
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        return QtWidgets, QtCore, QtGui, QtCore.Signal, False, 'PySide6'
    except ImportError:
        pass

    raise ImportError(
        'No supported Qt binding found.\n'
        'Install one of:  apt install python3-pyqt5  |  pip3 install PyQt6  |  pip3 install PySide6'
    )


_QtWidgets, _QtCore, _QtGui, _Signal, _IS_PYQT, _QT_BINDING = _import_qt()
print(f'[capture_gui] Using {_QT_BINDING}')

# Aliases used throughout the file
from typing import TYPE_CHECKING
QApplication    = _QtWidgets.QApplication
QMainWindow     = _QtWidgets.QMainWindow
QWidget         = _QtWidgets.QWidget
QHBoxLayout     = _QtWidgets.QHBoxLayout
QVBoxLayout     = _QtWidgets.QVBoxLayout
QPushButton     = _QtWidgets.QPushButton
QLabel          = _QtWidgets.QLabel
QTextEdit       = _QtWidgets.QTextEdit
QGroupBox       = _QtWidgets.QGroupBox
QSplitter       = _QtWidgets.QSplitter
QProgressBar    = _QtWidgets.QProgressBar
QTabWidget      = _QtWidgets.QTabWidget
QTableWidget    = _QtWidgets.QTableWidget
QTableWidgetItem= _QtWidgets.QTableWidgetItem
QHeaderView     = _QtWidgets.QHeaderView
QFileDialog     = _QtWidgets.QFileDialog
QAbstractItemView = _QtWidgets.QAbstractItemView
Qt              = _QtCore.Qt
QTimer          = _QtCore.QTimer
QObject         = _QtCore.QObject
pyqtSignal      = _Signal          # works for QObject subclasses
QImage          = _QtGui.QImage
QPixmap         = _QtGui.QPixmap
QColor          = _QtGui.QColor
QFont           = _QtGui.QFont



# ===================================================================
#  SIGNAL BRIDGE — moves data from ROS callbacks → Qt main thread
# ===================================================================

class ROSSignals(QObject):
    camera_frame = pyqtSignal(object)    # BGR ndarray
    debug_frame  = pyqtSignal(object)    # BGR ndarray
    lines_2d     = pyqtSignal(object)    # WeldLineArray
    lines_3d     = pyqtSignal(object)    # WeldLine3DArray
    log_message  = pyqtSignal(str)
    hz_update    = pyqtSignal(str, float)


# ===================================================================
#  ROS2 BACKGROUND NODE
# ===================================================================

class CaptureGUINode(Node):
    """
    Background ROS2 node.

    - Subscribes to live camera/debug/result topics.
    - Publishes frozen frames to /capture_gui/frozen_frame.
    - Calls /red_line_detector/capture and /depth_matcher/capture services.
    """

    FROZEN_TOPIC = '/capture_gui/frozen_frame'

    def __init__(self, signals: ROSSignals):
        super().__init__('capture_gui')
        self.signals = signals
        self.bridge  = CvBridge()

        self._waiting_capture = False
        self._frozen_frame: np.ndarray | None = None

        self._hz_stamps: dict[str, list] = {
            'camera': [], 'debug': [], 'lines2d': [], 'lines3d': []
        }

        # Publishers
        self.frozen_pub = self.create_publisher(Image, self.FROZEN_TOPIC, 1)

        # Subscribers
        self.create_subscription(Image, '/kinect2/qhd/image_color_rect', self._cb_camera, 5)
        self.create_subscription(Image, '/red_line_detector/debug_image',  self._cb_debug,  5)
        self.create_subscription(WeldLineArray,    '/vision/weld_lines_2d', self._cb_lines2d, 5)
        self.create_subscription(WeldLine3DArray,  '/vision/weld_lines_3d', self._cb_lines3d, 5)

        # Service clients
        self.detect_client = self.create_client(Trigger, '/red_line_detector/capture')
        self.depth_client  = self.create_client(Trigger, '/depth_matcher/capture')

        # Hz timer
        self.create_timer(1.0, self._emit_hz)
        self.get_logger().info('CaptureGUINode ready')

    # ── Camera ───────────────────────────────────────────────────────

    def _cb_camera(self, msg: Image):
        self._track_hz('camera')
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        if self._waiting_capture:
            self._waiting_capture = False
            self._frozen_frame = bgr.copy()
            try:
                frozen_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                frozen_msg.header = msg.header
                self.frozen_pub.publish(frozen_msg)
                self.signals.log_message.emit(
                    f'[CAPTURE] Frame grabbed '
                    f'({bgr.shape[1]}×{bgr.shape[0]}) → {self.FROZEN_TOPIC}'
                )
            except Exception as e:
                self.signals.log_message.emit(f'[ERROR] Could not publish frozen frame: {e}')

        self.signals.camera_frame.emit(bgr)

    def _cb_debug(self, msg: Image):
        self._track_hz('debug')
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.signals.debug_frame.emit(bgr)
        except Exception:
            pass

    def _cb_lines2d(self, msg: WeldLineArray):
        self._track_hz('lines2d')
        self.signals.lines_2d.emit(msg)

    def _cb_lines3d(self, msg: WeldLine3DArray):
        self._track_hz('lines3d')
        self.signals.lines_3d.emit(msg)

    # ── Public API ───────────────────────────────────────────────────

    def request_capture(self):
        self._waiting_capture = True
        self.signals.log_message.emit('[INFO] Waiting for next camera frame...')

    def republish_frozen(self):
        """Publish the stored frozen frame again for re-detection."""
        if self._frozen_frame is None:
            return False
        try:
            msg = self.bridge.cv2_to_imgmsg(self._frozen_frame, encoding='bgr8')
            self.frozen_pub.publish(msg)
            self.signals.log_message.emit('[INFO] Re-published frozen frame.')
            return True
        except Exception as e:
            self.signals.log_message.emit(f'[ERROR] {e}')
            return False

    def call_detect_service(self, callback):
        if not self.detect_client.service_is_ready():
            self.signals.log_message.emit(
                '[WARN] /red_line_detector/capture not available — '
                'is the detector running in capture_mode?'
            )
            callback(None)
            return
        fut = self.detect_client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: callback(f.result()))

    def call_depth_service(self, callback):
        if not self.depth_client.service_is_ready():
            self.signals.log_message.emit(
                '[WARN] /depth_matcher/capture not available — '
                'is depth_matcher running in capture_mode?'
            )
            callback(None)
            return
        fut = self.depth_client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: callback(f.result()))

    def get_frozen_frame(self) -> np.ndarray | None:
        return self._frozen_frame

    # ── Hz tracking ──────────────────────────────────────────────────

    def _track_hz(self, key: str):
        now = time.monotonic()
        self._hz_stamps[key].append(now)
        cutoff = now - 2.0
        self._hz_stamps[key] = [t for t in self._hz_stamps[key] if t >= cutoff]

    def _emit_hz(self):
        for key, stamps in self._hz_stamps.items():
            self.signals.hz_update.emit(key, len(stamps) / 2.0)


# ===================================================================
#  SUBPROCESS NODE MANAGER
# ===================================================================

class NodeProcess:
    """Launches a ros2 run subprocess and pipes its stdout to a log callback."""

    def __init__(self, name: str, package: str, executable: str,
                 ros_args: list[str], log_cb):
        self.name       = name
        self.package    = package
        self.executable = executable
        self.ros_args   = ros_args
        self.log_cb     = log_cb
        self._proc      = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self):
        if self.running:
            return
        cmd = ['ros2', 'run', self.package, self.executable, '--ros-args'] + self.ros_args
        self.log_cb(f'[NODE] $ {" ".join(cmd)}')
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            threading.Thread(target=self._stream, daemon=True).start()
        except FileNotFoundError as e:
            self.log_cb(f'[ERROR] {e} — is ROS2 sourced?')

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self.log_cb(f'[NODE] {self.name} stopped.')
        self._proc = None

    def _stream(self):
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                self.log_cb(f'[{self.name}] {line}')


# ===================================================================
#  HELPERS
# ===================================================================

def bgr_to_qpixmap(bgr: np.ndarray, max_w: int = 800, max_h: int = 500) -> QPixmap:
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        max_w, max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def led(color: str = 'gray') -> QLabel:
    lbl = QLabel('●')
    lbl.setStyleSheet(f'color: {color}; font-size: 18px;')
    return lbl


# ===================================================================
#  MAIN WINDOW
# ===================================================================

class CaptureWindow(QMainWindow):

    def __init__(self, ros_node: CaptureGUINode):
        super().__init__()
        self.node = ros_node
        self.sigs = ros_node.signals
        self._capture_count = 0

        # Managed subprocesses — detector remapped to frozen topic
        self._det_proc = NodeProcess(
            name='Detector', package='parol6_vision', executable='red_line_detector',
            ros_args=[
                '-p', 'capture_mode:=true',
                '-p', 'publish_debug_images:=true',
                '--remap',
                f'/kinect2/qhd/image_color_rect:={CaptureGUINode.FROZEN_TOPIC}',
            ],
            log_cb=self._log,
        )
        self._dm_proc = NodeProcess(
            name='DepthMatcher', package='parol6_vision', executable='depth_matcher',
            ros_args=['-p', 'capture_mode:=true'],
            log_cb=self._log,
        )

        self._build_ui()
        self._connect_signals()

        # Drive rclpy.spin_once on the Qt timer so callbacks fire
        self._ros_timer = QTimer(self)
        self._ros_timer.setInterval(30)
        self._ros_timer.timeout.connect(lambda: rclpy.spin_once(self.node, timeout_sec=0.0))
        self._ros_timer.start()

        # Node health poll
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(1000)
        self._health_timer.timeout.connect(self._poll_health)
        self._health_timer.start()

        self.setWindowTitle('PAROL6 Vision — Capture Testing GUI')
        self.resize(1400, 860)

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel('🤖  PAROL6 Vision — Capture Testing GUI')
        title.setStyleSheet(
            'font-size:16px; font-weight:bold; color:#e0e0e0;'
            'background:#1e1e2e; padding:6px 12px; border-radius:4px;'
        )
        root.addWidget(title)

        # Hz row
        hz_row = QHBoxLayout()
        self._hz_widgets: dict[str, tuple[QLabel, QProgressBar]] = {}
        for label, key in [('📷 Camera','camera'),('🔍 Debug','debug'),
                            ('📐 2D Lines','lines2d'),('🧊 3D Lines','lines3d')]:
            box = QGroupBox(label)
            box.setFixedWidth(165)
            bl  = QVBoxLayout(box)
            l   = led('gray')
            b   = QProgressBar()
            b.setRange(0, 35); b.setValue(0)
            b.setTextVisible(True); b.setFormat('0 Hz'); b.setFixedHeight(14)
            bl.addWidget(l, alignment=Qt.AlignmentFlag.AlignCenter)
            bl.addWidget(b)
            self._hz_widgets[key] = (l, b)
            hz_row.addWidget(box)
        hz_row.addStretch()
        root.addLayout(hz_row)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── Left panel ───────────────────────────────────────────────
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setSpacing(6)

        # Node controls
        nc = QGroupBox('Node Controls')
        nc_l = QVBoxLayout(nc)

        self._btn_start_det = QPushButton('▶  Start Detector')
        self._btn_stop_det  = QPushButton('■  Stop Detector')
        self._btn_stop_det.setEnabled(False)
        self._btn_start_dm  = QPushButton('▶  Start Depth Matcher')
        self._btn_stop_dm   = QPushButton('■  Stop Depth Matcher')
        self._btn_stop_dm.setEnabled(False)

        self._led_det = led('gray')
        self._led_dm  = led('gray')

        for btns, l in [
            ([self._btn_start_det, self._btn_stop_det], self._led_det),
            ([self._btn_start_dm,  self._btn_stop_dm],  self._led_dm),
        ]:
            row = QHBoxLayout()
            for b in btns:
                row.addWidget(b)
            row.addWidget(l)
            nc_l.addLayout(row)
        lv.addWidget(nc)

        # Capture controls
        wf = QGroupBox('Capture & Detection')
        wf_l = QVBoxLayout(wf)
        self._btn_capture  = QPushButton('📷  Capture Frame')
        self._btn_redetect = QPushButton('🔁  Re-detect (same frame)')
        self._btn_match    = QPushButton('🔍  Match Depth')
        self._btn_save     = QPushButton('💾  Save Frozen Frame')
        self._lbl_cap_info = QLabel('No frame captured yet.')
        self._lbl_cap_info.setWordWrap(True)
        self._lbl_cap_info.setStyleSheet('color:#aaaaaa; font-size:11px;')
        self._btn_redetect.setEnabled(False)
        self._btn_match.setEnabled(False)
        self._btn_save.setEnabled(False)
        for w in [self._btn_capture, self._btn_redetect,
                  self._btn_match, self._btn_save, self._lbl_cap_info]:
            wf_l.addWidget(w)
        lv.addWidget(wf)

        # Stats table
        sg = QGroupBox('Detection Stats (2D Lines)')
        sg_l = QVBoxLayout(sg)
        self._stats_tbl = QTableWidget(0, 4)
        self._stats_tbl.setHorizontalHeaderLabels(['ID','Confidence','Pixels','BBox'])
        self._stats_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._stats_tbl.setFixedHeight(110)
        self._stats_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sg_l.addWidget(self._stats_tbl)
        lv.addWidget(sg)

        # 3D result
        rg = QGroupBox('3D Result')
        rg_l = QVBoxLayout(rg)
        self._lbl_3d = QLabel('No 3D data yet.')
        self._lbl_3d.setWordWrap(True)
        rg_l.addWidget(self._lbl_3d)
        lv.addWidget(rg)

        # Log
        lg = QGroupBox('Log')
        lg_l = QVBoxLayout(lg)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont('Monospace', 9))
        self._log_view.setStyleSheet('background:#111122; color:#ccddee;')
        btn_clr = QPushButton('🗑  Clear')
        btn_clr.setFixedHeight(22)
        btn_clr.clicked.connect(self._log_view.clear)
        lg_l.addWidget(self._log_view)
        lg_l.addWidget(btn_clr)
        lv.addWidget(lg, stretch=1)

        splitter.addWidget(left)

        # ── Right panel — image tabs ──────────────────────────────────
        right = QWidget()
        rv    = QVBoxLayout(right)
        tabs  = QTabWidget()

        def img_label(placeholder: str) -> QLabel:
            l = QLabel(placeholder)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet('background:#0a0a1a; color:#888;')
            l.setMinimumSize(640, 360)
            return l

        self._lbl_live   = img_label('Waiting for camera...')
        self._lbl_frozen = img_label('No frame captured.')
        self._lbl_debug  = img_label('No detection result yet.')
        tabs.addTab(self._lbl_live,   '📷 Live Preview')
        tabs.addTab(self._lbl_frozen, '🧊 Frozen Frame')
        tabs.addTab(self._lbl_debug,  '🔍 Detection Result')
        self._tabs = tabs

        rv.addWidget(tabs)
        splitter.addWidget(right)
        splitter.setSizes([380, 1020])

        # Status bar
        sb = self.statusBar()
        self._sb_det = QLabel('Detector: ⬛ Stopped')
        self._sb_dm  = QLabel('Depth Matcher: ⬛ Stopped')
        self._sb_cap = QLabel('Captures: 0')
        sb.addPermanentWidget(self._sb_det)
        sb.addPermanentWidget(QLabel(' | '))
        sb.addPermanentWidget(self._sb_dm)
        sb.addPermanentWidget(QLabel(' | '))
        sb.addPermanentWidget(self._sb_cap)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color:#1a1a2e; color:#e0e0e0; }
            QGroupBox { border:1px solid #3a3a5c; border-radius:4px;
                        margin-top:8px; font-weight:bold; color:#aaaacc; }
            QGroupBox::title { subcontrol-origin:margin; padding:0 4px; }
            QPushButton { background:#2a2a4a; color:#d0d0f0;
                          border:1px solid #4a4a7a; border-radius:4px;
                          padding:5px 10px; font-size:12px; }
            QPushButton:hover   { background:#3a3a6a; }
            QPushButton:pressed { background:#1a1a3a; }
            QPushButton:disabled{ color:#555566; border-color:#333355; }
            QTabWidget::pane { border:1px solid #3a3a5c; }
            QTabBar::tab { background:#222244; color:#aaaacc;
                           padding:6px 14px; border-radius:3px; }
            QTabBar::tab:selected { background:#3a3a6a; color:#e0e0ff; }
            QTableWidget { background:#0d0d1e; color:#c0d0e0;
                           gridline-color:#2a2a4a; }
            QHeaderView::section { background:#222244; color:#aaaacc;
                                   border:none; padding:3px; }
            QProgressBar { text-align:center; border:1px solid #3a3a5c;
                           border-radius:3px; background:#111122; color:#e0e0e0; }
            QProgressBar::chunk { background:#3a6a3a; }
        """)

    # ── Signal connections ────────────────────────────────────────────

    def _connect_signals(self):
        self.sigs.camera_frame.connect(self._on_camera)
        self.sigs.debug_frame.connect(self._on_debug)
        self.sigs.lines_2d.connect(self._on_lines2d)
        self.sigs.lines_3d.connect(self._on_lines3d)
        self.sigs.log_message.connect(self._log)
        self.sigs.hz_update.connect(self._on_hz)

        self._btn_start_det.clicked.connect(self._start_det)
        self._btn_stop_det.clicked.connect(self._stop_det)
        self._btn_start_dm.clicked.connect(self._start_dm)
        self._btn_stop_dm.clicked.connect(self._stop_dm)

        self._btn_capture.clicked.connect(self._do_capture)
        self._btn_redetect.clicked.connect(self._do_redetect)
        self._btn_match.clicked.connect(self._do_match)
        self._btn_save.clicked.connect(self._do_save)

    # ── Node control ─────────────────────────────────────────────────

    def _start_det(self):
        self._det_proc.start()
        self._btn_start_det.setEnabled(False)
        self._btn_stop_det.setEnabled(True)

    def _stop_det(self):
        self._det_proc.stop()
        self._btn_start_det.setEnabled(True)
        self._btn_stop_det.setEnabled(False)

    def _start_dm(self):
        self._dm_proc.start()
        self._btn_start_dm.setEnabled(False)
        self._btn_stop_dm.setEnabled(True)

    def _stop_dm(self):
        self._dm_proc.stop()
        self._btn_start_dm.setEnabled(True)
        self._btn_stop_dm.setEnabled(False)

    def _poll_health(self):
        for proc, led_w, sb_lbl, start_btn, stop_btn, label in [
            (self._det_proc, self._led_det, self._sb_det,
             self._btn_start_det, self._btn_stop_det, 'Detector'),
            (self._dm_proc,  self._led_dm,  self._sb_dm,
             self._btn_start_dm,  self._btn_stop_dm,  'Depth Matcher'),
        ]:
            if proc.running:
                led_w.setStyleSheet('color:#00ff88; font-size:18px;')
                sb_lbl.setText(f'{label}: 🟢 Running')
            else:
                led_w.setStyleSheet('color:#666677; font-size:18px;')
                sb_lbl.setText(f'{label}: ⬛ Stopped')
                if not start_btn.isEnabled():
                    start_btn.setEnabled(True)
                    stop_btn.setEnabled(False)

    # ── Capture workflow ──────────────────────────────────────────────

    def _do_capture(self):
        self._btn_capture.setEnabled(False)
        self.node.request_capture()
        # Wait 400 ms for the frame to be grabbed then call detect service
        QTimer.singleShot(400, self._call_detect)

    def _call_detect(self):
        self._log('[GUI] Calling /red_line_detector/capture ...')
        self.node.call_detect_service(self._on_detect_response)

    def _on_detect_response(self, resp):
        self._btn_capture.setEnabled(True)
        if resp is None:
            self._log('[ERROR] Detect service unavailable.')
            return
        color = '#88ffaa' if resp.success else '#ff6666'
        self._lbl_cap_info.setText(('✅ ' if resp.success else '❌ ') + resp.message)
        self._lbl_cap_info.setStyleSheet(f'color:{color}; font-size:11px;')
        if resp.success:
            self._capture_count += 1
            self._sb_cap.setText(f'Captures: {self._capture_count}')
            self._btn_redetect.setEnabled(True)
            self._btn_match.setEnabled(True)
            self._btn_save.setEnabled(True)
            self._tabs.setCurrentIndex(2)
        self._log(f'[DETECT] success={resp.success} — {resp.message}')

    def _do_redetect(self):
        if not self.node.republish_frozen():
            return
        QTimer.singleShot(150, self._call_detect)

    def _do_match(self):
        self._btn_match.setEnabled(False)
        self._log('[GUI] Calling /depth_matcher/capture ...')
        self.node.call_depth_service(self._on_depth_response)

    def _on_depth_response(self, resp):
        self._btn_match.setEnabled(True)
        if resp is None:
            self._log('[ERROR] Depth matcher service unavailable.')
            return
        self._log(f'[DEPTH] success={resp.success} — {resp.message}')

    def _do_save(self):
        frozen = self.node.get_frozen_frame()
        if frozen is None:
            return
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Frozen Frame', f'capture_{ts}.png', 'PNG (*.png)')
        if path:
            cv2.imwrite(path, frozen)
            self._log(f'[GUI] Saved → {path}')

    # ── ROS signal handlers ───────────────────────────────────────────

    def _on_camera(self, bgr: np.ndarray):
        self._lbl_live.setPixmap(bgr_to_qpixmap(bgr))
        frozen = self.node.get_frozen_frame()
        if frozen is not None:
            self._lbl_frozen.setPixmap(bgr_to_qpixmap(frozen))

    def _on_debug(self, bgr: np.ndarray):
        self._lbl_debug.setPixmap(bgr_to_qpixmap(bgr))

    def _on_lines2d(self, msg: WeldLineArray):
        self._stats_tbl.setRowCount(0)
        for line in msg.lines:
            row = self._stats_tbl.rowCount()
            self._stats_tbl.insertRow(row)
            c = float(line.confidence)
            color = '#88ff88' if c >= 0.9 else ('#ffff88' if c >= 0.7 else '#ffaa44')
            bbox = (f'({line.bbox_min.x:.0f},{line.bbox_min.y:.0f})'
                    f'→({line.bbox_max.x:.0f},{line.bbox_max.y:.0f})')
            for col, (val, clr) in enumerate([
                (line.id, '#c0d0e0'),
                (f'{c:.3f}', color),
                (str(len(line.pixels)), '#c0d0e0'),
                (bbox, '#c0d0e0'),
            ]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(clr))
                self._stats_tbl.setItem(row, col, item)

    def _on_lines3d(self, msg: WeldLine3DArray):
        n = len(msg.lines)
        if n == 0:
            self._lbl_3d.setText('No 3D lines received.')
            return
        info = '\n'.join(
            f'  • {l.id}: {len(l.points)} pts, conf={l.confidence:.2f}, '
            f'depth_qual={l.depth_quality:.2f}'
            for l in msg.lines
        )
        self._lbl_3d.setText(f'✅ {n} 3D line(s):\n{info}')
        self._lbl_3d.setStyleSheet('color:#88ffaa;')

    def _on_hz(self, key: str, hz: float):
        if key not in self._hz_widgets:
            return
        l, b = self._hz_widgets[key]
        b.setValue(int(min(hz, 35)))
        b.setFormat(f'{hz:.1f} Hz')
        l.setStyleSheet(
            f'color:{"#00ff88" if hz >= 20 else "#ffee44" if hz >= 5 else "#ff8844" if hz > 0 else "#555566"};'
            'font-size:18px;'
        )

    # ── Log ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        sb = self._log_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 10
        self._log_view.append(f'[{ts}] {msg}')
        if at_bottom:
            sb.setValue(sb.maximum())

    # ── Close ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._ros_timer.stop()
        self._health_timer.stop()
        self._det_proc.stop()
        self._dm_proc.stop()
        event.accept()


# ===================================================================
#  ENTRY POINT
# ===================================================================

def main():
    """
    Run as a plain Python script:

        python3 scripts/capture_gui.py
    """
    rclpy.init()

    signals  = ROSSignals()
    ros_node = CaptureGUINode(signals)

    # ROS executor in background thread
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(ros_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # Qt on the main thread
    app    = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CaptureWindow(ros_node)
    window.show()

    exit_code = app.exec()   # PyQt6: exec() not exec_()

    executor.shutdown(timeout_sec=2.0)
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
