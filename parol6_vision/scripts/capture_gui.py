#!/usr/bin/env python3
"""
capture_gui.py — PAROL6 Vision Capture & Pipeline Testing GUI
==============================================================

Standalone script. Run with:
    python3 parol6_vision/scripts/capture_gui.py

Full pipeline:
    Camera → red_line_detector → depth_matcher → path_generator → moveit_controller

Each stage can be triggered individually or via "🚀 Run Full Pipeline".

Requirements:
    PyQt6 / PyQt5 / PySide6  (auto-detected)
    rclpy, cv_bridge, OpenCV  (in Docker ROS2 env)
    parol6_vision built and sourced
"""

import sys, os, time, subprocess, threading, datetime
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from parol6_msgs.msg import WeldLineArray, WeldLine3DArray
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2

# ── Qt auto-detect ───────────────────────────────────────────────────

def _import_qt():
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui
        return QtWidgets, QtCore, QtGui, QtCore.pyqtSignal, 'PyQt6'
    except ImportError:
        pass
    try:
        from PyQt5 import QtWidgets, QtCore, QtGui
        Qt = QtCore.Qt
        Qt.AlignmentFlag = Qt
        Qt.Orientation = Qt
        Qt.AspectRatioMode = Qt
        Qt.TransformationMode = Qt
        Qt.ScrollBarPolicy = Qt
        QtWidgets.QHeaderView.ResizeMode = QtWidgets.QHeaderView
        QtWidgets.QAbstractItemView.EditTrigger = QtWidgets.QAbstractItemView
        QtGui.QImage.Format = QtGui.QImage
        QtWidgets.QApplication.exec = QtWidgets.QApplication.exec_
        return QtWidgets, QtCore, QtGui, QtCore.pyqtSignal, 'PyQt5'
    except ImportError:
        pass
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        return QtWidgets, QtCore, QtGui, QtCore.Signal, 'PySide6'
    except ImportError:
        pass
    raise ImportError(
        'No Qt binding found.\n'
        'Run:  pip3 install PyQt6   OR   apt install python3-pyqt5'
    )


_W, _C, _G, _Signal, _QT = _import_qt()
print(f'[capture_gui] Using {_QT}')

# Class aliases
QApplication     = _W.QApplication
QMainWindow      = _W.QMainWindow
QWidget          = _W.QWidget
QHBoxLayout      = _W.QHBoxLayout
QVBoxLayout      = _W.QVBoxLayout
QPushButton      = _W.QPushButton
QLabel           = _W.QLabel
QTextEdit        = _W.QTextEdit
QGroupBox        = _W.QGroupBox
QSplitter        = _W.QSplitter
QProgressBar     = _W.QProgressBar
QTabWidget       = _W.QTabWidget
QTableWidget     = _W.QTableWidget
QTableWidgetItem = _W.QTableWidgetItem
QHeaderView      = _W.QHeaderView
QFileDialog      = _W.QFileDialog
QAbstractItemView= _W.QAbstractItemView
QScrollArea      = _W.QScrollArea
QCheckBox        = _W.QCheckBox
Qt               = _C.Qt
QTimer           = _C.QTimer
QObject          = _C.QObject
pyqtSignal       = _Signal
QImage           = _G.QImage
QPixmap          = _G.QPixmap
QColor           = _G.QColor
QFont            = _G.QFont


# ===================================================================
#  SIGNAL BRIDGE
# ===================================================================

class ROSSignals(QObject):
    camera_frame = pyqtSignal(object)
    debug_frame  = pyqtSignal(object)
    lines_2d     = pyqtSignal(object)
    lines_3d     = pyqtSignal(object)
    path_received= pyqtSignal(object)   # nav_msgs/Path
    log_message  = pyqtSignal(str)
    hz_update    = pyqtSignal(str, float)


# ===================================================================
#  ROS2 BACKGROUND NODE
# ===================================================================

class CaptureGUINode(Node):
    """Background ROS2 node — pub/sub + service clients for all pipeline stages."""

    FROZEN_TOPIC = '/capture_gui/frozen_frame'

    def __init__(self, signals: ROSSignals):
        super().__init__('capture_gui')
        self.signals = signals
        self.bridge  = CvBridge()

        self._waiting_capture = False
        self._frozen_frame    = None
        self._hz_stamps       = {k: [] for k in ('camera','debug','lines2d','lines3d','path')}

        # Publishers
        self.frozen_pub = self.create_publisher(Image, self.FROZEN_TOPIC, 1)

        # Subscribers
        self.create_subscription(Image, '/kinect2/qhd/image_color_rect', self._cb_camera, 5)
        self.create_subscription(Image, '/red_line_detector/debug_image',  self._cb_debug,  5)
        self.create_subscription(WeldLineArray,   '/vision/weld_lines_2d', self._cb_lines2d, 5)
        self.create_subscription(WeldLine3DArray, '/vision/weld_lines_3d', self._cb_lines3d, 5)
        self.create_subscription(Path,            '/vision/welding_path',  self._cb_path,    5)

        # Service clients
        self.detect_client = self.create_client(Trigger, '/red_line_detector/capture')
        self.depth_client  = self.create_client(Trigger, '/depth_matcher/capture')
        self.path_client   = self.create_client(Trigger, '/path_generator/trigger_path_generation')
        self.exec_client   = self.create_client(Trigger, '/moveit_controller/execute_welding_path')

        self.create_timer(1.0, self._emit_hz)
        self.get_logger().info('CaptureGUINode ready')

    # ── Callbacks ────────────────────────────────────────────────────

    def _cb_camera(self, msg):
        self._track_hz('camera')
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        if self._waiting_capture:
            self._waiting_capture = False
            self._frozen_frame = bgr.copy()
            try:
                fm = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                fm.header = msg.header
                self.frozen_pub.publish(fm)
                self.signals.log_message.emit(
                    f'[CAPTURE] {bgr.shape[1]}×{bgr.shape[0]} → {self.FROZEN_TOPIC}')
            except Exception as e:
                self.signals.log_message.emit(f'[ERROR] {e}')
        self.signals.camera_frame.emit(bgr)

    def _cb_debug(self, msg):
        self._track_hz('debug')
        try:
            self.signals.debug_frame.emit(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
        except Exception:
            pass

    def _cb_lines2d(self, msg):
        self._track_hz('lines2d')
        self.signals.lines_2d.emit(msg)

    def _cb_lines3d(self, msg):
        self._track_hz('lines3d')
        self.signals.lines_3d.emit(msg)

    def _cb_path(self, msg):
        self._track_hz('path')
        self.signals.path_received.emit(msg)

    # ── Public API ───────────────────────────────────────────────────

    def request_capture(self):
        self._waiting_capture = True
        self.signals.log_message.emit('[INFO] Waiting for next camera frame...')

    def republish_frozen(self):
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

    def get_frozen_frame(self):
        return self._frozen_frame

    def _call_service(self, client, name, callback):
        if not client.service_is_ready():
            self.signals.log_message.emit(f'[WARN] {name} not available')
            callback(None)
            return
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: callback(f.result()))

    def call_detect_service(self, cb):
        self._call_service(self.detect_client, '/red_line_detector/capture', cb)

    def call_depth_service(self, cb):
        self._call_service(self.depth_client, '/depth_matcher/capture', cb)

    def call_path_service(self, cb):
        self._call_service(self.path_client, '/path_generator/trigger_path_generation', cb)

    def call_exec_service(self, cb):
        self._call_service(self.exec_client, '/moveit_controller/execute_welding_path', cb)

    # ── Hz ───────────────────────────────────────────────────────────

    def _track_hz(self, key):
        now = time.monotonic()
        self._hz_stamps[key].append(now)
        self._hz_stamps[key] = [t for t in self._hz_stamps[key] if t >= now - 2.0]

    def _emit_hz(self):
        for k, stamps in self._hz_stamps.items():
            self.signals.hz_update.emit(k, len(stamps) / 2.0)


# ===================================================================
#  SUBPROCESS NODE MANAGER
# ===================================================================

class NodeProcess:
    def __init__(self, name, package, executable, ros_args, log_cb):
        self.name = name
        self._pkg  = package
        self._exe  = executable
        self._args = ros_args
        self._log  = log_cb
        self._proc = None

    @property
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def start(self):
        if self.running:
            return
        cmd = ['ros2', 'run', self._pkg, self._exe, '--ros-args'] + self._args
        self._log(f'[NODE] $ {" ".join(cmd)}')
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            threading.Thread(target=self._stream, daemon=True).start()
        except FileNotFoundError as e:
            self._log(f'[ERROR] {e}')

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._log(f'[NODE] {self.name} stopped.')
        self._proc = None

    def _stream(self):
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                self._log(f'[{self.name}] {line}')


# ===================================================================
#  HELPERS
# ===================================================================

def bgr_to_qpixmap(bgr, max_w=800, max_h=500):
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi).scaled(
        max_w, max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)


def led_label(color='gray'):
    lbl = QLabel('●')
    lbl.setStyleSheet(f'color:{color}; font-size:18px;')
    return lbl


def _node_row(start_btn, stop_btn, led_w, layout):
    row = QHBoxLayout()
    row.addWidget(start_btn)
    row.addWidget(stop_btn)
    row.addWidget(led_w)
    layout.addLayout(row)


# ===================================================================
#  PIPELINE STAGE CONSTANTS
# ===================================================================

STAGES = ['Capture', 'Detect', 'Depth', 'Path', 'Execute']


# ===================================================================
#  MAIN WINDOW
# ===================================================================

class CaptureWindow(QMainWindow):

    def __init__(self, ros_node: CaptureGUINode):
        super().__init__()
        self.node = ros_node
        self.sigs = ros_node.signals
        self._capture_count = 0
        self._pipeline_stage = -1   # -1 = idle

        # ── Node processes ────────────────────────────────────────────
        FROZEN = CaptureGUINode.FROZEN_TOPIC
        self._procs = {
            'det': NodeProcess('Detector',    'parol6_vision', 'red_line_detector',
                               ['-p', 'capture_mode:=true',
                                '-p', 'publish_debug_images:=true',
                                '--remap', f'/kinect2/qhd/image_color_rect:={FROZEN}'],
                               self._log),
            'dm':  NodeProcess('DepthMatcher', 'parol6_vision', 'depth_matcher',
                               ['-p', 'capture_mode:=true'], self._log),
            'pg':  NodeProcess('PathGen',      'parol6_vision', 'path_generator',
                               [], self._log),
            'mc':  NodeProcess('MoveIt',       'parol6_vision', 'moveit_controller',
                               [], self._log),
        }

        self._build_ui()
        self._connect_signals()

        # NOTE: do NOT call rclpy.spin_once here — the MultiThreadedExecutor
        # background thread is the sole ROS spinner. All Qt updates arrive
        # via pyqtSignal which is thread-safe. A second spin_once on the
        # Qt main thread would race against the executor and cause segfaults.
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(1000)
        self._health_timer.timeout.connect(self._poll_health)
        self._health_timer.start()

        self.setWindowTitle('PAROL6 Vision — Pipeline Testing GUI')
        self.resize(1440, 900)

    # ── UI BUILD ─────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(5)
        root.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel('🤖  PAROL6 Vision — Pipeline Testing GUI')
        title.setStyleSheet(
            'font-size:15px;font-weight:bold;color:#e0e0e0;'
            'background:#1e1e2e;padding:5px 12px;border-radius:4px;')
        root.addWidget(title)

        # Hz bars
        hz_row = QHBoxLayout()
        self._hz_widgets = {}
        for label, key in [('📷 Camera','camera'),('🔍 Debug','debug'),
                            ('📐 2D Lines','lines2d'),('🧊 3D Lines','lines3d'),
                            ('🛤️ Path','path')]:
            box = QGroupBox(label)
            box.setFixedWidth(145)
            bl  = QVBoxLayout(box)
            l   = led_label()
            b   = QProgressBar()
            b.setRange(0, 35); b.setValue(0)
            b.setTextVisible(True); b.setFormat('0 Hz'); b.setFixedHeight(13)
            bl.addWidget(l, alignment=Qt.AlignmentFlag.AlignCenter)
            bl.addWidget(b)
            self._hz_widgets[key] = (l, b)
            hz_row.addWidget(box)
        hz_row.addStretch()
        root.addLayout(hz_row)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ─────────────────────────────────────────────────────────────
        # LEFT PANEL (scrollable)
        # ─────────────────────────────────────────────────────────────
        left_inner = QWidget()
        lv = QVBoxLayout(left_inner)
        lv.setSpacing(5)
        lv.setContentsMargins(4, 4, 4, 4)

        # ── Node Controls ─────────────────────────────────────────────
        nc = QGroupBox('Node Controls')
        nc_l = QVBoxLayout(nc)

        def _make_node_row(key, label_start, label_stop):
            bs = QPushButton(f'▶  {label_start}')
            be = QPushButton(f'■  {label_stop}')
            be.setEnabled(False)
            lw = led_label()
            _node_row(bs, be, lw, nc_l)
            return bs, be, lw

        self._btn_start_det, self._btn_stop_det, self._led_det = _make_node_row('det', 'Start Detector',      'Stop Detector')
        self._btn_start_dm,  self._btn_stop_dm,  self._led_dm  = _make_node_row('dm',  'Start Depth Matcher', 'Stop Depth Matcher')
        self._btn_start_pg,  self._btn_stop_pg,  self._led_pg  = _make_node_row('pg',  'Start Path Generator','Stop Path Generator')
        self._btn_start_mc,  self._btn_stop_mc,  self._led_mc  = _make_node_row('mc',  'Start MoveIt Ctrl',   'Stop MoveIt Ctrl')
        lv.addWidget(nc)

        # ── Capture & Detection ───────────────────────────────────────
        wf = QGroupBox('① Capture & Detection')
        wf_l = QVBoxLayout(wf)
        self._btn_capture  = QPushButton('📷  Capture Frame')
        self._btn_redetect = QPushButton('🔁  Re-detect (same frame)')
        self._btn_save     = QPushButton('💾  Save Frozen Frame')
        self._lbl_cap_info = QLabel('No frame captured yet.')
        self._lbl_cap_info.setWordWrap(True)
        self._lbl_cap_info.setStyleSheet('color:#aaaaaa;font-size:11px;')
        self._btn_redetect.setEnabled(False)
        self._btn_save.setEnabled(False)
        for w in [self._btn_capture, self._btn_redetect,
                  self._btn_save, self._lbl_cap_info]:
            wf_l.addWidget(w)
        lv.addWidget(wf)

        # ── Depth Matching ────────────────────────────────────────────
        df = QGroupBox('② Depth Matching')
        df_l = QVBoxLayout(df)
        self._btn_match = QPushButton('🔍  Match Depth')
        self._btn_match.setEnabled(False)
        self._lbl_3d = QLabel('No 3D data yet.')
        self._lbl_3d.setWordWrap(True)
        df_l.addWidget(self._btn_match)
        df_l.addWidget(self._lbl_3d)
        lv.addWidget(df)

        # ── Path Generation ───────────────────────────────────────────
        pf = QGroupBox('③ Path Generation')
        pf_l = QVBoxLayout(pf)
        self._btn_gen_path = QPushButton('🛤️  Generate Path')
        self._btn_gen_path.setEnabled(False)
        self._lbl_path_stats = QLabel('No path yet.')
        self._lbl_path_stats.setWordWrap(True)
        pf_l.addWidget(self._btn_gen_path)
        pf_l.addWidget(self._lbl_path_stats)
        lv.addWidget(pf)

        # ── MoveIt Execution ──────────────────────────────────────────
        mf = QGroupBox('④ Robot Execution')
        mf_l = QVBoxLayout(mf)
        self._chk_enable_exec = QCheckBox('Enable robot execution (use with caution!)')
        self._chk_enable_exec.setStyleSheet('color:#ffaa44;')
        self._btn_execute = QPushButton('🤖  Execute Welding Path')
        self._btn_execute.setEnabled(False)
        self._lbl_exec_status = QLabel('Idle.')
        self._lbl_exec_status.setWordWrap(True)
        mf_l.addWidget(self._chk_enable_exec)
        mf_l.addWidget(self._btn_execute)
        mf_l.addWidget(self._lbl_exec_status)
        lv.addWidget(mf)

        # ── Full Pipeline ─────────────────────────────────────────────
        fp = QGroupBox('🚀 Full Pipeline')
        fp_l = QVBoxLayout(fp)

        # Progress bar showing which stage is active
        self._pipeline_bar = QProgressBar()
        self._pipeline_bar.setRange(0, len(STAGES))
        self._pipeline_bar.setValue(0)
        self._pipeline_bar.setTextVisible(True)
        self._pipeline_bar.setFormat('Idle')

        self._btn_full_pipeline = QPushButton('🚀  Run Full Pipeline')
        self._btn_full_pipeline.setStyleSheet(
            'background:#2a5a2a;color:#aaffaa;font-size:13px;font-weight:bold;'
            'border:1px solid #4a8a4a;border-radius:4px;padding:6px;')
        self._lbl_pipeline_status = QLabel('Press to run: Capture → Detect → Depth → Path')
        self._lbl_pipeline_status.setWordWrap(True)
        self._lbl_pipeline_status.setStyleSheet('color:#aaaaaa;font-size:11px;')

        fp_l.addWidget(self._btn_full_pipeline)
        fp_l.addWidget(self._pipeline_bar)
        fp_l.addWidget(self._lbl_pipeline_status)
        lv.addWidget(fp)

        # ── Detection Stats table ─────────────────────────────────────
        sg = QGroupBox('Detection Stats (2D Lines)')
        sg_l = QVBoxLayout(sg)
        self._stats_tbl = QTableWidget(0, 4)
        self._stats_tbl.setHorizontalHeaderLabels(['ID','Confidence','Pixels','BBox'])
        self._stats_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._stats_tbl.setFixedHeight(110)
        self._stats_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sg_l.addWidget(self._stats_tbl)
        lv.addWidget(sg)

        # ── Log ───────────────────────────────────────────────────────
        lg = QGroupBox('Log')
        lg_l = QVBoxLayout(lg)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont('Monospace', 9))
        self._log_view.setStyleSheet('background:#111122;color:#ccddee;')
        btn_clr = QPushButton('🗑  Clear')
        btn_clr.setFixedHeight(22)
        btn_clr.clicked.connect(self._log_view.clear)
        lg_l.addWidget(self._log_view)
        lg_l.addWidget(btn_clr)
        lv.addWidget(lg, stretch=1)
        lv.addStretch(0)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_inner)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(440)
        left_scroll.setStyleSheet('QScrollArea{border:none;}')
        splitter.addWidget(left_scroll)

        # ─────────────────────────────────────────────────────────────
        # RIGHT PANEL — image tabs
        # ─────────────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        tabs = QTabWidget()

        def _img_lbl(text):
            l = QLabel(text)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet('background:#0a0a1a;color:#888;')
            l.setMinimumSize(640, 360)
            return l

        self._lbl_live   = _img_lbl('Waiting for camera...')
        self._lbl_frozen = _img_lbl('No frame captured.')
        self._lbl_debug  = _img_lbl('No detection result yet.')
        tabs.addTab(self._lbl_live,   '📷 Live Preview')
        tabs.addTab(self._lbl_frozen, '🧊 Frozen Frame')
        tabs.addTab(self._lbl_debug,  '🔍 Detection Result')
        self._tabs = tabs
        rv.addWidget(tabs)
        splitter.addWidget(right)
        splitter.setSizes([380, 1060])

        # Status bar
        sb = self.statusBar()
        self._sb_labels = {}
        for key, text in [('det','Detector: ⬛'),('dm','DepthMatcher: ⬛'),
                           ('pg','PathGen: ⬛'),('mc','MoveIt: ⬛'),('cap','Captures: 0')]:
            lbl = QLabel(text)
            self._sb_labels[key] = lbl
            sb.addPermanentWidget(lbl)
            if key != 'cap':
                sb.addPermanentWidget(QLabel(' | '))

        self._apply_stylesheet()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background-color:#1a1a2e;color:#e0e0e0;}
            QGroupBox{border:1px solid #3a3a5c;border-radius:4px;margin-top:8px;
                       font-weight:bold;color:#aaaacc;}
            QGroupBox::title{subcontrol-origin:margin;padding:0 4px;}
            QPushButton{background:#2a2a4a;color:#d0d0f0;border:1px solid #4a4a7a;
                        border-radius:4px;padding:4px 10px;font-size:12px;}
            QPushButton:hover{background:#3a3a6a;}
            QPushButton:pressed{background:#1a1a3a;}
            QPushButton:disabled{color:#555566;border-color:#333355;}
            QCheckBox{color:#ffaa44;}
            QTabWidget::pane{border:1px solid #3a3a5c;}
            QTabBar::tab{background:#222244;color:#aaaacc;padding:5px 14px;border-radius:3px;}
            QTabBar::tab:selected{background:#3a3a6a;color:#e0e0ff;}
            QTableWidget{background:#0d0d1e;color:#c0d0e0;gridline-color:#2a2a4a;}
            QHeaderView::section{background:#222244;color:#aaaacc;border:none;padding:3px;}
            QProgressBar{text-align:center;border:1px solid #3a3a5c;border-radius:3px;
                          background:#111122;color:#e0e0e0;}
            QProgressBar::chunk{background:#3a6a3a;}
            QScrollArea{border:none;}
        """)

    # ── Signal connections ────────────────────────────────────────────

    def _connect_signals(self):
        self.sigs.camera_frame.connect(self._on_camera)
        self.sigs.debug_frame.connect(self._on_debug)
        self.sigs.lines_2d.connect(self._on_lines2d)
        self.sigs.lines_3d.connect(self._on_lines3d)
        self.sigs.path_received.connect(self._on_path)
        self.sigs.log_message.connect(self._log)
        self.sigs.hz_update.connect(self._on_hz)

        # Node control buttons
        self._btn_start_det.clicked.connect(lambda: self._start_proc('det', self._btn_start_det, self._btn_stop_det))
        self._btn_stop_det.clicked.connect( lambda: self._stop_proc( 'det', self._btn_start_det, self._btn_stop_det))
        self._btn_start_dm.clicked.connect( lambda: self._start_proc('dm',  self._btn_start_dm,  self._btn_stop_dm))
        self._btn_stop_dm.clicked.connect(  lambda: self._stop_proc( 'dm',  self._btn_start_dm,  self._btn_stop_dm))
        self._btn_start_pg.clicked.connect( lambda: self._start_proc('pg',  self._btn_start_pg,  self._btn_stop_pg))
        self._btn_stop_pg.clicked.connect(  lambda: self._stop_proc( 'pg',  self._btn_start_pg,  self._btn_stop_pg))
        self._btn_start_mc.clicked.connect( lambda: self._start_proc('mc',  self._btn_start_mc,  self._btn_stop_mc))
        self._btn_stop_mc.clicked.connect(  lambda: self._stop_proc( 'mc',  self._btn_start_mc,  self._btn_stop_mc))

        # Pipeline step buttons
        self._btn_capture.clicked.connect(self._do_capture)
        self._btn_redetect.clicked.connect(self._do_redetect)
        self._btn_save.clicked.connect(self._do_save)
        self._btn_match.clicked.connect(self._do_match)
        self._btn_gen_path.clicked.connect(self._do_gen_path)
        self._btn_execute.clicked.connect(self._do_execute)
        self._btn_full_pipeline.clicked.connect(self._do_full_pipeline)

        # Safety gate
        self._chk_enable_exec.stateChanged.connect(self._on_exec_gate_changed)

    # ── Node process control ──────────────────────────────────────────

    def _start_proc(self, key, start_btn, stop_btn):
        self._procs[key].start()
        start_btn.setEnabled(False)
        stop_btn.setEnabled(True)

    def _stop_proc(self, key, start_btn, stop_btn):
        self._procs[key].stop()
        start_btn.setEnabled(True)
        stop_btn.setEnabled(False)

    _NODE_META = {
        'det': ('Detector',      '_btn_start_det', '_btn_stop_det', '_led_det', 'det'),
        'dm':  ('DepthMatcher',  '_btn_start_dm',  '_btn_stop_dm',  '_led_dm',  'dm'),
        'pg':  ('PathGen',       '_btn_start_pg',  '_btn_stop_pg',  '_led_pg',  'pg'),
        'mc':  ('MoveIt',        '_btn_start_mc',  '_btn_stop_mc',  '_led_mc',  'mc'),
    }

    def _poll_health(self):
        meta = [
            (self._procs['det'], self._led_det, self._btn_start_det, self._btn_stop_det, 'det', 'Detector'),
            (self._procs['dm'],  self._led_dm,  self._btn_start_dm,  self._btn_stop_dm,  'dm',  'DepthMatcher'),
            (self._procs['pg'],  self._led_pg,  self._btn_start_pg,  self._btn_stop_pg,  'pg',  'PathGen'),
            (self._procs['mc'],  self._led_mc,  self._btn_start_mc,  self._btn_stop_mc,  'mc',  'MoveIt'),
        ]
        for proc, lw, bs, be, sb_key, label in meta:
            if proc.running:
                lw.setStyleSheet('color:#00ff88;font-size:18px;')
                self._sb_labels[sb_key].setText(f'{label}: 🟢')
            else:
                lw.setStyleSheet('color:#666677;font-size:18px;')
                self._sb_labels[sb_key].setText(f'{label}: ⬛')
                if not bs.isEnabled():
                    bs.setEnabled(True)
                    be.setEnabled(False)

    # ── Step 1: Capture ───────────────────────────────────────────────

    def _do_capture(self):
        self._btn_capture.setEnabled(False)
        self.node.request_capture()
        QTimer.singleShot(400, self._call_detect)

    def _call_detect(self):
        self._log('[GUI] Calling /red_line_detector/capture...')
        self.node.call_detect_service(self._on_detect_resp)

    def _on_detect_resp(self, resp, _pipeline=False):
        self._btn_capture.setEnabled(True)
        if resp is None:
            self._lbl_cap_info.setText('❌ Detect service unavailable')
            self._lbl_cap_info.setStyleSheet('color:#ff6666;font-size:11px;')
            if _pipeline:
                self._pipeline_failed('Detect service unavailable')
            return
        ok = resp.success
        color = '#88ffaa' if ok else '#ff6666'
        self._lbl_cap_info.setText(('✅ ' if ok else '❌ ') + resp.message)
        self._lbl_cap_info.setStyleSheet(f'color:{color};font-size:11px;')
        if ok:
            self._capture_count += 1
            self._sb_labels['cap'].setText(f'Captures: {self._capture_count}')
            self._btn_redetect.setEnabled(True)
            self._btn_match.setEnabled(True)
            self._btn_save.setEnabled(True)
            self._tabs.setCurrentIndex(2)
        self._log(f'[DETECT] {"OK" if ok else "FAIL"} — {resp.message}')
        if _pipeline:
            if ok:
                self._advance_pipeline(2)  # → depth
            else:
                self._pipeline_failed(resp.message)

    def _do_redetect(self):
        if not self.node.republish_frozen():
            return
        QTimer.singleShot(150, self._call_detect)

    def _do_save(self):
        frozen = self.node.get_frozen_frame()
        if frozen is None:
            return
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Frame', f'capture_{ts}.png', 'PNG (*.png)')
        if path:
            cv2.imwrite(path, frozen)
            self._log(f'[GUI] Saved → {path}')

    # ── Step 2: Depth ─────────────────────────────────────────────────

    def _do_match(self, _pipeline=False):
        self._btn_match.setEnabled(False)
        self._log('[GUI] Calling /depth_matcher/capture...')
        self.node.call_depth_service(lambda r: self._on_depth_resp(r, _pipeline))

    def _on_depth_resp(self, resp, _pipeline=False):
        self._btn_match.setEnabled(True)
        if resp is None:
            if _pipeline:
                self._pipeline_failed('Depth service unavailable')
            return
        ok = resp.success
        self._lbl_3d.setStyleSheet(f'color:{"#88ffaa" if ok else "#ff6666"};')
        self._log(f'[DEPTH] {"OK" if ok else "FAIL"} — {resp.message}')
        if _pipeline:
            if ok:
                self._advance_pipeline(3)  # → path gen
            else:
                self._pipeline_failed(resp.message)

    # ── Step 3: Path generation ───────────────────────────────────────

    def _do_gen_path(self, _pipeline=False):
        self._btn_gen_path.setEnabled(False)
        self._log('[GUI] Calling /path_generator/trigger_path_generation...')
        self.node.call_path_service(lambda r: self._on_path_resp(r, _pipeline))

    def _on_path_resp(self, resp, _pipeline=False):
        self._btn_gen_path.setEnabled(True)
        if resp is None:
            if _pipeline:
                self._pipeline_failed('Path generator service unavailable')
            return
        ok = resp.success
        self._log(f'[PATH] {"OK" if ok else "FAIL"} — {resp.message}')
        if _pipeline:
            if ok:
                # Check if execute is enabled
                if self._chk_enable_exec.isChecked():
                    self._advance_pipeline(4)  # → execute
                else:
                    self._pipeline_complete('Path generated. Execute is disabled — done.')
            else:
                self._pipeline_failed(resp.message)

    # ── Step 4: Execute ───────────────────────────────────────────────

    def _on_exec_gate_changed(self, state):
        has_path = 'waypoint' in self._lbl_path_stats.text()
        self._btn_execute.setEnabled(bool(state) and has_path)

    def _do_execute(self, _pipeline=False):
        if not self._chk_enable_exec.isChecked():
            return
        self._btn_execute.setEnabled(False)
        self._lbl_exec_status.setText('⏳ Executing...')
        self._lbl_exec_status.setStyleSheet('color:#ffee44;')
        self._log('[GUI] Calling /moveit_controller/execute_welding_path...')
        self.node.call_exec_service(lambda r: self._on_exec_resp(r, _pipeline))

    def _on_exec_resp(self, resp, _pipeline=False):
        self._btn_execute.setEnabled(self._chk_enable_exec.isChecked())
        if resp is None:
            self._lbl_exec_status.setText('❌ MoveIt service unavailable')
            self._lbl_exec_status.setStyleSheet('color:#ff6666;')
            if _pipeline:
                self._pipeline_failed('MoveIt service unavailable')
            return
        ok = resp.success
        self._lbl_exec_status.setText(('✅ ' if ok else '❌ ') + resp.message)
        self._lbl_exec_status.setStyleSheet(f'color:{"#88ffaa" if ok else "#ff6666"};')
        self._log(f'[EXEC] {"OK" if ok else "FAIL"} — {resp.message}')
        if _pipeline:
            if ok:
                self._pipeline_complete('Welding execution complete!')
            else:
                self._pipeline_failed(resp.message)

    # ── Full Pipeline ─────────────────────────────────────────────────

    def _do_full_pipeline(self):
        """Kick off the sequential pipeline: Capture → Detect → Depth → Path [→ Execute]."""
        self._btn_full_pipeline.setEnabled(False)
        self._pipeline_bar.setValue(0)
        self._lbl_pipeline_status.setText('⏳ Stage 1/4: Capturing...')
        self._lbl_pipeline_status.setStyleSheet('color:#ffee44;font-size:11px;')

        # Stage 1: capture
        self._pipeline_bar.setFormat('Capture')
        self._btn_capture.setEnabled(False)
        self.node.request_capture()
        QTimer.singleShot(400, lambda: self.node.call_detect_service(
            lambda r: self._on_detect_resp(r, _pipeline=True)))

    def _advance_pipeline(self, stage: int):
        """Called after each stage succeeds to trigger the next."""
        self._pipeline_bar.setValue(stage - 1)
        if stage == 2:  # depth
            self._pipeline_bar.setFormat('Depth Matching')
            self._lbl_pipeline_status.setText('⏳ Stage 2/4: Matching depth...')
            self.node.call_depth_service(lambda r: self._on_depth_resp(r, _pipeline=True))
        elif stage == 3:  # path gen
            self._pipeline_bar.setFormat('Path Generation')
            self._lbl_pipeline_status.setText('⏳ Stage 3/4: Generating path...')
            self.node.call_path_service(lambda r: self._on_path_resp(r, _pipeline=True))
        elif stage == 4:  # execute
            self._pipeline_bar.setFormat('Executing')
            self._lbl_pipeline_status.setText('⏳ Stage 4/4: Executing weld...')
            self._do_execute(_pipeline=True)

    def _pipeline_complete(self, msg='Done.'):
        self._pipeline_bar.setValue(len(STAGES))
        self._pipeline_bar.setFormat('Complete ✅')
        self._lbl_pipeline_status.setText(f'✅ {msg}')
        self._lbl_pipeline_status.setStyleSheet('color:#88ffaa;font-size:11px;')
        self._btn_full_pipeline.setEnabled(True)
        self._btn_capture.setEnabled(True)
        self._log(f'[PIPELINE] Complete: {msg}')

    def _pipeline_failed(self, reason):
        self._pipeline_bar.setFormat('Failed ❌')
        self._lbl_pipeline_status.setText(f'❌ Pipeline failed: {reason}')
        self._lbl_pipeline_status.setStyleSheet('color:#ff6666;font-size:11px;')
        self._btn_full_pipeline.setEnabled(True)
        self._btn_capture.setEnabled(True)
        self._log(f'[PIPELINE] Failed: {reason}')

    # ── ROS signal handlers ───────────────────────────────────────────

    def _on_camera(self, bgr):
        self._lbl_live.setPixmap(bgr_to_qpixmap(bgr))
        frozen = self.node.get_frozen_frame()
        if frozen is not None:
            self._lbl_frozen.setPixmap(bgr_to_qpixmap(frozen))

    def _on_debug(self, bgr):
        self._lbl_debug.setPixmap(bgr_to_qpixmap(bgr))

    def _on_lines2d(self, msg):
        self._stats_tbl.setRowCount(0)
        for line in msg.lines:
            row = self._stats_tbl.rowCount()
            self._stats_tbl.insertRow(row)
            c = float(line.confidence)
            clr = '#88ff88' if c >= 0.9 else ('#ffff88' if c >= 0.7 else '#ffaa44')
            bbox = (f'({line.bbox_min.x:.0f},{line.bbox_min.y:.0f})'
                    f'→({line.bbox_max.x:.0f},{line.bbox_max.y:.0f})')
            for col, (val, fc) in enumerate([
                (line.id, '#c0d0e0'), (f'{c:.3f}', clr),
                (str(len(line.pixels)), '#c0d0e0'), (bbox, '#c0d0e0')
            ]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(fc))
                self._stats_tbl.setItem(row, col, item)

    def _on_lines3d(self, msg):
        n = len(msg.lines)
        if n == 0:
            self._lbl_3d.setText('No 3D lines.')
            return
        info = '\n'.join(
            f'  • {l.id}: {len(l.points)} pts, conf={l.confidence:.2f}'
            for l in msg.lines)
        self._lbl_3d.setText(f'✅ {n} 3D line(s):\n{info}')
        self._lbl_3d.setStyleSheet('color:#88ffaa;')
        self._btn_gen_path.setEnabled(True)

    def _on_path(self, msg):
        n = len(msg.poses)
        if n == 0:
            self._lbl_path_stats.setText('Empty path.')
            return
        start = msg.poses[0].pose.position
        end   = msg.poses[-1].pose.position
        total_len = sum(
            ((msg.poses[i].pose.position.x - msg.poses[i-1].pose.position.x)**2 +
             (msg.poses[i].pose.position.y - msg.poses[i-1].pose.position.y)**2 +
             (msg.poses[i].pose.position.z - msg.poses[i-1].pose.position.z)**2)**0.5
            for i in range(1, n)
        )
        self._lbl_path_stats.setText(
            f'✅ {n} waypoints | length={total_len*1000:.1f} mm\n'
            f'  Start: ({start.x:.3f}, {start.y:.3f}, {start.z:.3f})\n'
            f'  End:   ({end.x:.3f}, {end.y:.3f}, {end.z:.3f})'
        )
        self._lbl_path_stats.setStyleSheet('color:#88ffaa;')
        if self._chk_enable_exec.isChecked():
            self._btn_execute.setEnabled(True)
        # If pipeline is running at stage 3, advance
        if self._pipeline_bar.format() == 'Path Generation':
            self._pipeline_complete('Path generated. Enable execution to run the robot.')

    def _on_hz(self, key, hz):
        if key not in self._hz_widgets:
            return
        l, b = self._hz_widgets[key]
        b.setValue(int(min(hz, 35)))
        b.setFormat(f'{hz:.1f} Hz')
        l.setStyleSheet(
            f'color:{"#00ff88" if hz>=20 else "#ffee44" if hz>=5 else "#ff8844" if hz>0 else "#555566"};'
            'font-size:18px;')

    # ── Log ───────────────────────────────────────────────────────────

    def _log(self, msg):
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        sb = self._log_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 10
        self._log_view.append(f'[{ts}] {msg}')
        if at_bottom:
            sb.setValue(sb.maximum())

    # ── Cleanup ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._health_timer.stop()
        for p in self._procs.values():
            p.stop()
        event.accept()


# ===================================================================
#  ENTRY POINT
# ===================================================================

def main():
    rclpy.init()
    signals  = ROSSignals()
    ros_node = CaptureGUINode(signals)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(ros_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    app    = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CaptureWindow(ros_node)
    window.show()

    exit_code = app.exec()

    executor.shutdown(timeout_sec=2.0)
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
