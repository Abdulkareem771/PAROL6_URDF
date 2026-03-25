"""
Mask Pipeline Tester — PySide6 Tool
=====================================
End-to-end test harness for the PAROL6 mask-based path detection pipeline.

Pipeline:
  1. Load an image (local file, folder, or single-frame ROS grab)
  2. Run YOLO segmentation → each detected object ID gets a configurable mask color
     (default: ID 0 = Green, ID 1 = Red — matching detect_path.py convention)
  3. Apply the path-detection algorithm (from detect_path.py) inline:
       - Build HSV masks from the painted green/red regions
       - Compute bounding boxes around each colour region
       - Compute the intersection bounding box (the seam path)
  4. Visualise: G mask | R mask | Annotated overlay with intersection
  5. (Optional) Publish result to a ROS topic

Design: The path-detection logic is reproduced faithfully here so we never
need to import or modify the teammate's script.

New in v2:
  - Hide Labels / ID Text checkbox
  - Path Visualisation mode selector:
      • Rectangle (original detect_path.py bbox)
      • Centerline (horizontal line through the centre of the intersection)
      • Band (filled horizontal strip of configurable pixel width)
  - Advanced ROS section (optional, non-breaking):
      • Subscribe to a ROS topic to grab a single frame as input
      • Publish the annotated result canvas to a ROS topic
"""
import sys
import os
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QSlider, QComboBox, QColorDialog,
    QFileDialog, QMessageBox, QCheckBox, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QVBoxLayout, QGridLayout, QLineEdit, QSpinBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C, STYLE_SHEET

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False

try:
    import rclpy
    from sensor_msgs.msg import Image as ROSImage
    from cv_bridge import CvBridge
    ROS2_OK = True
except ImportError:
    ROS2_OK = False

# ─── HSV ranges matching detect_path.py exactly ──────────────────────────────
LOWER_GREEN = np.array([35,  50,  50])
UPPER_GREEN = np.array([85, 255, 255])
LOWER_RED1  = np.array([0,   70,  50])
UPPER_RED1  = np.array([10, 255, 255])
LOWER_RED2  = np.array([170, 70,  50])
UPPER_RED2  = np.array([180, 255, 255])

# Default colour mapping: YOLO object ID → mask RGB colour
DEFAULT_ID_COLORS = {
    0: (0, 200, 80),    # Green  — matches lower/upper_green HSV
    1: (220, 40,  40),  # Red    — matches lower/upper_red HSV
}

# Path visualisation, seam path colour (yellow)
SEAM_COLOR = (255, 255, 0)


def _find_seam_path(rgb_image: np.ndarray,
                    path_mode: str = "Rectangle",
                    band_width: int = 8,
                    hide_labels: bool = False):
    """
    Inline reimplementation of detect_path.segment_blocks() logic.
    Accepts an RGB numpy array.

    path_mode:
      "Rectangle" — original yellow bounding box
      "Centerline" — single horizontal line through the vertical centre of bbox_I
      "Band" — filled horizontal strip of `band_width` pixels at the centre

    Returns: (G_mask, R_mask, annotated_rgb, bbox_G, bbox_R, bbox_I)
    """
    img_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    # Green mask
    G = cv2.morphologyEx(cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN),
                         cv2.MORPH_OPEN, kernel)
    # Red mask (two ranges)
    R = cv2.morphologyEx(
        cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
                       cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)),
        cv2.MORPH_OPEN, kernel)

    annotated = rgb_image.copy()

    def _bbox(mask):
        rows, cols = np.where(mask == 255)
        if len(rows) == 0:
            return None
        return (int(cols.min()) - 2, int(rows.min()) - 2,
                int(cols.max()) + 2, int(rows.max()) + 2)

    bbox_G = _bbox(G)
    bbox_R = _bbox(R)

    if bbox_G:
        cv2.rectangle(annotated, bbox_G[:2], bbox_G[2:], (0, 255, 0), 2)
        if not hide_labels:
            cv2.putText(annotated, "Green Block", (bbox_G[0], bbox_G[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    if bbox_R:
        cv2.rectangle(annotated, bbox_R[:2], bbox_R[2:], (255, 60, 60), 2)
        if not hide_labels:
            cv2.putText(annotated, "Red Block", (bbox_R[0], bbox_R[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60), 1)

    bbox_I = None
    if bbox_G and bbox_R:
        ix1 = max(bbox_G[0], bbox_R[0])
        iy1 = max(bbox_G[1], bbox_R[1])
        ix2 = min(bbox_G[2], bbox_R[2])
        iy2 = min(bbox_G[3], bbox_R[3])
        if ix1 < ix2 and iy1 < iy2:
            bbox_I = (ix1, iy1, ix2, iy2)
            cy = (iy1 + iy2) // 2  # vertical centre

            if path_mode == "Rectangle":
                cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), SEAM_COLOR, 3)
                if not hide_labels:
                    cv2.putText(annotated, "SEAM PATH", (ix1, iy1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, SEAM_COLOR, 2)

            elif path_mode == "Centerline":
                cv2.line(annotated, (ix1, cy), (ix2, cy), SEAM_COLOR, 2)
                if not hide_labels:
                    cv2.putText(annotated, "SEAM", (ix1, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, SEAM_COLOR, 1)

            elif path_mode == "Band":
                half = max(1, band_width // 2)
                y_top = max(0, cy - half)
                y_bot = min(annotated.shape[0], cy + half)
                cv2.rectangle(annotated, (ix1, y_top), (ix2, y_bot), SEAM_COLOR, -1)
                if not hide_labels:
                    cv2.putText(annotated, "SEAM", (ix1, y_top - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, SEAM_COLOR, 1)

    return G, R, annotated, bbox_G, bbox_R, bbox_I


class MaskPipelineTester(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Mask Pipeline Tester — YOLO → Colour Mask → Path Detection",
                         width=1500, height=880)
        self._model = None
        self._results = None
        self._conf = 0.30
        self._id_colors = dict(DEFAULT_ID_COLORS)
        self._last_annotated = None

        # ROS state (only used if checkboxes are ticked and ROS2_OK)
        self._ros_node = None
        self._ros_sub = None
        self._ros_pub = None
        self._bridge = None
        self._pending_ros_frame = None

        self._setup_ui()
        self._build_quad_canvas()
        self.image_loaded.connect(self._on_image_loaded)
        self._init_ros()

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        # Input
        self._add_section_header("Input Image")
        self._build_default_image_loader()

        # Model
        self._add_section_header("YOLO Model")
        self.model_lbl = QLabel("No model loaded")
        self.model_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.model_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self.model_lbl)
        self._add_button("🧠  Load YOLO Weights (.pt)", self._load_model)

        # Confidence
        self._add_section_header("Confidence Threshold")
        conf_row = QHBoxLayout()
        self._lbl_conf = QLabel(f"{self._conf:.2f}")
        self._lbl_conf.setStyleSheet(f"color: {C['accent']}; font-weight: bold;")
        self._slider_conf = QSlider(Qt.Horizontal)
        self._slider_conf.setRange(1, 99)
        self._slider_conf.setValue(int(self._conf * 100))
        self._slider_conf.valueChanged.connect(self._on_conf_change)
        conf_row.addWidget(self._slider_conf)
        conf_row.addWidget(self._lbl_conf)
        self.sidebar_layout.addLayout(conf_row)

        # Display options
        self._add_section_header("Display Options")
        self._chk_hide_labels = QCheckBox("Hide Labels / ID Text")
        self._chk_hide_labels.setStyleSheet(f"color: {C['text']};")
        self._chk_hide_labels.setChecked(False)
        self.sidebar_layout.addWidget(self._chk_hide_labels)

        # Path visualisation mode
        self._add_section_header("Path Visualisation")
        self._path_mode_combo = QComboBox()
        for m in ["Rectangle", "Centerline", "Band"]:
            self._path_mode_combo.addItem(m)
        self.sidebar_layout.addWidget(self._path_mode_combo)

        band_row = QHBoxLayout()
        band_lbl = QLabel("Band width (px):")
        band_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self._band_spin = QSpinBox()
        self._band_spin.setRange(1, 120)
        self._band_spin.setValue(8)
        self._band_spin.setEnabled(False)
        self._band_spin.setStyleSheet(
            f"background: {C['border']}; color: {C['text']}; border: none; padding: 2px;"
        )
        band_row.addWidget(band_lbl)
        band_row.addWidget(self._band_spin)
        self.sidebar_layout.addLayout(band_row)
        self._path_mode_combo.currentTextChanged.connect(
            lambda t: self._band_spin.setEnabled(t == "Band"))

        # Per-ID Colour Mapping
        self._add_section_header("Object ID → Mask Colour")
        info = QLabel("ID 0 = Green (seam side A)\nID 1 = Red (seam side B)\nDouble-click to change.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {C['text2']}; font-size: 10px;")
        self.sidebar_layout.addWidget(info)

        self._id_list = QListWidget()
        self._id_list.setMaximumHeight(110)
        self._id_list.setFont(QFont("Monospace", 10))
        self._id_list.setStyleSheet(
            f"background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border']};"
        )
        self._id_list.itemDoubleClicked.connect(self._pick_id_color)
        self.sidebar_layout.addWidget(self._id_list)

        self._add_button("🎨  Change Colour for Selected ID", self._pick_id_color_btn)
        self._add_button("↩  Reset to Green/Red Defaults", self._reset_colors)

        # ── Advanced ROS ─────────────────────────────────────────────────────
        self._add_section_header("Advanced — ROS 2")
        ros_status = "rclpy available ✅" if ROS2_OK else "rclpy not available (offline mode)"
        ros_lbl = QLabel(ros_status)
        ros_lbl.setStyleSheet(
            f"color: {'#a6e3a1' if ROS2_OK else '#f38ba8'}; font-size: 10px;")
        self.sidebar_layout.addWidget(ros_lbl)

        # Sub checkbox + topic
        self._chk_ros_sub = QCheckBox("Import single frame from ROS topic")
        self._chk_ros_sub.setStyleSheet(f"color: {C['text']};")
        self._chk_ros_sub.setEnabled(ROS2_OK)
        self.sidebar_layout.addWidget(self._chk_ros_sub)

        self._ros_in_topic = QLineEdit()
        self._ros_in_topic.setPlaceholderText("Input topic (e.g. /camera/image_raw)")
        self._ros_in_topic.setEnabled(ROS2_OK)
        self.sidebar_layout.addWidget(self._ros_in_topic)

        btn_grab = QPushButton("📡  Grab One Frame from ROS")
        btn_grab.setEnabled(ROS2_OK)
        btn_grab.setStyleSheet(f"background: {C['warn']}; color: {C['bg']}; font-weight: bold;")
        btn_grab.clicked.connect(self._grab_ros_frame)
        self.sidebar_layout.addWidget(btn_grab)

        # Pub checkbox + topic
        self._chk_ros_pub = QCheckBox("Publish result to ROS topic")
        self._chk_ros_pub.setStyleSheet(f"color: {C['text']};")
        self._chk_ros_pub.setEnabled(ROS2_OK)
        self.sidebar_layout.addWidget(self._chk_ros_pub)

        self._ros_out_topic = QLineEdit()
        self._ros_out_topic.setPlaceholderText("Output topic (e.g. /vision/result)")
        self._ros_out_topic.setEnabled(ROS2_OK)
        self.sidebar_layout.addWidget(self._ros_out_topic)

        # Pipeline Result
        self._add_section_header("Pipeline Result")
        self.lbl_result = QLabel("Run pipeline to see result.")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.sidebar_layout.addWidget(self.lbl_result)

        # Run & Export
        self.sidebar_layout.addStretch()
        self._add_button("▶  Run Full Pipeline", self._run_pipeline, primary=True)
        self._add_button("💾  Save Annotated Result", self._save_result)

    def _build_quad_canvas(self):
        """Replace the single QGraphicsView with a 2×2 quad-panel canvas."""
        old_view = self.splitter.widget(1)
        old_view.setParent(None)

        quad = QFrame()
        quad_layout = QGridLayout(quad)
        quad_layout.setSpacing(4)
        quad_layout.setContentsMargins(4, 4, 4, 4)

        self._panels = {}
        titles = ["Original + YOLO Masks", "G Mask (Green Region)",
                  "R Mask (Red Region)", "Path Detection Result"]
        for idx, title in enumerate(titles):
            row, col = divmod(idx, 2)
            frame = QFrame()
            frame.setStyleSheet(
                f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 4px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.setSpacing(2)
            lbl_title = QLabel(f" {title}")
            lbl_title.setStyleSheet(
                f"color: {C['text2']}; font-size: 10px; font-weight: bold;"
                f" background: {C['panel']}; padding: 3px;")
            fl.addWidget(lbl_title)
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setScaledContents(False)
            from PySide6.QtWidgets import QSizePolicy
            img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            img_lbl.setMinimumSize(200, 150)
            fl.addWidget(img_lbl)
            self._panels[idx] = img_lbl
            quad_layout.addWidget(frame, row, col)

        self.splitter.addWidget(quad)
        self.splitter.setSizes([380, 1120])

    # ─────────────────────────────────────────────────────────────────────────
    # ROS setup
    # ─────────────────────────────────────────────────────────────────────────
    def _init_ros(self):
        if not ROS2_OK:
            return
        if not rclpy.ok():
            rclpy.init()
        self._ros_node = rclpy.create_node('mask_pipeline_tester_node')
        self._bridge = CvBridge()
        # Spin ROS in background at 100 Hz so subscriptions are served
        self._ros_spin_timer = QTimer(self)
        self._ros_spin_timer.timeout.connect(
            lambda: rclpy.spin_once(self._ros_node, timeout_sec=0.005))
        self._ros_spin_timer.start(10)

    def _grab_ros_frame(self):
        """Subscribe to a ROS topic, wait for exactly one frame, then unsubscribe."""
        if not ROS2_OK or not self._ros_node:
            return
        topic = self._ros_in_topic.text().strip()
        if not topic:
            QMessageBox.warning(self, "No Topic", "Enter a ROS input topic first.")
            return

        self._pending_ros_frame = None
        self.lbl_result.setText(f"Waiting for frame on: {topic} …")

        def _cb(msg):
            try:
                cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                self._pending_ros_frame = cv_img
                self._set_rgb_image(cv_img)
                self.lbl_result.setText(
                    f"Frame grabbed from {topic}  "
                    f"({cv_img.shape[1]}×{cv_img.shape[0]})")
            except Exception as e:
                self.lbl_result.setText(f"Frame grab error: {e}")
            finally:
                # Unsubscribe after first frame
                if self._ros_sub:
                    self._ros_node.destroy_subscription(self._ros_sub)
                    self._ros_sub = None

        if self._ros_sub:
            self._ros_node.destroy_subscription(self._ros_sub)
        self._ros_sub = self._ros_node.create_subscription(ROSImage, topic, _cb, 1)

    def _publish_result(self, rgb: np.ndarray):
        """Publish the annotated result to the configured ROS output topic."""
        if not self._chk_ros_pub.isChecked():
            return
        if not ROS2_OK or not self._ros_node:
            return
        topic = self._ros_out_topic.text().strip()
        if not topic:
            return

        # Create or reuse publisher (recreate if topic changed)
        if self._ros_pub is None:
            self._ros_pub = self._ros_node.create_publisher(ROSImage, topic, 10)

        try:
            msg = self._bridge.cv2_to_imgmsg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                              encoding="bgr8")
            self._ros_pub.publish(msg)
        except Exception as e:
            print(f"[ROS Publish Error] {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Model & Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        if not ULTRALYTICS_OK:
            QMessageBox.critical(self, "Error", "ultralytics is not installed.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Select YOLO Weights", "", "Models (*.pt *.onnx)")
        if p:
            try:
                self._model = YOLO(p)
                self.model_lbl.setText(os.path.basename(p))
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _on_image_loaded(self, rgb):
        self._show_panel(0, rgb)

    def _on_conf_change(self, val):
        self._conf = val / 100.0
        self._lbl_conf.setText(f"{self._conf:.2f}")

    def _run_pipeline(self):
        if self._rgb is None:
            return
        t0 = time.perf_counter()
        img = self._rgb.copy()

        hide_labels = self._chk_hide_labels.isChecked()
        path_mode = self._path_mode_combo.currentText()
        band_width = self._band_spin.value()

        # --- Step 1: YOLO Inference ---
        if self._model is not None:
            self._results = self._model.predict(img, conf=self._conf, verbose=False)
            img = self._apply_id_masks(img, self._results, hide_labels)
            self._populate_id_list()

        self._show_panel(0, img)

        # --- Step 2: Path Detection ---
        G, R, annotated, bbox_G, bbox_R, bbox_I = _find_seam_path(
            img,
            path_mode=path_mode,
            band_width=band_width,
            hide_labels=hide_labels,
        )

        # Show quad panels
        self._show_panel(1, cv2.cvtColor(cv2.merge([G, G, G]), cv2.COLOR_BGR2RGB))
        self._show_panel(2, cv2.cvtColor(cv2.merge([R, R, R]), cv2.COLOR_BGR2RGB))
        self._show_panel(3, annotated)
        self._last_annotated = annotated

        # --- Step 3: Optional ROS publish ---
        self._publish_result(annotated)

        elapsed = (time.perf_counter() - t0) * 1000
        g_px = int(np.sum(G > 0)) if G is not None else 0
        r_px = int(np.sum(R > 0)) if R is not None else 0
        pub_note = ""
        if self._chk_ros_pub.isChecked() and self._ros_out_topic.text().strip():
            pub_note = f"\n📡 Published to: {self._ros_out_topic.text().strip()}"
        if bbox_I:
            w = bbox_I[2] - bbox_I[0]; h = bbox_I[3] - bbox_I[1]
            self.lbl_result.setText(
                f"✅ Seam path found!  [{path_mode}]\n"
                f"Intersection: ({bbox_I[0]},{bbox_I[1]}) → ({bbox_I[2]},{bbox_I[3]})\n"
                f"Size: {w}×{h}px\n"
                f"Green px: {g_px:,}  Red px: {r_px:,}\n"
                f"Pipeline: {elapsed:.1f} ms{pub_note}"
            )
        else:
            self.lbl_result.setText(
                f"⚠️  No seam intersection found.\n"
                f"Green px: {g_px:,}  Red px: {r_px:,}\n"
                f"Check that ID 0 = green, ID 1 = red.\n"
                f"Pipeline: {elapsed:.1f} ms"
            )

    def _apply_id_masks(self, rgb: np.ndarray, results, hide_labels: bool = False) -> np.ndarray:
        """Paint each detected object ID with its assigned colour using the segmentation mask."""
        img = rgb.copy()
        if not results or not results[0].boxes:
            return img

        boxes = results[0].boxes
        masks_data = results[0].masks
        h, w = img.shape[:2]

        for i, box in enumerate(boxes):
            color = self._id_colors.get(i, (128, 128, 128))

            if masks_data is not None:
                raw_mask = masks_data.data[i].cpu().numpy()
                msk = cv2.resize(raw_mask, (w, h)) > 0.5
                img[msk] = color
            else:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            if not hide_labels:
                x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                cv2.putText(img, f"ID:{i}", (x1 + 4, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return img

    def _show_panel(self, idx: int, rgb: np.ndarray):
        if rgb is None:
            return
        lbl = self._panels[idx]
        available = lbl.size()
        w = max(available.width(), 100)
        h = max(available.height() - 30, 80)
        scaled = cv2.resize(rgb, (w, h))
        sh, sw, ch = scaled.shape
        qimg = QImage(scaled.data, sw, sh, ch * sw, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))

    # ─────────────────────────────────────────────────────────────────────────
    def _populate_id_list(self):
        self._id_list.clear()
        if not self._results or not self._results[0].boxes:
            return
        names = self._results[0].names
        for i, box in enumerate(self._results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            color = self._id_colors.get(i, (128, 128, 128))
            item = QListWidgetItem(f"ID {i}: {names[cls]} ({conf:.2f})")
            item.setBackground(QColor(*color))
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            item.setForeground(QColor("#000000" if lum > 128 else "#ffffff"))
            self._id_list.addItem(item)

    def _pick_id_color_btn(self):
        self._pick_id_color(self._id_list.currentItem())

    def _pick_id_color(self, item=None):
        row = self._id_list.currentRow()
        if row < 0:
            return
        current = self._id_colors.get(row, (128, 128, 128))
        result = QColorDialog.getColor(QColor(*current), self, f"Colour for ID {row}")
        if result.isValid():
            self._id_colors[row] = (result.red(), result.green(), result.blue())
            self._populate_id_list()

    def _reset_colors(self):
        self._id_colors = dict(DEFAULT_ID_COLORS)
        self._populate_id_list()

    def _save_result(self):
        if self._last_annotated is None:
            QMessageBox.warning(self, "Nothing to Save", "Run the pipeline first.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Result", "seam_path_result.png", "PNG (*.png)")
        if p:
            cv2.imwrite(p, cv2.cvtColor(self._last_annotated, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Saved", f"Result saved to:\n{p}")


if __name__ == "__main__":
    run_app(MaskPipelineTester)
