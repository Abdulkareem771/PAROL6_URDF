"""
Mask Pipeline Tester — PySide6 Tool
=====================================
End-to-end test harness for the PAROL6 mask-based path detection pipeline.

Pipeline:
  1. Load an image (local file, folder, or ROS topic)
  2. Run YOLO segmentation → each detected object ID gets a configurable mask color
     (default: ID 0 = Green, ID 1 = Red — matching detect_path.py convention)
  3. Apply the path-detection algorithm (from detect_path.py) inline:
       - Build HSV masks from the painted green/red regions
       - Compute bounding boxes around each colour region
       - Compute the intersection bounding box (the seam path)
  4. Visualise: G mask | R mask | Annotated overlay with intersection

Design: The path-detection logic is reproduced faithfully here so we never
need to import or modify the teammate's script.
"""
import sys
import os
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QSlider, QComboBox, QColorDialog,
    QFileDialog, QMessageBox, QCheckBox, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QVBoxLayout, QGridLayout
)
from PySide6.QtCore import Qt
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


def _find_seam_path(rgb_image: np.ndarray):
    """
    Inline reimplementation of detect_path.segment_blocks() logic.
    Accepts an RGB numpy array, returns:
      (G_mask, R_mask, annotated_rgb, bbox_G, bbox_R, bbox_I)
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
        cv2.putText(annotated, "Green Block", (bbox_G[0], bbox_G[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    if bbox_R:
        cv2.rectangle(annotated, bbox_R[:2], bbox_R[2:], (255, 60, 60), 2)
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
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (255, 255, 0), 3)
            cv2.putText(annotated, "SEAM PATH", (ix1, iy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    return G, R, annotated, bbox_G, bbox_R, bbox_I


class MaskPipelineTester(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Mask Pipeline Tester — YOLO → Colour Mask → Path Detection",
                         width=1500, height=860)
        self._model = None
        self._results = None
        self._conf = 0.30
        self._id_colors = dict(DEFAULT_ID_COLORS)
        self._last_G = None
        self._last_R = None
        self._setup_ui()
        self._build_quad_canvas()
        self.image_loaded.connect(self._on_image_loaded)

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

        # Per-ID Colour Mapping
        self._add_section_header("Object ID → Mask Colour")
        info = QLabel("ID 0 = Green (seam side A)\nID 1 = Red (seam side B)\nClick ID to change colour.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {C['text2']}; font-size: 10px;")
        self.sidebar_layout.addWidget(info)

        self._id_list = QListWidget()
        self._id_list.setMaximumHeight(120)
        self._id_list.setFont(QFont("Monospace", 10))
        self._id_list.setStyleSheet(
            f"background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border']};"
        )
        self._id_list.itemDoubleClicked.connect(self._pick_id_color)
        self.sidebar_layout.addWidget(self._id_list)

        self._add_button("🎨  Change Colour for Selected ID", self._pick_id_color_btn)
        self._add_button("↩  Reset to Green/Red Defaults", self._reset_colors)

        # Stats
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
        # Remove the current QGraphicsView (right side of splitter)
        old_view = self.splitter.widget(1)
        old_view.setParent(None)

        quad = QFrame()
        quad_layout = QGridLayout(quad)
        quad_layout.setSpacing(4)
        quad_layout.setContentsMargins(4, 4, 4, 4)

        self._panels = {}
        titles = ["Original + YOLO Masks", "G Mask (Green Region)", "R Mask (Red Region)", "Path Detection Result"]
        for idx, title in enumerate(titles):
            row, col = divmod(idx, 2)
            frame = QFrame()
            frame.setStyleSheet(f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 4px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.setSpacing(2)
            lbl_title = QLabel(f" {title}")
            lbl_title.setStyleSheet(f"color: {C['text2']}; font-size: 10px; font-weight: bold; background: {C['panel']}; padding: 3px;")
            fl.addWidget(lbl_title)
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setScaledContents(False)
            img_lbl.setSizePolicy(img_lbl.sizePolicy().Expanding, img_lbl.sizePolicy().Expanding)
            img_lbl.setMinimumSize(200, 150)
            fl.addWidget(img_lbl)
            self._panels[idx] = img_lbl
            quad_layout.addWidget(frame, row, col)

        self.splitter.addWidget(quad)
        self.splitter.setSizes([340, 1160])

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

    # ─────────────────────────────────────────────────────────────────────────
    def _run_pipeline(self):
        if self._rgb is None:
            return
        t0 = time.perf_counter()
        img = self._rgb.copy()

        # --- Step 1: YOLO Inference ---
        if self._model is not None:
            self._results = self._model.predict(img, conf=self._conf, verbose=False)
            img = self._apply_id_masks(img, self._results)
            self._populate_id_list()

        self._show_panel(0, img)

        # --- Step 2: Path Detection ---
        G, R, annotated, bbox_G, bbox_R, bbox_I = _find_seam_path(img)

        # Show quad panels
        self._show_panel(1, cv2.cvtColor(cv2.merge([G, G, G]), cv2.COLOR_BGR2RGB))
        self._show_panel(2, cv2.cvtColor(cv2.merge([R, R, R]), cv2.COLOR_BGR2RGB))
        self._show_panel(3, annotated)
        self._last_annotated = annotated

        elapsed = (time.perf_counter() - t0) * 1000
        g_px = int(np.sum(G > 0)) if G is not None else 0
        r_px = int(np.sum(R > 0)) if R is not None else 0
        if bbox_I:
            w = bbox_I[2] - bbox_I[0]; h = bbox_I[3] - bbox_I[1]
            self.lbl_result.setText(
                f"✅ Seam path found!\n"
                f"Intersection: ({bbox_I[0]},{bbox_I[1]}) → ({bbox_I[2]},{bbox_I[3]})\n"
                f"Size: {w}×{h}px\n"
                f"Green pixels: {g_px:,}  Red pixels: {r_px:,}\n"
                f"Pipeline: {elapsed:.1f} ms"
            )
        else:
            self.lbl_result.setText(
                f"⚠️  No seam intersection found.\n"
                f"Green px: {g_px:,}  Red px: {r_px:,}\n"
                f"Check that ID 0 = green, ID 1 = red in colours.\n"
                f"Pipeline: {elapsed:.1f} ms"
            )

    def _apply_id_masks(self, rgb: np.ndarray, results) -> np.ndarray:
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
                # Fallback: fill bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # Draw label
            x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            cv2.putText(img, f"ID:{i}", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return img

    def _show_panel(self, idx: int, rgb: np.ndarray):
        if rgb is None:
            return
        lbl = self._panels[idx]
        available = lbl.size()
        scaled = cv2.resize(rgb, (max(available.width(), 100), max(available.height() - 30, 80)))
        h, w, ch = scaled.shape
        qimg = QImage(scaled.data, w, h, ch * w, QImage.Format_RGB888)
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
        if not hasattr(self, '_last_annotated') or self._last_annotated is None:
            QMessageBox.warning(self, "Nothing to Save", "Run the pipeline first.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Result", "seam_path_result.png", "PNG (*.png)")
        if p:
            cv2.imwrite(p, cv2.cvtColor(self._last_annotated, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Saved", f"Result saved to:\n{p}")


if __name__ == "__main__":
    run_app(MaskPipelineTester)
