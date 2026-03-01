"""
YOLO Inspector — PySide6 Tool
==============================
Modern replacement for yolo_gui.py.
Clean, single-responsibility, built on BaseVisionApp.

Features:
 - Load any Ultralytics YOLO .pt weights
 - Real-time confidence threshold slider
 - Object ID labels [ID: i] on all bounding boxes
 - Per-object-ID color picker (click any detected object ID to assign colour)
 - View modes: Original / Bounding Boxes / Segmentation Mask / Polygon
 - Live FPS counter on inference
 - Single image + batch folder support
"""
import sys
import os
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QSlider, QColorDialog,
    QFileDialog, QComboBox, QDoubleSpinBox, QFrame, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QFont

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False


# Auto-palette for when no manual color is assigned
AUTO_PALETTE_BGR = [
    (50, 220, 80),    # Green
    (50, 50, 220),    # Red
    (220, 50, 50),    # Blue
    (50, 200, 220),   # Yellow
    (220, 50, 220),   # Magenta
    (220, 180, 50),   # Cyan
    (50, 130, 220),   # Orange
    (140, 50, 220),   # Purple
]


class YoloInspector(BaseVisionApp):
    def __init__(self):
        super().__init__(title="YOLO Inspector", width=1350, height=820)
        self._model = None
        self._results = None
        self._id_colors_bgr = {}     # {object_id: (B,G,R)}
        self._conf = 0.30
        self._view_mode = "Bounding Boxes"
        self._setup_ui()
        self.image_loaded.connect(self._run_inference)

    def _setup_ui(self):
        # Input Image
        self._add_section_header("Input Image")
        self._build_default_image_loader()

        # Model
        self._add_section_header("Model Weights")
        self.model_entry = QLabel("No model loaded")
        self.model_entry.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.model_entry.setWordWrap(True)
        self.sidebar_layout.addWidget(self.model_entry)
        self._add_button("🧠  Load Weights (.pt)", self._load_model)

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

        # View Mode
        self._add_section_header("View Mode")
        self._view_combo = QComboBox()
        for mode in ["Original", "Bounding Boxes", "Segmentation Mask", "Polygon", "Solid Mask"]:
            self._view_combo.addItem(mode)
        self._view_combo.setCurrentText("Bounding Boxes")
        self._view_combo.currentTextChanged.connect(self._on_view_mode_change)
        self.sidebar_layout.addWidget(self._view_combo)

        # Auto Multi-Color toggle
        from PySide6.QtWidgets import QCheckBox
        self._chk_auto = QCheckBox("Auto Multi-Color (palette by ID)")
        self._chk_auto.setStyleSheet(f"color: {C['text']};")
        self._chk_auto.setChecked(True)
        self._chk_auto.toggled.connect(lambda _: self._render())
        self.sidebar_layout.addWidget(self._chk_auto)

        # Per-ID Color assignment
        self._add_section_header("Per-ID Color Picker")
        self._id_list = QListWidget()
        self._id_list.setMaximumHeight(130)
        self._id_list.setFont(QFont("Monospace", 10))
        self._id_list.setStyleSheet(
            f"background-color: {C['panel']}; color: {C['text']};"
            f"border: 1px solid {C['border']};"
        )
        self.sidebar_layout.addWidget(self._id_list)

        btn_assign = QPushButton("🎨  Assign Color to Selected ID")
        btn_assign.clicked.connect(self._assign_id_color)
        self.sidebar_layout.addWidget(btn_assign)

        btn_clear_colors = QPushButton("↩  Clear All ID Colors")
        btn_clear_colors.clicked.connect(self._clear_id_colors)
        self.sidebar_layout.addWidget(btn_clear_colors)

        # Stats
        self._add_section_header("Performance")
        self.lbl_stats = QLabel("Inference: -- ms  | Objects: --")
        self.lbl_stats.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.lbl_stats.setWordWrap(True)
        self.sidebar_layout.addWidget(self.lbl_stats)

        self.sidebar_layout.addStretch()

    # ─────────────────────────────────────────────────────────────────────────
    # Model & Inference
    # ─────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        if not ULTRALYTICS_OK:
            QMessageBox.critical(self, "Error", "ultralytics is not installed.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open YOLO Weights", "", "Models (*.pt *.onnx)")
        if p:
            try:
                self._model = YOLO(p)
                self.model_entry.setText(os.path.basename(p))
                if self._rgb is not None:
                    self._run_inference(self._rgb)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _run_inference(self, rgb: np.ndarray):
        if self._model is None:
            return
        t0 = time.perf_counter()
        self._results = self._model.predict(rgb, conf=self._conf, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000
        n = len(self._results[0].boxes) if self._results and self._results[0].boxes else 0
        self.lbl_stats.setText(f"Inference: {elapsed:.1f} ms  |  Objects: {n}")
        self._populate_id_list()
        self._render()

    def _populate_id_list(self):
        self._id_list.clear()
        if not self._results:
            return
        boxes = self._results[0].boxes
        names = self._results[0].names
        if not boxes:
            return
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            color = self._id_colors_bgr.get(i)
            col_str = f"rgb({color[2]},{color[1]},{color[0]})" if color else "—"
            item = QListWidgetItem(f"ID {i}: {names[cls]} ({conf:.2f})  [{col_str}]")
            if color:
                item.setBackground(QColor(color[2], color[1], color[0]))
                item.setForeground(QColor("#000000"))
            self._id_list.addItem(item)

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────
    def _render(self):
        if self._rgb is None:
            return
        mode = self._view_mode
        if mode == "Original" or not self._results or not self._results[0].boxes:
            self._display_rgb(self._rgb)
            return

        img = self._rgb.copy()
        boxes = self._results[0].boxes
        masks = self._results[0].masks
        names = self._results[0].names

        def get_bgr(i):
            if i in self._id_colors_bgr and not self._chk_auto.isChecked():
                return self._id_colors_bgr[i]
            if self._chk_auto.isChecked() or i not in self._id_colors_bgr:
                return AUTO_PALETTE_BGR[i % len(AUTO_PALETTE_BGR)]
            return (50, 50, 220)

        for i, box in enumerate(boxes):
            bgr = get_bgr(i)
            rgb = (bgr[2], bgr[1], bgr[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"[ID:{i}] {names[cls]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if mode == "Bounding Boxes":
                cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
                cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif mode == "Solid Mask":
                cv2.rectangle(img, (x1, y1), (x2, y2), rgb, -1)
                cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif mode in ("Segmentation Mask", "Polygon") and masks:
                raw = masks.data[i].cpu().numpy()
                h, w = img.shape[:2]
                msk = cv2.resize(raw, (w, h)) > 0.5

                if mode == "Segmentation Mask":
                    overlay = img.copy()
                    overlay[msk] = np.array(rgb, dtype=np.uint8)
                    img = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)
                    cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                elif mode == "Polygon":
                    contours, _ = cv2.findContours(
                        msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(img, contours, -1, rgb, 2)
                    cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self._display_rgb(img)

    # ─────────────────────────────────────────────────────────────────────────
    # Controls
    # ─────────────────────────────────────────────────────────────────────────
    def _on_conf_change(self, val):
        self._conf = val / 100.0
        self._lbl_conf.setText(f"{self._conf:.2f}")
        if self._rgb is not None and self._model is not None:
            self._run_inference(self._rgb)

    def _on_view_mode_change(self, text):
        self._view_mode = text
        self._render()

    def _assign_id_color(self):
        sel = self._id_list.currentRow()
        if sel < 0:
            return
        result = QColorDialog.getColor(parent=self, title=f"Color for Object ID {sel}")
        if result.isValid():
            self._id_colors_bgr[sel] = (result.blue(), result.green(), result.red())
            self._populate_id_list()
            self._render()

    def _clear_id_colors(self):
        self._id_colors_bgr.clear()
        self._populate_id_list()
        self._render()


if __name__ == "__main__":
    run_app(YoloInspector)
