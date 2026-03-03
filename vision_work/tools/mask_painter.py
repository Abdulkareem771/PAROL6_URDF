"""
Mask Painter — PySide6 Tool
============================
A standalone OpenCV brush tool for manually painting color masks on images.
Feeds directly into the Pipeline Prototyper and Pipeline Studio.

Brush modes:
 - Paint: draw filled circles in the selected color
 - Erase: clear painted pixels back to original

Designed to:
 1. Manually paint workpiece regions Green/Red for detect_path.py style pipelines
 2. Generate ground-truth masks for ML dataset creation
 3. Human-in-the-loop path annotation (Approach C)
"""
import sys
import os
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QSlider, QColorDialog,
    QFileDialog, QButtonGroup, QRadioButton, QMessageBox
)
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C


class MaskPainter(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Mask Painter — Manual Color Mask Tool", width=1350, height=800)

        # State
        self._mask = None          # Float32 RGBA painted overlay
        self._brush_color = (0, 200, 80)  # Default: green (RGB)
        self._brush_size = 18
        self._is_painting = False
        self._erasing = False

        self._setup_ui()
        self._connect_canvas_events()

    # ─────────────────────────────────────────────────────────────────────────
    # UI Setup
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.image_loaded.connect(self._on_new_image)

        # Input
        self._add_section_header("Input Image")
        self._build_default_image_loader()

        # Brush Color
        self._add_section_header("Brush Color")
        self._color_preview = QLabel()
        self._color_preview.setFixedHeight(32)
        self._color_preview.setStyleSheet(f"background-color: rgb(0,200,80); border-radius: 4px;")
        self.sidebar_layout.addWidget(self._color_preview)

        color_btns = QHBoxLayout()
        btn_pick = QPushButton("🎨 Pick")
        btn_pick.clicked.connect(self._pick_color)
        btn_green = QPushButton("🟢 Green")
        btn_green.clicked.connect(lambda: self._set_color(0, 200, 80))
        btn_red = QPushButton("🔴 Red")
        btn_red.clicked.connect(lambda: self._set_color(220, 40, 40))
        btn_blue = QPushButton("🔵 Blue")
        btn_blue.clicked.connect(lambda: self._set_color(40, 100, 220))
        color_btns.addWidget(btn_pick)
        color_btns.addWidget(btn_green)
        color_btns.addWidget(btn_red)
        color_btns.addWidget(btn_blue)
        self.sidebar_layout.addLayout(color_btns)

        # Brush Mode
        self._add_section_header("Brush Mode")
        mode_layout = QHBoxLayout()
        self._mode_group = QButtonGroup()
        self._rb_paint = QRadioButton("🖌 Paint")
        self._rb_erase = QRadioButton("🗑 Erase")
        self._rb_paint.setChecked(True)
        self._mode_group.addButton(self._rb_paint)
        self._mode_group.addButton(self._rb_erase)
        self._rb_paint.toggled.connect(lambda c: setattr(self, '_erasing', not c))
        for rb in [self._rb_paint, self._rb_erase]:
            rb.setStyleSheet(f"color: {C['text']};")
            mode_layout.addWidget(rb)
        self.sidebar_layout.addLayout(mode_layout)

        # Brush Size
        self._add_section_header("Brush Size")
        size_row = QHBoxLayout()
        self._lbl_size = QLabel(f"Size: {self._brush_size}px")
        self._lbl_size.setStyleSheet(f"color: {C['text2']};")
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(2, 80)
        self._slider.setValue(self._brush_size)
        self._slider.valueChanged.connect(self._on_size_change)
        size_row.addWidget(self._slider)
        size_row.addWidget(self._lbl_size)
        self.sidebar_layout.addLayout(size_row)

        # Actions
        self._add_section_header("Actions")
        self._add_button("↩  Undo Last Stroke", self._undo)
        self._add_button("🗑  Clear All Paint", self._clear_mask)
        self._add_button("💾  Export Painted Image", self._export_painted, primary=True)
        self._add_button("💾  Export Mask Only (RGBA)", self._export_mask_only)

        self.sidebar_layout.addStretch()

    # ─────────────────────────────────────────────────────────────────────────
    # Canvas Events
    # ─────────────────────────────────────────────────────────────────────────
    def _connect_canvas_events(self):
        self.view.mousePressEvent = self._canvas_press
        self.view.mouseMoveEvent = self._canvas_move
        self.view.mouseReleaseEvent = self._canvas_release

    def _canvas_press(self, event):
        if event.button() == Qt.MiddleButton:
            # Pass through for panning
            from PySide6.QtGui import QGraphicsView
            self._is_panning = True
            self.view.setDragMode(self.view.DragMode.ScrollHandDrag)
            event.ignore()
            return
        if event.button() == Qt.LeftButton and self._rgb is not None:
            self._undo_stack_push()
            self._is_painting = True
            self._paint_at(event)
        event.accept()

    def _canvas_move(self, event):
        if self._is_painting and self._rgb is not None:
            self._paint_at(event)
        event.accept()

    def _canvas_release(self, event):
        self._is_painting = False
        self.view.setDragMode(self.view.DragMode.NoDrag)
        event.accept()

    def _paint_at(self, event):
        """Map Qt view coordinates → image pixel coordinates → paint on mask."""
        if self._mask is None or self._rgb is None:
            return

        scene_pos = self.view.mapToScene(event.pos())
        img_h, img_w = self._rgb.shape[:2]
        scene_rect = self.scene.sceneRect()
        x = int(scene_pos.x() / scene_rect.width()  * img_w)
        y = int(scene_pos.y() / scene_rect.height() * img_h)

        r, g, b = self._brush_color
        if self._erasing:
            cv2.circle(self._mask, (x, y), self._brush_size, (0, 0, 0, 0), -1)
        else:
            cv2.circle(self._mask, (x, y), self._brush_size, (b, g, r, 180), -1)

        self._render_composite()

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────
    def _render_composite(self):
        if self._rgb is None or self._mask is None:
            return

        base = self._rgb.copy().astype(np.float32)
        alpha = self._mask[:, :, 3:4] / 255.0
        mask_rgb = self._mask[:, :, :3][:, :, ::-1].astype(np.float32)  # BGR→RGB

        composite = base * (1 - alpha) + mask_rgb * alpha
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        self._display_rgb(composite)

    def _on_new_image(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        self._mask = np.zeros((h, w, 4), dtype=np.uint8)
        self._undo_stack = []

    # ─────────────────────────────────────────────────────────────────────────
    # Undo
    # ─────────────────────────────────────────────────────────────────────────
    def _undo_stack_push(self):
        if self._mask is not None:
            if not hasattr(self, '_undo_stack'):
                self._undo_stack = []
            self._undo_stack.append(self._mask.copy())
            if len(self._undo_stack) > 20:
                self._undo_stack.pop(0)

    def _undo(self):
        if hasattr(self, '_undo_stack') and self._undo_stack:
            self._mask = self._undo_stack.pop()
            self._render_composite()

    # ─────────────────────────────────────────────────────────────────────────
    # Color helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _pick_color(self):
        r, g, b = self._brush_color
        initial = f"#{r:02x}{g:02x}{b:02x}"
        result = QColorDialog.getColor(parent=self, title="Pick Brush Color",
                                       options=QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if result.isValid():
            self._set_color(result.red(), result.green(), result.blue())

    def _set_color(self, r, g, b):
        self._brush_color = (r, g, b)
        self._color_preview.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border-radius: 4px;"
        )

    def _on_size_change(self, val):
        self._brush_size = val
        self._lbl_size.setText(f"Size: {val}px")

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────
    def _clear_mask(self):
        if self._mask is not None:
            self._undo_stack_push()
            self._mask[:] = 0
            self._display_rgb(self._rgb)

    def _export_painted(self):
        if self._rgb is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Painted Image", "painted_result.png", "PNG (*.png)")
        if p:
            base = self._rgb.copy().astype(np.float32)
            alpha = self._mask[:, :, 3:4] / 255.0
            mask_rgb = self._mask[:, :, :3][:, :, ::-1].astype(np.float32)
            composite = base * (1 - alpha) + mask_rgb * alpha
            composite = np.clip(composite, 0, 255).astype(np.uint8)
            cv2.imwrite(p, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Saved", f"Painted image saved to:\n{p}")

    def _export_mask_only(self):
        if self._mask is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Mask", "mask.png", "PNG (*.png)")
        if p:
            cv2.imwrite(p, self._mask)
            QMessageBox.information(self, "Saved", f"Mask saved to:\n{p}")


if __name__ == "__main__":
    run_app(MaskPainter)
