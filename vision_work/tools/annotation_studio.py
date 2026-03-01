"""
Annotation Studio — PySide6 Tool (Large Composite)
====================================================
YOLO-assisted semi-automatic label editor.

Workflow:
  1. Load an image
  2. Run YOLO to pre-fill bounding boxes (or draw manually)
  3. Click any box → select it (highlighted)
  4. Drag box corners to nudge them
  5. Right-click → delete box
  6. Assign class label by typing in the class field
  7. Export as YOLO .txt annotation file (class_id cx cy w h normalised)

Tools:
  - Add Box: drag on canvas to draw a new bounding box
  - Select: click inside a box to select it
  - Delete: right-click selected box
  - Class edit: sidebar text input updates selected box's class
"""
import sys
import os
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QSlider, QFileDialog, QComboBox,
    QMessageBox, QLineEdit, QListWidget, QListWidgetItem, QFrame
)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPen, QBrush, QColor, QFont

# Re-use the base imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsItem, QGraphicsScene

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False


HANDLE_SZ = 8
COLORS_HEX = ["#f38ba8", "#a6e3a1", "#89dceb", "#f9e2af", "#cba6f7", "#89b4fa"]


class BoxItem(QGraphicsRectItem):
    """A selectable, draggable bounding box on the QGraphicsScene."""
    def __init__(self, x, y, w, h, cls_id=0, cls_name="object"):
        super().__init__(x, y, w, h)
        self.cls_id = cls_id
        self.cls_name = cls_name
        color = QColor(COLORS_HEX[cls_id % len(COLORS_HEX)])
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self._label = None

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.isSelected():
            r = self.rect()
            painter.setBrush(QBrush(QColor(255, 255, 255, 60)))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawRect(r)
        # Draw label
        painter.setPen(QColor(COLORS_HEX[self.cls_id % len(COLORS_HEX)]))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(self.rect().topLeft() + QPointF(2, -4), f"{self.cls_id}:{self.cls_name}")


class AnnotationStudio(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Annotation Studio — YOLO-Assisted Label Editor", width=1400, height=860)
        self._model = None
        self._conf = 0.30
        self._class_names = {}
        self._drawing = False
        self._draw_start = None
        self._draw_rect = None

        self._setup_ui()
        self._setup_canvas_events()
        self.image_loaded.connect(self._on_image_loaded)

    # ─────────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self._add_section_header("Input Image")
        self._build_default_image_loader()

        self._add_section_header("YOLO Auto-Fill")
        self._model_lbl = QLabel("No model loaded")
        self._model_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self._model_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self._model_lbl)
        self._add_button("🧠 Load YOLO Weights", self._load_model)

        conf_row = QHBoxLayout()
        self._lbl_conf = QLabel("0.30")
        self._lbl_conf.setStyleSheet(f"color: {C['accent']}; font-weight: bold;")
        self._slider_conf = QSlider(Qt.Horizontal)
        self._slider_conf.setRange(1, 99)
        self._slider_conf.setValue(30)
        self._slider_conf.valueChanged.connect(self._on_conf_change)
        conf_row.addWidget(self._slider_conf)
        conf_row.addWidget(self._lbl_conf)
        self.sidebar_layout.addLayout(conf_row)

        self._add_button("🤖 Run YOLO → Pre-fill Boxes", self._run_yolo_fill, primary=True)

        self._add_section_header("Manual Drawing")
        self._add_button("➕ Add Box (drag on canvas)", self._enable_draw_mode)
        self._add_button("🗑 Delete Selected Box", self._delete_selected)
        self._add_button("🗑 Clear All Boxes", self._clear_all)

        self._add_section_header("Selected Box — Class")
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("Class name (e.g. seam)")
        self.sidebar_layout.addWidget(self.class_edit)

        cls_id_row = QHBoxLayout()
        cls_id_row.addWidget(QLabel("Class ID:"))
        self.cls_id_spin = QComboBox()
        for i in range(20):
            self.cls_id_spin.addItem(str(i))
        cls_id_row.addWidget(self.cls_id_spin)
        self.sidebar_layout.addLayout(cls_id_row)

        self._add_button("✅ Apply to Selected Box", self._apply_class)

        self._add_section_header("Box List")
        self._box_list = QListWidget()
        self._box_list.setMaximumHeight(140)
        self._box_list.setFont(QFont("Monospace", 9))
        self._box_list.setStyleSheet(
            f"background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border']};"
        )
        self._box_list.currentRowChanged.connect(self._select_box_from_list)
        self.sidebar_layout.addWidget(self._box_list)

        self._add_section_header("Export")
        self._add_button("💾 Export YOLO .txt Annotation", self._export_yolo)
        self._add_button("💾 Save Annotated Image", self._export_image)

        self.sidebar_layout.addStretch()

    def _setup_canvas_events(self):
        self._draw_mode = False
        original_press = self.view.mousePressEvent
        original_move = self.view.mouseMoveEvent
        original_release = self.view.mouseReleaseEvent

        def _press(event):
            if self._draw_mode and event.button() == Qt.LeftButton:
                self._drawing = True
                self._draw_start = self.view.mapToScene(event.pos())
                if self._draw_rect:
                    self.scene.removeItem(self._draw_rect)
                self._draw_rect = self.scene.addRect(
                    QRectF(self._draw_start, self._draw_start),
                    QPen(QColor("#89b4fa"), 2)
                )
            else:
                original_press(event)

        def _move(event):
            if self._drawing and self._draw_rect and self._draw_start:
                pos = self.view.mapToScene(event.pos())
                r = QRectF(self._draw_start, pos).normalized()
                self._draw_rect.setRect(r)
            else:
                original_move(event)

        def _release(event):
            if self._drawing and event.button() == Qt.LeftButton:
                self._drawing = False
                if self._draw_rect:
                    r = self._draw_rect.rect()
                    self.scene.removeItem(self._draw_rect)
                    self._draw_rect = None
                    if r.width() > 5 and r.height() > 5:
                        cls_id = self.cls_id_spin.currentIndex()
                        cls_name = self.class_edit.text().strip() or "object"
                        self._add_box_item(r.x(), r.y(), r.width(), r.height(), cls_id, cls_name)
                self._draw_mode = False
                self.view.setCursor(Qt.ArrowCursor)
            else:
                original_release(event)

        self.view.mousePressEvent = _press
        self.view.mouseMoveEvent = _move
        self.view.mouseReleaseEvent = _release

    # ─────────────────────────────────────────────────────────────────────────
    def _on_image_loaded(self, rgb):
        # Clear existing boxes
        for item in self.scene.items():
            if isinstance(item, BoxItem):
                self.scene.removeItem(item)
        self._refresh_box_list()

    def _load_model(self):
        if not ULTRALYTICS_OK:
            QMessageBox.critical(self, "Error", "ultralytics not installed."); return
        p, _ = QFileDialog.getOpenFileName(self, "Open YOLO Weights", "", "Models (*.pt *.onnx)")
        if p:
            try:
                self._model = YOLO(p)
                self._model_lbl.setText(os.path.basename(p))
                self._class_names = self._model.names
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _run_yolo_fill(self):
        if self._rgb is None or self._model is None:
            QMessageBox.warning(self, "Not ready", "Load both an image and model first.")
            return
        results = self._model.predict(self._rgb, conf=self._conf, verbose=False)
        if not results or not results[0].boxes:
            QMessageBox.information(self, "No detections", "YOLO found no objects at this threshold.")
            return

        # Clear old auto-boxes
        for item in list(self.scene.items()):
            if isinstance(item, BoxItem):
                self.scene.removeItem(item)

        img_h, img_w = self._rgb.shape[:2]
        scene_rect = self.scene.sceneRect()
        sx = scene_rect.width() / img_w
        sy = scene_rect.height() / img_h

        names = results[0].names
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, f"cls{cls_id}")
            self._add_box_item(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy, cls_id, cls_name)

        self._refresh_box_list()

    def _add_box_item(self, x, y, w, h, cls_id=0, cls_name="object"):
        box = BoxItem(x, y, w, h, cls_id, cls_name)
        self.scene.addItem(box)
        self._refresh_box_list()

    def _enable_draw_mode(self):
        self._draw_mode = True
        self.view.setCursor(Qt.CrossCursor)

    def _delete_selected(self):
        for item in self.scene.selectedItems():
            if isinstance(item, BoxItem):
                self.scene.removeItem(item)
        self._refresh_box_list()

    def _clear_all(self):
        for item in list(self.scene.items()):
            if isinstance(item, BoxItem):
                self.scene.removeItem(item)
        self._refresh_box_list()

    def _apply_class(self):
        cls_id = self.cls_id_spin.currentIndex()
        cls_name = self.class_edit.text().strip() or "object"
        for item in self.scene.selectedItems():
            if isinstance(item, BoxItem):
                item.cls_id = cls_id
                item.cls_name = cls_name
                color = QColor(COLORS_HEX[cls_id % len(COLORS_HEX)])
                pen = QPen(color, 2)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.update()
        self._refresh_box_list()

    def _on_conf_change(self, val):
        self._conf = val / 100.0
        self._lbl_conf.setText(f"{self._conf:.2f}")

    def _refresh_box_list(self):
        self._box_list.clear()
        for i, item in enumerate(self._get_boxes()):
            r = item.rect()
            pos = item.pos()
            self._box_list.addItem(
                f"[{i}] {item.cls_name} ({item.cls_id}) "
                f"x={int(r.x()+pos.x())} y={int(r.y()+pos.y())} "
                f"w={int(r.width())} h={int(r.height())}"
            )

    def _select_box_from_list(self, row):
        boxes = self._get_boxes()
        self.scene.clearSelection()
        if 0 <= row < len(boxes):
            boxes[row].setSelected(True)

    def _get_boxes(self):
        return [item for item in self.scene.items() if isinstance(item, BoxItem)]

    # ─────────────────────────────────────────────────────────────────────────
    def _export_yolo(self):
        if self._rgb is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save YOLO Annotation", "annotation.txt", "Text (*.txt)")
        if not p:
            return
        img_h, img_w = self._rgb.shape[:2]
        scene_rect = self.scene.sceneRect()
        sx = img_w / scene_rect.width()
        sy = img_h / scene_rect.height()
        lines = []
        for item in self._get_boxes():
            r = item.rect()
            pos = item.pos()
            x1 = (r.x() + pos.x()) * sx
            y1 = (r.y() + pos.y()) * sy
            w = r.width() * sx
            h = r.height() * sy
            cx = (x1 + w / 2) / img_w
            cy = (y1 + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{item.cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(p, 'w') as f:
            f.write('\n'.join(lines))
        QMessageBox.information(self, "Saved", f"{len(lines)} annotations saved to:\n{p}")

    def _export_image(self):
        if self._rgb is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Annotated Image", "annotated.png", "PNG (*.png)")
        if not p:
            return
        img = self._rgb.copy()
        img_h, img_w = img.shape[:2]
        scene_rect = self.scene.sceneRect()
        sx = img_w / scene_rect.width()
        sy = img_h / scene_rect.height()
        for item in self._get_boxes():
            r = item.rect()
            pos = item.pos()
            x1 = int((r.x() + pos.x()) * sx)
            y1 = int((r.y() + pos.y()) * sy)
            x2 = int((r.x() + pos.x() + r.width()) * sx)
            y2 = int((r.y() + pos.y() + r.height()) * sy)
            color_hex = COLORS_HEX[item.cls_id % len(COLORS_HEX)].lstrip('#')
            cr, cg, cb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            cv2.rectangle(img, (x1, y1), (x2, y2), (cr, cg, cb), 2)
            cv2.putText(img, f"{item.cls_id}:{item.cls_name}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        QMessageBox.information(self, "Saved", f"Annotated image saved to:\n{p}")


if __name__ == "__main__":
    run_app(AnnotationStudio)
