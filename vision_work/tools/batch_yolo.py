"""
Batch YOLO Exporter — PySide6 Tool
=====================================
Load a YOLO model, select a folder of images, and batch-export anything
you want from the detections. Single-purpose: no inference viewer noise.

Features
 - Select which object IDs to export (e.g. only ID 0 and ID 2)
 - Operations:
     • Crop: save each bounding box as an individual image file
     • Solid Mask: paint the bbox region in a chosen colour, save full image
     • Seg Mask: save the Per-pixel segmentation mask as a B&W PNG
     • Colour Mask: save the segmentation region painted in a given colour
     • Annotated Image: draw boxes + labels on image and save
 - Confidence threshold slider
 - Auto-subfolders per class name so results stay organized
 - Progress bar so you know it hasn't died

Output structure:
  <output_folder>/
    crops/        (Crop mode)
    solid_masks/  (Solid Mask mode)
    seg_masks/    (Seg Mask mode)
    colour_masks/ (Colour Mask mode)
    annotated/    (Annotated Image mode)
"""
import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QSlider,
    QFileDialog, QMessageBox, QCheckBox, QListWidget, QListWidgetItem,
    QProgressBar, QComboBox, QFrame, QColorDialog
)
from PySide6.QtCore import Qt, QThread, Signal
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

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class BatchWorker(QThread):
    """Runs the batch job off the main thread."""
    progress = Signal(int, int)       # (current, total)
    log = Signal(str)
    done = Signal(int, float)         # (saved_count, elapsed_s)

    def __init__(self, model, image_paths, out_dir, mode, target_ids,
                 conf, mask_color, save_all_ids):
        super().__init__()
        self.model = model
        self.image_paths = image_paths
        self.out_dir = out_dir
        self.mode = mode
        self.target_ids = target_ids
        self.conf = conf
        self.mask_color = mask_color  # (R, G, B)
        self.save_all_ids = save_all_ids

    def run(self):
        t0 = time.perf_counter()
        total = len(self.image_paths)
        saved = 0
        os.makedirs(self.out_dir, exist_ok=True)

        for idx, img_path in enumerate(self.image_paths):
            self.progress.emit(idx + 1, total)
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            try:
                results = self.model.predict(rgb, conf=self.conf, verbose=False)
            except Exception as e:
                self.log.emit(f"[Error] {img_path}: {e}")
                continue

            if not results or not results[0].boxes:
                continue

            boxes = results[0].boxes
            masks_data = results[0].masks
            names = results[0].names
            stem = Path(img_path).stem
            img_h, img_w = rgb.shape[:2]

            annotated = rgb.copy()

            for det_i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, f"cls{cls_id}")
                conf_val = float(box.conf[0])

                # Filter by target IDs unless save_all_ids
                if not self.save_all_ids and det_i not in self.target_ids:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                mode = self.mode
                r, g, b = self.mask_color

                if mode == "Crop":
                    sub = os.path.join(self.out_dir, "crops", cls_name)
                    os.makedirs(sub, exist_ok=True)
                    crop = bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(os.path.join(sub, f"{stem}_id{det_i}.png"), crop)
                        saved += 1

                elif mode == "Solid Mask":
                    sub = os.path.join(self.out_dir, "solid_masks")
                    os.makedirs(sub, exist_ok=True)
                    out_img = rgb.copy()
                    out_img[y1:y2, x1:x2] = (r, g, b)
                    cv2.imwrite(os.path.join(sub, f"{stem}_id{det_i}.png"),
                                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                    saved += 1

                elif mode == "Seg Mask (B&W)":
                    sub = os.path.join(self.out_dir, "seg_masks")
                    os.makedirs(sub, exist_ok=True)
                    if masks_data and det_i < len(masks_data.data):
                        raw = masks_data.data[det_i].cpu().numpy()
                        msk = (cv2.resize(raw, (img_w, img_h)) > 0.5).astype(np.uint8) * 255
                        cv2.imwrite(os.path.join(sub, f"{stem}_id{det_i}.png"), msk)
                        saved += 1

                elif mode == "Colour Mask":
                    sub = os.path.join(self.out_dir, "colour_masks")
                    os.makedirs(sub, exist_ok=True)
                    out_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    if masks_data and det_i < len(masks_data.data):
                        raw = masks_data.data[det_i].cpu().numpy()
                        msk = cv2.resize(raw, (img_w, img_h)) > 0.5
                        out_img[msk] = (r, g, b)
                    cv2.imwrite(os.path.join(sub, f"{stem}_id{det_i}.png"),
                                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                    saved += 1

                elif mode == "Annotated Image":
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (r, g, b), 2)
                    cv2.putText(annotated, f"[{det_i}] {cls_name} {conf_val:.2f}",
                                (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if mode == "Annotated Image":
                sub = os.path.join(self.out_dir, "annotated")
                os.makedirs(sub, exist_ok=True)
                cv2.imwrite(os.path.join(sub, f"{stem}.png"),
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                saved += 1

        elapsed = time.perf_counter() - t0
        self.done.emit(saved, elapsed)


class BatchYolo(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Batch YOLO Exporter", width=1300, height=820)
        self._model = None
        self._conf = 0.30
        self._image_paths = []
        self._mask_color = (50, 200, 80)   # Default green
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        # Model
        self._add_section_header("YOLO Model")
        self._model_lbl = QLabel("No model loaded")
        self._model_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self._model_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self._model_lbl)
        self._add_button("🧠  Load Weights (.pt)", self._load_model)

        # Confidence
        self._add_section_header("Confidence Threshold")
        conf_row = QHBoxLayout()
        self._lbl_conf = QLabel(f"{self._conf:.2f}")
        self._lbl_conf.setStyleSheet(f"color: {C['accent']}; font-weight: bold;")
        self._sl_conf = QSlider(Qt.Horizontal)
        self._sl_conf.setRange(1, 99)
        self._sl_conf.setValue(30)
        self._sl_conf.valueChanged.connect(lambda v: (
            setattr(self, '_conf', v / 100.0),
            self._lbl_conf.setText(f"{self._conf:.2f}")))
        conf_row.addWidget(self._sl_conf)
        conf_row.addWidget(self._lbl_conf)
        self.sidebar_layout.addLayout(conf_row)

        # Image folder
        self._add_section_header("Input Images")
        self._folder_lbl = QLabel("No folder selected")
        self._folder_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self._folder_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self._folder_lbl)
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.sidebar_layout.addWidget(self._count_lbl)
        self._add_button("📁  Select Image Folder", self._select_folder)

        # Output folder
        self._add_section_header("Output Folder")
        self._out_lbl = QLabel("No output folder selected")
        self._out_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self._out_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self._out_lbl)
        self._add_button("💾  Select Output Folder", self._select_out)

        # Target IDs
        self._add_section_header("Target Object IDs")
        info = QLabel("Enter IDs to export (0,1,2) or leave empty for all.")
        info.setStyleSheet(f"color: {C['text2']}; font-size: 10px;")
        info.setWordWrap(True)
        self.sidebar_layout.addWidget(info)
        from PySide6.QtWidgets import QLineEdit
        self._ids_edit = QLineEdit()
        self._ids_edit.setPlaceholderText("e.g. 0,1  (empty = all IDs)")
        self.sidebar_layout.addWidget(self._ids_edit)

        # Export mode
        self._add_section_header("Export Mode")
        self._mode_combo = QComboBox()
        for m in ["Crop", "Solid Mask", "Seg Mask (B&W)", "Colour Mask", "Annotated Image"]:
            self._mode_combo.addItem(m)
        self.sidebar_layout.addWidget(self._mode_combo)

        # Mask colour
        self._add_section_header("Mask / Box Colour")
        self._col_preview = QLabel()
        self._col_preview.setFixedHeight(28)
        self._col_preview.setStyleSheet("background-color: rgb(50,200,80); border-radius: 4px;")
        self.sidebar_layout.addWidget(self._col_preview)
        col_btns = QHBoxLayout()
        for label, rgb in [("Green", (50,200,80)), ("Red", (220,40,40)), ("Blue", (40,100,220))]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, c=rgb: self._set_color(*c))
            col_btns.addWidget(btn)
        btn_pick = QPushButton("🎨")
        btn_pick.clicked.connect(self._pick_color)
        col_btns.addWidget(btn_pick)
        self.sidebar_layout.addLayout(col_btns)

        # Progress + Run
        self._add_section_header("Progress")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: {C['border']}; border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {C['accent']}; border-radius: 4px; }}"
        )
        self.sidebar_layout.addWidget(self._progress)
        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self._status_lbl)

        self.sidebar_layout.addStretch()
        self._run_btn = self._add_button("▶  Start Batch Export", self._run_batch, primary=True)

        # Canvas: show a preview of the selected folder contents as thumbnails
        # (reused as log/status display via the existing graphics scene)

    # ─────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        if not ULTRALYTICS_OK:
            QMessageBox.critical(self, "Error", "ultralytics is not installed."); return
        p, _ = QFileDialog.getOpenFileName(self, "YOLO Weights", "", "Models (*.pt *.onnx)")
        if p:
            try:
                self._model = YOLO(p)
                self._model_lbl.setText(os.path.basename(p))
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self._image_paths = [
                os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if os.path.splitext(f)[1].lower() in EXTS
            ]
            self._folder_lbl.setText(folder)
            self._count_lbl.setText(f"{len(self._image_paths)} images found")
            # Preview first image
            if self._image_paths:
                bgr = cv2.imread(self._image_paths[0])
                if bgr is not None:
                    self._display_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _select_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._out_folder = folder
            self._out_lbl.setText(folder)

    def _pick_color(self):
        r, g, b = self._mask_color
        result = QColorDialog.getColor(QColor(r, g, b), self, "Pick Mask Colour")
        if result.isValid():
            self._set_color(result.red(), result.green(), result.blue())

    def _set_color(self, r, g, b):
        self._mask_color = (r, g, b)
        self._col_preview.setStyleSheet(f"background-color: rgb({r},{g},{b}); border-radius: 4px;")

    def _run_batch(self):
        if not self._model:
            QMessageBox.warning(self, "No model", "Load YOLO weights first."); return
        if not self._image_paths:
            QMessageBox.warning(self, "No images", "Select an image folder first."); return
        if not hasattr(self, '_out_folder'):
            QMessageBox.warning(self, "No output", "Select an output folder first."); return

        # Parse target IDs
        raw = self._ids_edit.text().strip()
        if raw:
            try:
                target_ids = {int(x.strip()) for x in raw.split(',')}
            except ValueError:
                QMessageBox.critical(self, "Bad IDs", "Enter comma-separated integers, e.g. 0,1"); return
            save_all = False
        else:
            target_ids = set()
            save_all = True

        mode = self._mode_combo.currentText()
        self._progress.setValue(0)
        self._status_lbl.setText("Running...")
        self._run_btn.setEnabled(False)

        self._worker = BatchWorker(
            model=self._model,
            image_paths=self._image_paths,
            out_dir=self._out_folder,
            mode=mode,
            target_ids=target_ids,
            conf=self._conf,
            mask_color=self._mask_color,
            save_all_ids=save_all,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(lambda m: self._status_lbl.setText(m))
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, current, total):
        pct = int(current / total * 100)
        self._progress.setValue(pct)
        self._status_lbl.setText(f"Processing {current}/{total}...")

    def _on_done(self, saved, elapsed):
        self._progress.setValue(100)
        self._status_lbl.setText(f"Done! {saved} files saved in {elapsed:.1f}s.")
        self._run_btn.setEnabled(True)
        QMessageBox.information(self, "Batch Complete",
                                f"{saved} output files saved in {elapsed:.1f}s.")


if __name__ == "__main__":
    run_app(BatchYolo)
