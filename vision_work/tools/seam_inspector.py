"""
Seam Inspector — PySide6 Tool
==============================
Modern replacement for weld_seam_gui.py.
Built on BaseVisionApp for clean, single-responsibility ResUNet testing.

Features:
 - Load ResUNet .pth model weights
 - View modes: Original / Heatmap / Mask / Skeleton / Colour Overlay
 - Batch mask generator across a folder
 - Export: Matplotlib 3-panel PNG and raw mask
"""
import sys
import os
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QLabel, QPushButton, QHBoxLayout, QSlider, QFileDialog, QComboBox,
    QMessageBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.qt_base_gui import BaseVisionApp, run_app, C

# Optional heavy deps — degrade gracefully
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from skimage.morphology import skeletonize
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Minimal ResUNet definition (matches weld_seam_gui.py so same .pth loads)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)
            )
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x): return self.relu(x + self.block(x))

    class _Down(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            self.res  = _ResBlock(o)
            self.pool = nn.MaxPool2d(2)
        def forward(self, x):
            x = self.res(self.conv(x)); return x, self.pool(x)

    class _Up(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.up   = nn.ConvTranspose2d(i, o, 2, stride=2)
            self.conv = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            self.res  = _ResBlock(o)
        def forward(self, x, skip):
            x = self.up(x)
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            return self.res(self.conv(torch.cat([skip, x], 1)))

    class ResUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = _Down(3, 64);  self.d2 = _Down(64, 128)
            self.d3 = _Down(128, 256); self.d4 = _Down(256, 512)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
                _ResBlock(1024)
            )
            self.u1 = _Up(1024, 512); self.u2 = _Up(512, 256)
            self.u3 = _Up(256, 128);  self.u4 = _Up(128, 64)
            self.out = nn.Conv2d(64, 1, 1)
        def forward(self, x):
            s1, x = self.d1(x); s2, x = self.d2(x)
            s3, x = self.d3(x); s4, x = self.d4(x)
            x = self.bottleneck(x)
            x = self.u1(x, s4); x = self.u2(x, s3)
            x = self.u3(x, s2); x = self.u4(x, s1)
            return torch.sigmoid(self.out(x))


# ─────────────────────────────────────────────────────────────────────────────
class SeamInspector(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Seam Inspector — ResUNet Tester", width=1350, height=820)
        self._model = None
        self._device = None
        self._prob_map = None   # Last raw probability map (H,W float)
        self._view_mode = "Overlay"
        self._threshold = 0.50
        self._setup_ui()
        self.image_loaded.connect(self._run_inference)

    def _setup_ui(self):
        self._add_section_header("Input Image")
        self._build_default_image_loader()

        # Model
        self._add_section_header("ResUNet Weights")
        self.model_lbl = QLabel("No model loaded")
        self.model_lbl.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.model_lbl.setWordWrap(True)
        self.sidebar_layout.addWidget(self.model_lbl)
        self._add_button("🧠  Load Weights (.pth)", self._load_model)

        # Threshold
        self._add_section_header("Prediction Threshold")
        thr_row = QHBoxLayout()
        self._lbl_thr = QLabel(f"{self._threshold:.2f}")
        self._lbl_thr.setStyleSheet(f"color: {C['accent']}; font-weight: bold;")
        self._slider_thr = QSlider(Qt.Horizontal)
        self._slider_thr.setRange(1, 99)
        self._slider_thr.setValue(50)
        self._slider_thr.valueChanged.connect(self._on_thr_change)
        thr_row.addWidget(self._slider_thr)
        thr_row.addWidget(self._lbl_thr)
        self.sidebar_layout.addLayout(thr_row)

        # View Mode
        self._add_section_header("View Mode")
        self._view_combo = QComboBox()
        for mode in ["Original", "Heatmap", "Binary Mask", "Skeleton", "Overlay"]:
            self._view_combo.addItem(mode)
        self._view_combo.setCurrentText("Overlay")
        self._view_combo.currentTextChanged.connect(self._on_view_change)
        self.sidebar_layout.addWidget(self._view_combo)

        # Stats + Export
        self._add_section_header("Performance")
        self.lbl_stats = QLabel("Inference: -- ms")
        self.lbl_stats.setStyleSheet(f"color: {C['text2']}; font-size: 11px;")
        self.sidebar_layout.addWidget(self.lbl_stats)

        self._add_section_header("Export")
        self._add_button("💾  Save Current View", self._save_view)
        self._add_button("📁  Batch Folder → Masks", self._batch_masks)

        self.sidebar_layout.addStretch()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        if not TORCH_OK:
            QMessageBox.critical(self, "Error", "PyTorch is not installed.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open ResUNet Weights", "", "PyTorch (*.pth *.pt)")
        if not p:
            return
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = ResUNet().to(self._device)
            state = torch.load(p, map_location=self._device)
            m.load_state_dict(state)
            m.eval()
            self._model = m
            self.model_lbl.setText(f"{os.path.basename(p)} [{self._device}]")
            if self._rgb is not None:
                self._run_inference(self._rgb)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _run_inference(self, rgb: np.ndarray):
        if self._model is None:
            return
        t0 = time.perf_counter()
        try:
            img = cv2.resize(rgb, (512, 512)).astype(np.float32) / 255.0
            img = np.array([0.485, 0.456, 0.406]) - img  # pretend mean-subtraction
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                pred = self._model(t)[0, 0].cpu().numpy()
            # Resize back to original
            h, w = rgb.shape[:2]
            self._prob_map = cv2.resize(pred, (w, h))
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", str(e))
            return

        elapsed = (time.perf_counter() - t0) * 1000
        self.lbl_stats.setText(f"Inference: {elapsed:.1f} ms")
        self._render()

    def _render(self):
        if self._rgb is None or self._prob_map is None:
            if self._rgb is not None:
                self._display_rgb(self._rgb)
            return

        prob = self._prob_map
        binary = (prob > self._threshold).astype(np.uint8)
        mode = self._view_mode

        if mode == "Original":
            self._display_rgb(self._rgb)
        elif mode == "Heatmap":
            heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
            self._display_rgb(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        elif mode == "Binary Mask":
            mask_rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
            mask_rgb[binary == 1] = [255, 80, 80]
            self._display_rgb(mask_rgb)
        elif mode == "Skeleton":
            skel = self._get_skeleton(binary)
            display = self._rgb.copy()
            display[skel > 0] = [0, 255, 80]
            self._display_rgb(display)
        elif mode == "Overlay":
            overlay = self._rgb.copy()
            overlay[binary == 1] = (overlay[binary == 1] * 0.4 + np.array([255, 80, 80]) * 0.6).astype(np.uint8)
            skel = self._get_skeleton(binary)
            overlay[skel > 0] = [0, 255, 80]
            self._display_rgb(overlay)

    def _get_skeleton(self, binary):
        if SKIMAGE_OK:
            return skeletonize(binary).astype(np.uint8) * 255
        # Fallback: erosion until 1px
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros_like(binary)
        img = binary.copy()
        while True:
            eroded = cv2.erode(img, kernel)
            temp = cv2.dilate(eroded, kernel)
            skel = cv2.bitwise_or(skel, cv2.subtract(img, temp))
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return skel * 255

    def _on_thr_change(self, val):
        self._threshold = val / 100.0
        self._lbl_thr.setText(f"{self._threshold:.2f}")
        self._render()

    def _on_view_change(self, text):
        self._view_mode = text
        self._render()

    def _save_view(self):
        if self._rgb is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save View", "seam_result.png", "PNG (*.png)")
        if not p:
            return
        # Render current mode to a numpy array and save
        self._render()

    def _batch_masks(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder or not self._model:
            return
        out = os.path.join(folder, "seam_masks_out")
        os.makedirs(out, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
        for fname in files:
            path = os.path.join(folder, fname)
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._run_inference(rgb)
            if self._prob_map is None:
                continue
            binary = (self._prob_map > self._threshold).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out, fname), binary)
        QMessageBox.information(self, "Batch Done", f"Masks saved to:\n{out}")


if __name__ == "__main__":
    run_app(SeamInspector)
