import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QSlider, QFrame, QFileDialog, QMessageBox, QPushButton
)
from PySide6.QtGui import QColor, QPen, QImage, QPixmap
from PySide6.QtCore import Qt, QPointF

# Add the parent directory to sys.path so we can import vision_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.qt_base_gui import BaseVisionApp, run_app, C

class DrawableGraphicsView(BaseVisionApp):
    """Overrides the base view to capture mouse drawing events. But because BaseVisionApp holds the view, 
       we inject the drawing logic into the main app instead for simplicity."""
    pass

class ManualAnnotator(BaseVisionApp):
    def __init__(self):
        super().__init__(title="Manual Path Annotator", width=1280, height=720)
        
        # Drawing State
        self._is_drawing = False
        self._last_point = None
        self._mask_array = None # The underlying pure BW or RGB mask we are drawing
        self._mask_pixmap_item = None # The transparent Qt layer showing the mask visually
        
        # Build UI
        self._build_default_image_loader()
        self._build_drawing_tools()
        self._build_export_tools()
        
        # Connect base signal so we know when a new image is loaded via Browse/Paste
        self.image_loaded.connect(self._on_new_image)
        
        # Override View mouse events for drawing
        self.view.viewport().installEventFilter(self)
        self.view.mousePressEvent = self._custom_mouse_press
        self.view.mouseMoveEvent = self._custom_mouse_move
        self.view.mouseReleaseEvent = self._custom_mouse_release

    def _build_drawing_tools(self):
        self._add_section_header("Brush Settings")
        
        # Brush Size
        lbl_size = QLabel("Brush Size (px):")
        self.sidebar_layout.addWidget(lbl_size)
        
        self.spin_size = QSpinBox()
        self.spin_size.setRange(1, 100)
        self.spin_size.setValue(5)
        self.sidebar_layout.addWidget(self.spin_size)

        # Clear Button
        self._add_button("🗑️ Clear Canvas", self._clear_mask)

    def _build_export_tools(self):
        self._add_section_header("Export")
        self._add_button("💾 Save Red Line Target", lambda: self._save_mask("red"), primary=True)
        self._add_button("💾 Save B&W Boolean Mask", lambda: self._save_mask("bw"))

    def _on_new_image(self, rgb_array):
        """Called automatically when BaseVisionApp loads a new image via Browse or Paste"""
        h, w = rgb_array.shape[:2]
        
        # Create a blank transparent mask array (RGBA) matching the image size for the UI
        self._mask_array = np.zeros((h, w, 4), dtype=np.uint8)
        self._update_mask_visual()

    def _update_mask_visual(self):
        """Converts the internal numpy mask to a QPixmap and overlays it."""
        if self._mask_array is None: return
        h, w = self._mask_array.shape[:2]
        
        qimg = QImage(self._mask_array.data, w, h, w * 4, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        
        if self._mask_pixmap_item is None:
            self._mask_pixmap_item = self.scene.addPixmap(pix)
            self._mask_pixmap_item.setZValue(1) # Ensure it sits above the base image
        else:
            self._mask_pixmap_item.setPixmap(pix)

    def _draw_on_mask(self, pt1: QPointF, pt2: QPointF):
        if self._mask_array is None: return
        
        # Convert QPointF (Scene Coords) to integer pixel coords
        x1, y1 = int(pt1.x()), int(pt1.y())
        x2, y2 = int(pt2.x()), int(pt2.y())
        
        thickness = self.spin_size.value()
        
        # Draw on the numpy array. We draw Red (255, 0, 0) with full opacity (255)
        # Array format is RGBA, so: [R, G, B, A]
        color = (255, 0, 0, 255) 
        
        cv2.line(self._mask_array, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        self._update_mask_visual()

    def _clear_mask(self):
        if self._mask_array is not None:
            self._mask_array.fill(0)
            self._update_mask_visual()

    def _save_mask(self, mode):
        if self._mask_array is None:
            QMessageBox.warning(self, "Export", "No image loaded.")
            return
            
        initial_name = "manual_mask.png"
        if hasattr(self, 'path_entry') and self.path_entry.text() and self.path_entry.text() != "<Clipboard Image>":
            initial_name = f"mask_{os.path.basename(self.path_entry.text())}"
            
        p, _ = QFileDialog.getSaveFileName(self, "Save Mask", initial_name, "PNG Images (*.png)")
        if not p: return
        
        # Extract just the RGB or BW data from our RGBA drawing array
        if mode == "red":
            # Just take the RGB channels. It will be black background with a red line.
            out_img = self._mask_array[:, :, :3]
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR) # Convert to BGR for saving
            cv2.imwrite(p, out_img)
            
        elif mode == "bw":
            # Extract the Alpha channel or Red channel to make a thresholded B&W image
            r_channel = self._mask_array[:, :, 0]
            _, bw = cv2.threshold(r_channel, 1, 255, cv2.THRESH_BINARY)
            cv2.imwrite(p, bw)
            
        QMessageBox.information(self, "Saved", f"Mask saved to:\n{p}")

    # --- Mouse Event Overrides for Drawing ---
    def _custom_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.view.mapToScene(event.pos())
            
            # Check if click is actually inside the image bounds
            if self._mask_array is not None:
                h, w = self._mask_array.shape[:2]
                if 0 <= scene_pos.x() < w and 0 <= scene_pos.y() < h:
                    self._is_drawing = True
                    self._last_point = scene_pos
                    # Draw a single dot if they just click
                    self._draw_on_mask(scene_pos, scene_pos)
                    event.accept()
                    return
                    
        # Fallback to default (like Middle click panning)
        super(type(self.view), self.view).mousePressEvent(event)

    def _custom_mouse_move(self, event):
        if self._is_drawing and self._last_point:
            scene_pos = self.view.mapToScene(event.pos())
            self._draw_on_mask(self._last_point, scene_pos)
            self._last_point = scene_pos
            event.accept()
            return
            
        super(type(self.view), self.view).mouseMoveEvent(event)

    def _custom_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            self._last_point = None
            event.accept()
            return
            
        super(type(self.view), self.view).mouseReleaseEvent(event)


if __name__ == "__main__":
    run_app(ManualAnnotator)
