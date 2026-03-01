import os
import cv2
import numpy as np
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QMessageBox,
    QLineEdit
)
from PySide6.QtGui import (
    QImage, QPixmap, QColor, QFont, QKeySequence, QShortcut, QIcon, QPainter
)
from PySide6.QtCore import Qt, QRectF, Signal, Slot

# Global Dark Theme
C = {
    "bg": "#1e1e2e",
    "panel": "#181825",
    "border": "#313244",
    "text": "#cdd6f4",
    "text2": "#a6adc8",
    "accent": "#89b4fa",
    "green": "#a6e3a1",
    "warn": "#f9e2af",
    "err": "#f38ba8"
}

STYLE_SHEET = f"""
QMainWindow, QDialog, QMessageBox {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Segoe UI', Arial, sans-serif;
}}
QWidget#SidebarWidget {{
    background-color: {C['panel']};
}}
QScrollArea {{
    border: none;
    background-color: {C['panel']};
}}
QSplitter::handle {{
    background-color: {C['border']};
}}
QPushButton {{
    background-color: {C['border']};
    color: {C['text']};
    padding: 6px 12px;
    border: none;
    border-radius: 4px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {C['accent']};
    color: {C['bg']};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {C['border']};
    color: {C['text']};
    border: 1px solid {C['bg']};
    padding: 6px;
    border-radius: 4px;
}}
QLabel {{
    background-color: transparent;
    color: {C['text']};
}}
QLabel[class="header"] {{
    color: {C['text2']};
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
}}
"""

class ZoomPanGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(C['bg']))
        
        self._is_panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom_factor = 1.15
        else:
            zoom_factor = 1.0 / 1.15
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_x = event.scenePosition().x()
            self._pan_start_y = event.scenePosition().y()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (event.x() - self._pan_start_x))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (event.y() - self._pan_start_y))
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            event.accept()
            return
        super().mouseMoveEvent(event)

class BaseVisionApp(QMainWindow):
    """
    A unified PySide6 base class for all PAROL6 Computer Vision R&D tools.
    Provides styling, layout, image loading/pasting, and a high-performance
    Zoom/Pan QGraphicsScene out of the box.
    """
    image_loaded = Signal(np.ndarray) # Emitted when a new raw image is loaded (RGB format)

    def __init__(self, title="WeldVision Qt Toolkit", width=1280, height=720):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(width, height)
        
        self._rgb = None # The underlying raw image array
        self._pixmap_item = None # The current QGraphicsPixmapItem
        
        # Central Splitter Layout
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # Sidebar (Left)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setMinimumWidth(320)
        self.sidebar_scroll.setMaximumWidth(400)
        
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setObjectName("SidebarWidget")
        self.sidebar_scroll.setWidget(self.sidebar_widget)
        
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_layout.setContentsMargins(15, 15, 15, 15)
        self.sidebar_layout.setSpacing(10)
        
        self.splitter.addWidget(self.sidebar_scroll)
        
        # Canvas (Right)
        self.scene = QGraphicsScene()
        self.view = ZoomPanGraphicsView(self.scene)
        self.splitter.addWidget(self.view)
        
        self.splitter.setSizes([350, 930])
        
        # Shortcuts
        self.shortcut_paste = QShortcut(QKeySequence("Ctrl+V"), self)
        self.shortcut_paste.activated.connect(self._handle_paste)

    def _add_section_header(self, text):
        lbl = QLabel(text)
        lbl.setProperty("class", "header")
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"background-color: {C['border']};")
        
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(lbl)
        self.sidebar_layout.addWidget(line)
        self.sidebar_layout.addSpacing(5)

    def _add_button(self, text, callback, primary=False):
        btn = QPushButton(text)
        if primary:
            btn.setStyleSheet(f"background-color: {C['accent']}; color: {C['bg']}; padding: 10px; font-size: 13px;")
        btn.clicked.connect(callback)
        self.sidebar_layout.addWidget(btn)
        return btn

    def _build_default_image_loader(self):
        """Builds a standard 'Test Image' section in the sidebar."""
        self._add_section_header("Test Image")
        
        self.path_entry = QLineEdit()
        self.path_entry.setReadOnly(True)
        self.path_entry.setPlaceholderText("No image loaded...")
        self.sidebar_layout.addWidget(self.path_entry)
        
        layout = QHBoxLayout()
        btn_browse = QPushButton("📁 Browse")
        btn_browse.clicked.connect(self._handle_browse)
        btn_paste = QPushButton("📋 Paste")
        btn_paste.clicked.connect(self._handle_paste)
        
        layout.addWidget(btn_browse)
        layout.addWidget(btn_paste)
        self.sidebar_layout.addLayout(layout)

    def _handle_browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if p:
            self._load_from_path(p)

    def _handle_paste(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        if mime_data.hasImage():
            qimg = clipboard.image()
            self._load_from_qimage(qimg)
            if hasattr(self, 'path_entry'):
                self.path_entry.setText("<Clipboard Image>")
        else:
            QMessageBox.warning(self, "Paste Error", "No valid image found in clipboard.")

    def _load_from_path(self, path):
        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{path}")
            return
        if hasattr(self, 'path_entry'):
            self.path_entry.setText(path)
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._set_rgb_image(rgb)

    def _load_from_qimage(self, qimg: QImage):
        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        width, height = qimg.width(), qimg.height()
        ptr = qimg.constBits()
        # Create a view into the memory, then copy to own it
        arr = np.array(memoryview(ptr)).reshape(height, width, 3).copy()
        self._set_rgb_image(arr)

    def _set_rgb_image(self, rgb_array: np.ndarray):
        """Internal setter that triggers the base logic and emits signal."""
        self._rgb = rgb_array
        self._display_rgb(self._rgb)
        
        # Fit view to scene initially
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
        # Emit signal to subclasses
        self.image_loaded.emit(self._rgb)

    def _display_rgb(self, rgb_array: np.ndarray):
        """Converts an RGB numpy array to a QPixmap and renders it to the QGraphicsScene."""
        h, w, ch = rgb_array.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        if self._pixmap_item is None:
            self._pixmap_item = self.scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
            
        self.scene.setSceneRect(QRectF(pixmap.rect()))

    def get_current_rgb(self):
        """Returns the base raw image."""
        return self._rgb

# Entry point wrapper helper
def run_app(AppClass):
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = AppClass()
    window.show()
    sys.exit(app.exec())
