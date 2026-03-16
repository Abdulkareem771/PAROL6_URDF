from PyQt6.QtWidgets import QPushButton, QGraphicsDropShadowEffect
from PyQt6.QtCore import QVariantAnimation, pyqtProperty, Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QColor

DARK_STYLESHEET = """
QMainWindow, QWidget, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 0 8px 8px 8px;
    background: #1e1e2e;
}
QTabBar::tab {
    background: #181825;
    color: #a6adc8;
    padding: 8px 16px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:selected { 
    background: #313244; 
    color: #cba6f7; 
    font-weight: bold; 
    border-top: 2px solid #cba6f7;
}
QTabBar::tab:hover { 
    background: #313244; 
    color: #cdd6f4; 
}
QPushButton {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #cdd6f4;
    font-weight: 600;
}
QPushButton:hover   { background: #45475a; border-color: #cba6f7; }
QPushButton:pressed { background: #585b70; border-color: #cba6f7; }
QPushButton:disabled{ color: #585b70; border-color: #313244; }
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 16px;
    margin-bottom: 6px;
    padding-top: 14px;
    font-weight: bold;
    color: #b4befe;
}
QGroupBox::title { 
    subcontrol-origin: margin; 
    subcontrol-position: top left;
    padding: 0 4px;
    left: 8px; 
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {
    background: #11111b;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
    padding: 4px 6px;
    selection-background-color: #f5c2e7;
    selection-color: #11111b;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #89b4fa;
}
QComboBox QAbstractItemView { background: #313244; border: 1px solid #45475a; }
QStatusBar { background: #181825; color: #a6adc8; border-top: 1px solid #45475a; }
QToolBar { background: #181825; border-bottom: 1px solid #45475a; spacing: 8px; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #45475a;
    border-radius: 4px;
    background: #11111b;
}
QCheckBox::indicator:checked {
    background: #cba6f7;
    border: 1px solid #cba6f7;
}
"""


