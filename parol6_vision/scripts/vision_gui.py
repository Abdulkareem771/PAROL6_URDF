#!/usr/bin/env python3
"""
vision_gui.py — PAROL6 Vision Execution Control Panel
A simple PyQt6 interface to launch and monitor the vision pipeline and
trajectory executor, following the dark style of parol6_firmware_configurator.
"""
import sys
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QPlainTextEdit, QMessageBox, QGroupBox
)
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QFont, QTextCursor

DARK_STYLESHEET = """
QMainWindow, QWidget, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}
QPushButton {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    color: #cdd6f4;
    font-weight: bold;
}
QPushButton:hover   { background: #45475a; border-color: #cba6f7; }
QPushButton:pressed { background: #585b70; }
QPushButton:disabled{ color: #585b70; border-color: #313244; }
QPushButton#stopBtn { background: #f38ba8; color: #11111b; border: none; }
QPushButton#stopBtn:hover { background: #eba0ac; border: 1px solid #f38ba8; }
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    color: #a6adc8;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QPlainTextEdit {
    background: #11111b;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #a6e3a1;
    font-family: monospace;
    padding: 6px;
}
QLabel { color: #cdd6f4; }
"""

class VisionExecutionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAROL6 Vision Control Panel")
        self.resize(800, 600)
        self.process = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Header
        header = QLabel("Vision Trajectory Execution (Anti-Gravity Control)")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #cba6f7;")
        layout.addWidget(header)

        # Controls Group
        controls_group = QGroupBox("Launch Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.btn_live = QPushButton("Launch Live Vision")
        self.btn_live.clicked.connect(lambda: self.launch_system(use_bag=False))
        
        self.btn_bag = QPushButton("Launch Bag Replay")
        self.btn_bag.clicked.connect(lambda: self.launch_system(use_bag=True))
        
        self.btn_stop = QPushButton("Stop All")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_system)
        self.btn_stop.setEnabled(False)

        controls_layout.addWidget(self.btn_live)
        controls_layout.addWidget(self.btn_bag)
        controls_layout.addWidget(self.btn_stop)
        
        layout.addWidget(controls_group)

        # Terminal Output
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        layout.addWidget(QLabel("ROS2 System Log:"))
        layout.addWidget(self.terminal)

    def launch_system(self, use_bag=True):
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            QMessageBox.warning(self, "Warning", "A launch process is already running!")
            return

        cmd = 'ros2'
        args = ['launch', 'parol6_vision', 'vision_moveit.launch.py']
        if not use_bag:
            args.append('use_bag:=false')
            
        self.terminal.clear()
        self.terminal.appendPlainText(f">>> Running: {cmd} {' '.join(args)}\n")

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.handle_finished)
        
        self.process.start(cmd, args)
        
        self.btn_live.setEnabled(False)
        self.btn_bag.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf8', errors='ignore')
        self.terminal.moveCursor(QTextCursor.MoveOperation.End)
        self.terminal.insertPlainText(stdout)
        self.terminal.moveCursor(QTextCursor.MoveOperation.End)

    def stop_system(self):
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            self.terminal.appendPlainText("\n>>> Sending SIGINT to ROS2 Launch Process...")
            # For ROS2 launch, terminate() usually sends SIGINT to stop nodes cleanly
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()

    def handle_finished(self):
        self.terminal.appendPlainText("\n>>> Process Finished.")
        self.process = None
        self.btn_live.setEnabled(True)
        self.btn_bag.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        self.stop_system()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    gui = VisionExecutionGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
