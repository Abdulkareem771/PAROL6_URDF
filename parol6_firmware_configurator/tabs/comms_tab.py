"""
comms_tab.py — Transport mode, ROS rate, control loop rate, Ethernet settings.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSlider, QSpinBox, QLineEdit, QFormLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
from core.config_model import CommsConfig


class CommsTab(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("📡  Communications & Timing")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        root.addWidget(title)

        # ── Transport ─────────────────────────────────────────────────
        trans_box = QGroupBox("Transport Layer")
        trans_lay = QVBoxLayout(trans_box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Protocol:"))
        self.transport = QComboBox()
        self.transport.addItems(["USB_CDC_HS", "UART_115200", "ETHERNET"])
        self.transport.setToolTip(
            "USB_CDC_HS: Teensy native USB 2.0 HS (480 Mbps). Best for ≥50 Hz ROS.\n"
            "UART_115200: Standard serial, compatible with all platforms.\n"
            "ETHERNET: Teensy 4.1 built-in 1 Gbps. Requires QNEthernet library."
        )
        self.transport.currentTextChanged.connect(self._on_transport_changed)
        self.transport.currentTextChanged.connect(self.changed)
        row1.addWidget(self.transport)
        row1.addStretch()
        trans_lay.addLayout(row1)

        # Ethernet sub-group (shown only when ETHERNET selected)
        self.eth_box = QGroupBox("Ethernet Settings")
        eth_form = QFormLayout(self.eth_box)
        self.eth_ip   = QLineEdit("192.168.1.177")
        self.eth_gw   = QLineEdit("192.168.1.1")
        self.eth_sub  = QLineEdit("255.255.255.0")
        self.eth_port = QSpinBox()
        self.eth_port.setRange(1, 65535)
        self.eth_port.setValue(8888)
        for w in (self.eth_ip, self.eth_gw, self.eth_sub):
            w.textChanged.connect(self.changed)
        self.eth_port.valueChanged.connect(self.changed)
        eth_form.addRow("Device IP:", self.eth_ip)
        eth_form.addRow("Gateway:",   self.eth_gw)
        eth_form.addRow("Subnet:",    self.eth_sub)
        eth_form.addRow("Port:",      self.eth_port)
        trans_lay.addWidget(self.eth_box)
        self.eth_box.setVisible(False)

        root.addWidget(trans_box)

        # ── Timing ────────────────────────────────────────────────────
        timing_box = QGroupBox("Timing")
        timing_lay = QFormLayout(timing_box)

        self.ros_rate = self._make_rate_row(1, 200, 25, "Hz — how often ROS sends waypoints")
        self.fb_rate  = self._make_rate_row(1, 100, 10, "Hz — how often Teensy sends <ACK,...>")

        self.ctrl_rate = QComboBox()
        self.ctrl_rate.addItems(["1000", "500"])
        self.ctrl_rate.setToolTip("ISR IntervalTimer period. 1000 Hz = 1 ms period.")
        self.ctrl_rate.currentTextChanged.connect(self.changed)

        self.timeout_ms = QSpinBox()
        self.timeout_ms.setRange(50, 5000)
        self.timeout_ms.setValue(200)
        self.timeout_ms.setSuffix(" ms")
        self.timeout_ms.setToolTip("Watchdog timeout. 200 ms = 5 missed 25 Hz packets.")
        self.timeout_ms.valueChanged.connect(self.changed)

        timing_lay.addRow("ROS Command Rate:", self.ros_rate)
        timing_lay.addRow("Feedback Rate:",    self.fb_rate)
        timing_lay.addRow("Control Loop Rate:", self.ctrl_rate)
        timing_lay.addRow("Command Timeout:",   self.timeout_ms)

        root.addWidget(timing_box)

        # ── Serial Port (GUI only, not flashed) ───────────────────────
        port_box = QGroupBox("Serial Port (GUI monitoring — not flashed)")
        port_lay = QHBoxLayout(port_box)
        port_lay.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(180)
        port_lay.addWidget(self.port_combo)

        port_lay.addWidget(QLabel("Baud:"))
        self.baud = QComboBox()
        self.baud.addItems(["115200", "921600", "9600"])
        port_lay.addWidget(self.baud)

        from PyQt6.QtWidgets import QPushButton
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        port_lay.addWidget(refresh_btn)
        port_lay.addStretch()
        root.addWidget(port_box)

        root.addStretch()
        self._refresh_ports()

    def _make_rate_row(self, lo, hi, default, tooltip) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(default)
        w.setSuffix(" Hz")
        w.setToolTip(tooltip)
        w.valueChanged.connect(self.changed)
        return w

    def _on_transport_changed(self, t: str) -> None:
        self.eth_box.setVisible(t == "ETHERNET")

    def _refresh_ports(self) -> None:
        from core.serial_monitor import list_serial_ports
        self.port_combo.clear()
        self.port_combo.addItem("")
        for p in list_serial_ports():
            self.port_combo.addItem(p)

    def load(self, c: CommsConfig) -> None:
        self.transport.setCurrentText(c.transport)
        self.ros_rate.setValue(c.ros_command_rate_hz)
        self.fb_rate.setValue(c.feedback_rate_hz)
        self.ctrl_rate.setCurrentText(str(c.control_loop_rate_hz))
        self.timeout_ms.setValue(c.command_timeout_ms)
        self.eth_ip.setText(c.ethernet_ip)
        self.eth_gw.setText(c.ethernet_gateway)
        self.eth_sub.setText(c.ethernet_subnet)
        self.eth_port.setValue(c.ethernet_port)
        if c.serial_port:
            self.port_combo.setCurrentText(c.serial_port)
        self.baud.setCurrentText(str(c.baud_rate))

    def save(self, c: CommsConfig) -> None:
        c.transport              = self.transport.currentText()
        c.ros_command_rate_hz    = self.ros_rate.value()
        c.feedback_rate_hz       = self.fb_rate.value()
        c.control_loop_rate_hz   = int(self.ctrl_rate.currentText())
        c.command_timeout_ms     = self.timeout_ms.value()
        c.ethernet_ip            = self.eth_ip.text()
        c.ethernet_gateway       = self.eth_gw.text()
        c.ethernet_subnet        = self.eth_sub.text()
        c.ethernet_port          = self.eth_port.value()
        c.serial_port            = self.port_combo.currentText()
        c.baud_rate              = int(self.baud.currentText())

    @property
    def serial_port(self) -> str:
        return self.port_combo.currentText()

    @property
    def baud_rate(self) -> int:
        return int(self.baud.currentText())
