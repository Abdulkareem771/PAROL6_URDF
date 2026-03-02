"""
serial_monitor.py — Background thread that reads/writes a serial port.
Emits Qt signals for received lines and parsed telemetry packets.
"""
from __future__ import annotations
import re
import time
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Regex to parse <ACK,seq,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6[,isr_us]>
_ACK_RE = re.compile(
    r"<ACK,(\d+),([\d\.,\-]+)>",
    re.ASCII,
)


def list_serial_ports() -> list[str]:
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


class SerialWorker(QThread):
    """Reads lines from serial port; emits raw lines and parsed telemetry."""

    raw_line    = pyqtSignal(str)            # every line received
    telemetry   = pyqtSignal(dict)           # parsed <ACK,...> packet as dict
    error_msg   = pyqtSignal(str)            # connection / IO errors
    connected   = pyqtSignal(bool)           # True=connected False=disconnected
    packet_rate = pyqtSignal(float)          # packets per second (approx)

    def __init__(self, port: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self._port   = port
        self._baud   = baud
        self._running = False
        self._ser: "serial.Serial | None" = None  # type: ignore[name-defined]

    def run(self) -> None:
        if not SERIAL_AVAILABLE:
            self.error_msg.emit("pyserial not installed — run: pip install pyserial")
            return

        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0.5)
            self.connected.emit(True)
        except Exception as e:
            self.error_msg.emit(f"Cannot open {self._port}: {e}")
            return

        self._running = True
        pkt_count = 0
        rate_t0   = time.monotonic()

        while self._running:
            try:
                line = self._ser.readline().decode("utf-8", errors="replace").strip()
            except Exception as e:
                self.error_msg.emit(f"Serial read error: {e}")
                break

            if not line:
                continue

            self.raw_line.emit(line)
            pkt_count += 1

            # Parse telemetry
            m = _ACK_RE.match(line)
            if m:
                seq_str, data_str = m.group(1), m.group(2)
                nums = [float(x) for x in data_str.split(",") if x]
                if len(nums) >= 12:
                    pkt = {
                        "seq": int(seq_str),
                        "pos": nums[0:6],
                        "vel": nums[6:12],
                        "isr_us": nums[12] if len(nums) > 12 else None,
                    }
                    self.telemetry.emit(pkt)

            # Approximate packet rate every second
            now = time.monotonic()
            if now - rate_t0 >= 1.0:
                self.packet_rate.emit(pkt_count / (now - rate_t0))
                pkt_count = 0
                rate_t0 = now

        if self._ser and self._ser.is_open:
            self._ser.close()
        self.connected.emit(False)

    def send(self, text: str) -> None:
        if self._ser and self._ser.is_open:
            self._ser.write((text + "\n").encode())

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
