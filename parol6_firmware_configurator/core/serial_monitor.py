"""
serial_monitor.py — Background thread that reads/writes a serial port.
Emits Qt signals for received lines and parsed telemetry packets.
"""
from __future__ import annotations
import re
import time
import socket
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Regex to parse <ACK,seq,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6[,lim_state]>
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
    data_rate   = pyqtSignal(float)          # bytes per second (approx)

    def __init__(self, port: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self._port   = port
        self._baud   = baud
        self._running = False
        self._ser: "serial.Serial | None" = None  # type: ignore[name-defined]

    def run(self) -> None:
        self._running = True
        pkt_count = 0
        byte_count = 0
        rate_t0   = time.monotonic()

        self._is_udp = self._port.startswith("udp://")
        if self._is_udp:
            try:
                parts = self._port.replace("udp://", "").split(":")
                self._udp_ip = parts[0]
                self._udp_port = int(parts[1])
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.bind(("", self._udp_port))
                self._sock.settimeout(0.5)
                self.connected.emit(True)
            except Exception as e:
                self.error_msg.emit(f"Cannot bind UDP port {self._udp_port}: {e}")
                return
        else:
            if not SERIAL_AVAILABLE:
                self.error_msg.emit("pyserial not installed — run: pip install pyserial")
                return
            try:
                self._ser = serial.Serial(self._port, self._baud, timeout=0.5)
                self.connected.emit(True)
            except Exception as e:
                self.error_msg.emit(f"Cannot open {self._port}: {e}")
                return

        while self._running:
            try:
                if self._is_udp:
                    try:
                        data, addr = self._sock.recvfrom(1024)
                        byte_count += len(data)
                        line = data.decode("utf-8", errors="replace").strip()
                        # Update target IP to whoever sent us data (for robustness)
                        self._udp_ip = addr[0]
                    except socket.timeout:
                        line = ""
                else:
                    raw_bytes = self._ser.readline()
                    byte_count += len(raw_bytes)
                    line = raw_bytes.decode("utf-8", errors="replace").strip()
            except Exception as e:
                self.error_msg.emit(f"Read error: {e}")
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
                    # Backward-compatible packet formats:
                    #   12 fields: pos[0..5] + vel[0..5]
                    #   13 fields: pos + vel + lim_state          ← current firmware format
                    #   18 fields: pos + vel + pwm[0..5]
                    #   19 fields: pos + vel + pwm[0..5] + lim_state
                    #   20 fields: pos + vel + pwm + lim_state + isr_us
                    has_pwm       = len(nums) >= 18
                    lim_idx       = 18 if has_pwm else 12
                    has_lim_state = len(nums) > lim_idx           # any extra field = lim_state
                    isr_idx       = lim_idx + 1                   # isr after lim_state if present
                    pkt = {
                        "seq":      int(seq_str),
                        "pos":      nums[0:6],
                        "vel":      nums[6:12],
                        "pwm":      nums[12:18] if has_pwm else [0.0] * 6,
                        "lim_state": int(nums[lim_idx]) if has_lim_state else None,
                        "isr_us":   nums[isr_idx] if len(nums) > isr_idx else None,
                    }
                    self.telemetry.emit(pkt)

            # Approximate packet rate every second
            now = time.monotonic()
            if now - rate_t0 >= 1.0:
                dt = now - rate_t0
                self.packet_rate.emit(pkt_count / dt)
                self.data_rate.emit(byte_count / dt)
                pkt_count = 0
                byte_count = 0
                rate_t0 = now

        if getattr(self, "_is_udp", False):
            if getattr(self, "_sock", None):
                self._sock.close()
        else:
            if self._ser and self._ser.is_open:
                self._ser.close()
        self.connected.emit(False)

    def send(self, text: str) -> None:
        payload = (text + "\n").encode()
        if getattr(self, "_is_udp", False) and getattr(self, "_sock", None):
            try:
                self._sock.sendto(payload, (self._udp_ip, self._udp_port))
            except Exception as e:
                self.error_msg.emit(f"UDP send error: {e}")
        elif self._ser and self._ser.is_open:
            self._ser.write(payload)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
