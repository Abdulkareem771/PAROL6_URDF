"""
serial_monitor.py — Background thread that reads/writes a serial port.
Emits Qt signals for received lines and parsed telemetry packets.
"""
from __future__ import annotations

import glob
import os
import re
import socket
import time

from PyQt6.QtCore import QThread, pyqtSignal

try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


_ACK_RE = re.compile(r"<ACK,(\d+),([\d\.,\-]+)>", re.ASCII)


def list_serial_ports() -> list[str]:
    ports: list[str] = []
    if SERIAL_AVAILABLE:
        ports.extend(p.device for p in serial.tools.list_ports.comports())

    # Docker and udev setups sometimes hide devices from pyserial, so fall back
    # to common Linux serial paths and stable by-id symlinks.
    patterns = [
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/ttyAMA*",
        "/dev/ttyS*",
        "/dev/serial/by-id/*",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.exists(path):
                ports.append(path)

    seen: set[str] = set()
    ordered: list[str] = []
    for port in ports:
        if port not in seen:
            seen.add(port)
            ordered.append(port)
    return ordered


class SerialWorker(QThread):
    raw_line = pyqtSignal(str)
    telemetry = pyqtSignal(dict)
    error_msg = pyqtSignal(str)
    connected = pyqtSignal(bool)
    packet_rate = pyqtSignal(float)
    data_rate = pyqtSignal(float)

    def __init__(self, port: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self._port = port
        self._baud = baud
        self._running = False
        self._ser: "serial.Serial | None" = None  # type: ignore[name-defined]

    def run(self) -> None:
        self._running = True
        pkt_count = 0
        byte_count = 0
        rate_t0 = time.monotonic()

        self._is_udp = self._port.startswith("udp://")
        if self._is_udp:
            try:
                host, port = self._port.replace("udp://", "").split(":")
                self._udp_ip = host
                self._udp_port = int(port)
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.bind(("", self._udp_port))
                self._sock.settimeout(0.5)
                self.connected.emit(True)
            except Exception as exc:
                self.error_msg.emit(f"Cannot bind UDP port {self._port}: {exc}")
                return
        else:
            if not SERIAL_AVAILABLE:
                self.error_msg.emit("pyserial not installed — run: pip install pyserial")
                return
            try:
                self._ser = serial.Serial(self._port, self._baud, timeout=0.5)
                self.connected.emit(True)
            except Exception as exc:
                self.error_msg.emit(f"Cannot open {self._port}: {exc}")
                return

        while self._running:
            try:
                if self._is_udp:
                    try:
                        data, addr = self._sock.recvfrom(1024)
                        byte_count += len(data)
                        line = data.decode("utf-8", errors="replace").strip()
                        self._udp_ip = addr[0]
                    except socket.timeout:
                        line = ""
                else:
                    raw_bytes = self._ser.readline()
                    byte_count += len(raw_bytes)
                    line = raw_bytes.decode("utf-8", errors="replace").strip()
            except Exception as exc:
                self.error_msg.emit(f"Read error: {exc}")
                break

            if not line:
                continue

            self.raw_line.emit(line)
            pkt_count += 1
            self._emit_telemetry(line)

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
        elif self._ser and self._ser.is_open:
            self._ser.close()

        self.connected.emit(False)

    def _emit_telemetry(self, line: str) -> None:
        match = _ACK_RE.match(line)
        if not match:
            return

        seq_str, data_str = match.group(1), match.group(2)
        nums = [float(x) for x in data_str.split(",") if x]
        n = len(nums)

        if n < 12:
            return

        # Detect ACK format:
        #
        # FORMAT A — parol6_firmware (flat):
        #   <ACK,seq, p0,p1,p2,p3,p4,p5, v0,v1,v2,v3,v4,v5, [pwm0..5,] [lim,] [state,] [isr]>
        #   n >= 12, positions at [0:6], velocities at [6:12]
        #
        # FORMAT B — realtime_servo_blackpill (interleaved):
        #   <ACK,seq, p0,v0, p1,v1, p2,v2, p3,v3, p4,v4, p5,v5>
        #   Always exactly 12 values, but order is p,v,p,v,...
        #
        # Heuristic: if all 12 values arrive and the even[6:12] values look like velocities
        # (all smaller than the odd[0:6] position spread), both formats look the same with 12 values.
        # We distinguish by checking if this is the blackpill project via a registry hint —
        # instead we do it structurally: FORMAT B never has lim_state/state_byte trailing fields,
        # and its velocity commands (velocity_command in the JointState) are typically small.
        # Since FORMAT B always has exactly 12 values, and FORMAT A typically has 14 (12+lim+state),
        # use n == 12 as the interleaved indicator when no extra fields follow.
        #
        # In practice: n == 12 → try interleaved first (blackpill) OR flat with no state fields.
        # n >= 13 → flat (parol6_firmware style).

        if n == 12:
            # Ambiguous — could be flat (no trailing fields) or interleaved.
            # Use the interleaved parse which is the more common STM32 case.
            pos = [nums[i * 2]     for i in range(6)]
            vel = [nums[i * 2 + 1] for i in range(6)]
            pkt = {
                "seq":        int(seq_str),
                "pos":        pos,
                "vel":        vel,
                "pwm":        [0.0] * 6,
                "lim_state":  None,
                "state_byte": None,
            }
        else:
            # FORMAT A — flat layout (parol6_firmware)
            has_pwm     = n >= 18
            lim_idx     = 18 if has_pwm else 12
            pkt = {
                "seq":        int(seq_str),
                "pos":        nums[0:6],
                "vel":        nums[6:12],
                "pwm":        nums[12:18] if has_pwm else [0.0] * 6,
                "lim_state":  int(nums[lim_idx])     if n > lim_idx     else None,
                "state_byte": int(nums[lim_idx + 1]) if n > lim_idx + 1 else None,
            }
            if n > lim_idx + 2:
                pkt["isr_us"] = nums[-1]

        self.telemetry.emit(pkt)

    def send(self, text: str) -> None:
        payload = (text + "\n").encode()
        if getattr(self, "_is_udp", False) and getattr(self, "_sock", None):
            try:
                self._sock.sendto(payload, (self._udp_ip, self._udp_port))
            except Exception as exc:
                self.error_msg.emit(f"UDP send error: {exc}")
        elif self._ser and self._ser.is_open:
            self._ser.write(payload)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def pulse_dtr(self, duration_ms: int = 50) -> None:
        if self._ser and self._ser.is_open:
            self._ser.dtr = True
            time.sleep(duration_ms / 1000.0)
            self._ser.dtr = False
