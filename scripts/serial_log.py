#!/usr/bin/env python3
"""
serial_log.py — Diagnose PAROL6 Teensy serial feedback in real time.

Usage (on host, inside Docker, or anywhere with pyserial):
    python3 scripts/serial_log.py /dev/ttyACM0 115200

Output per line:
    [seq]  J1:  0.0000  J2:  0.0000  ...  J6:  0.0000  |  lim:00  state:1  isr:-- us
Press Ctrl+C to stop.
"""

import sys
import time
import re

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run:  pip install pyserial")
    sys.exit(1)

ACK_RE = re.compile(r"<ACK,(\d+),([.\d,\-]+)>")

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

print(f"Opening {PORT} @ {BAUD} baud — press Ctrl+C to stop\n")

prev_seq = None
pkt_count = 0
t0 = time.monotonic()

try:
    with serial.Serial(PORT, BAUD, timeout=1.0) as ser:
        while True:
            raw = ser.readline()
            if not raw:
                print("[TIMEOUT — no data from Teensy]")
                continue
            line = raw.decode("utf-8", errors="replace").strip()

            m = ACK_RE.match(line)
            if not m:
                # Print non-ACK lines in yellow for debugging
                print(f"\033[93m[RAW] {line}\033[0m")
                continue

            seq = int(m.group(1))
            nums = [float(x) for x in m.group(2).split(",") if x]

            if len(nums) < 12:
                print(f"[WARN] Short packet ({len(nums)} fields): {line}")
                continue

            pos = nums[0:6]
            vel = nums[6:12]
            lim = int(nums[12]) if len(nums) > 12 else None
            state = int(nums[13]) if len(nums) > 13 else None
            isr = nums[-1] if len(nums) > 14 else None

            gap = "" if prev_seq is None else (f"  \033[91mLOST {seq - prev_seq - 1}\033[0m" if seq != prev_seq + 1 else "")
            prev_seq = seq
            pkt_count += 1

            pos_str = "  ".join(f"J{i+1}:{p:+7.4f}" for i, p in enumerate(pos))
            state_names = {1: "NOMINAL", 2: "HOMING", 3: "FAULT/ESTOP"}
            state_str = state_names.get(state, f"?{state}") if state is not None else "n/a"
            lim_str = f"{lim:06b}" if lim is not None else "------"
            isr_str = f"{isr:.1f}µs" if isr is not None else "--µs"

            # Print; highlight if any joint moves > 0.001 rad
            changed = any(abs(p) > 0.001 for p in pos)
            color = "\033[92m" if changed else "\033[0m"
            print(f"{color}[{seq:6d}] {pos_str}  | lim:{lim_str} state:{state_str} isr:{isr_str}{gap}\033[0m")

            # Print stats every 5 seconds
            elapsed = time.monotonic() - t0
            if elapsed > 5.0:
                print(f"\033[96m--- Rate: {pkt_count / elapsed:.1f} pkt/s ---\033[0m")
                pkt_count = 0
                t0 = time.monotonic()

except KeyboardInterrupt:
    print("\nStopped.")
except serial.SerialException as e:
    print(f"Serial error: {e}")
    print("Is the port correct? Try: ls /dev/ttyACM* /dev/ttyUSB*")
    sys.exit(1)
