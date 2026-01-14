#!/usr/bin/env python3
import serial
import time
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 virtual_esp32.py <serial_port>")
        sys.exit(1)

    port = sys.argv[1]
    print(f"[VIRTUAL ESP32] Starting on {port}...")

    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
    except Exception as e:
        print(f"[ERROR] Could not open port {port}: {e}")
        sys.exit(1)

    print("[VIRTUAL ESP32] Booting...")
    time.sleep(2)
    # Simulate the "READY" signal after homing
    ser.write(b"READY: VIRTUAL_ESP32\n")
    print("[VIRTUAL ESP32] Sent READY signal. Waiting for commands...")

    while True:
        try:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if line.startswith('<') and line.endswith('>'):
                    # Parse the command <J1,J2,J3,J4,J5,J6>
                    content = line[1:-1]
                    joints = content.split(',')
                    print(f"[CMD] Moving Joints: {joints}")
                else:
                    print(f"[RAW] {line}")
            time.sleep(0.001)
        except KeyboardInterrupt:
            print("[VIRTUAL ESP32] Shutting down.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            break

if __name__ == '__main__':
    main()
