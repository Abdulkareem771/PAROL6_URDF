#send ON to turn LED on
#send OFF to turn LED off
#send BLINK to blink LED

# main.py (MicroPython)
from machine import Pin, UART
import time

# LED pin (change if your board uses different pin)
LED_PIN = 2

led = Pin(LED_PIN, Pin.OUT)
uart = UART(1, 115200)   # USB serial on many ESP32 boards

# Blink state
blink = False
blink_interval = 0.5  # seconds
last_ms = time.ticks_ms()

def set_on():
    global blink
    blink = False
    led.value(1)
    print("OK:ON")

def set_off():
    global blink
    blink = False
    led.value(0)
    print("OK:OFF")

def set_blink():
    global blink, last_ms
    blink = True
    last_ms = time.ticks_ms()
    print("OK:BLINK")

# initial state
print("ESP32 ready")
set_off()

while True:
    # read incoming command (line-based)
    if uart.any():
        line = uart.readline()
        if line:
            try:
                cmd = line.decode('utf-8', 'ignore').strip().upper()
            except Exception:
                cmd = ''
            if cmd == 'ON':
                set_on()
            elif cmd == 'OFF':
                set_off()
            elif cmd == 'BLINK':
                set_blink()
            else:
                # unknown command -> send error back
                print("ERR:UNKNOWN_CMD:", cmd)

    # handle blinking without blocking
    if blink:
        now = time.ticks_ms()
        if time.ticks_diff(now, last_ms) >= int(blink_interval * 1000):
            led.value(not led.value())
            last_ms = now

    time.sleep(0.05)
