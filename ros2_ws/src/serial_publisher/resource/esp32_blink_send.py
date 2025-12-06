from machine import Pin
from time import sleep

# On many ESP32 boards, the onboard LED is on GPIO 2.
led = Pin(2, Pin.OUT)

while True:
    led.value(1)        # LED ON
    print("LED ON")
    sleep(1)
    
    led.value(0)        # LED OFF
    print("LED OFF")
    sleep(1)

#to listen to the serial port:
#ros2 topic echo /esp_serial
