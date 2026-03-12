/**
 * stage1_echo.cpp — Stage 1: Bare USB Serial Echo + Heartbeat
 *
 * WHAT THIS TESTS:
 *   - Teensy boots and USB CDC serial works
 *   - You can send/receive data in the GUI Serial tab
 *
 * SUCCESS LOOKS LIKE:
 *   - You see "<STAGE1_READY>" immediately on connect
 *   - "<HEARTBEAT,1,N>" appears every second (N = counter)
 *   - Anything you send is echoed back
 *
 * NO libraries. NO encoders. NO steppers. NOTHING can crash this.
 */

#include <Arduino.h>

static uint32_t hb_count = 0;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
    delay(500);
    Serial.println("<STAGE1_READY>");
    Serial.println("# Send anything — it will be echoed back.");
    Serial.println("# Heartbeat prints every 1 second.");
}

void loop() {
    // Echo everything received
    while (Serial.available()) {
        char c = Serial.read();
        Serial.write(c);
    }

    // Heartbeat every 1 s
    static uint32_t last_hb = 0;
    uint32_t now = millis();
    if (now - last_hb >= 1000) {
        last_hb = now;
        digitalToggleFast(LED_BUILTIN);   // blink LED as visual confirm
        Serial.print("<HEARTBEAT,1,");
        Serial.print(hb_count++);
        Serial.println(">");
    }
}
