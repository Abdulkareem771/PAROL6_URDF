/**
 * stage3_encoder.cpp — Stage 3: Real J6 Encoder (SoftwareInterrupt)
 *
 * WHAT THIS TESTS:
 *   - SoftwareInterruptEncoder reads pulses on pin 18
 *   - J6 encoder wiring and direction are correct
 *   - Real position value appears in ACK packets
 *
 * SUCCESS LOOKS LIKE:
 *   - J6 value in ACK changes when you physically rotate J6 output shaft
 *   - Positive rotation = increasing value (or consistent direction)
 *   - Value stays stable when shaft is still
 *
 * HARDWARE: Encoder A-phase on Teensy pin 18 only (no B-phase needed here).
 *           The SoftwareInterruptEncoder counts rising edges and converts via
 *           PULSES_PER_REV and gear ratio.
 */

#include <Arduino.h>
#include "hal/SoftwareInterruptEncoder.h"

// ── J6 encoder config ────────────────────────────────────────────────────────
static constexpr int   ENC_PIN      = 19;     // ← CORRECTED: pin 19 (was 18)
static constexpr int   ENC_IDX      = 5;      // Axis slot
static constexpr float GEAR_RATIO   = 10.0f;  // Adjust to your actual gearbox
static constexpr int   PULSES_REV   = 400;    // Encoder pulses/rev (motor shaft)

// ── Globals ──────────────────────────────────────────────────────────────────
static SoftwareInterruptEncoder enc(ENC_PIN, ENC_IDX);
static uint32_t seq     = 0;
static bool     enabled = false;
static uint32_t last_fb = 0;
static String   buf;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);

    enc.init();   // Sets up interrupt on ENC_PIN

    delay(500);
    Serial.println("<STAGE3_READY>");
    Serial.print("# Encoder on pin "); Serial.print(ENC_PIN);
    Serial.println(" — rotate J6 shaft and watch J6 position change.");
    Serial.println("# MKServo42C enable is MANUAL (physical button on driver).");
    Serial.println("# Send <ZERO> to reset encoder reference to 0.");
}

void loop() {
    // Parse incoming
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            buf.trim();
            if (buf == "<ENABLE>") {
                enabled = true;
                Serial.println("<ENABLE_ACK>");
            } else if (buf == "<ZERO>" || buf == "<HOME>") {
                enc.init();  // reset counts
                Serial.println("<ZERO_ACK>");
            }
            buf = "";
        } else {
            buf += c;
        }
    }

    // Feedback at 25 Hz
    uint32_t now = millis();
    if (now - last_fb >= 40) {
        last_fb = now;
        digitalToggleFast(LED_BUILTIN);

        float pos_j6 = enc.read_angle();   // radians at OUTPUT shaft
        uint8_t state_byte = enabled ? 1 : 3;

        Serial.print("<ACK,"); Serial.print(seq++);
        Serial.print(",0.0000,0.0000,0.0000,0.0000,0.0000,");
        Serial.print(pos_j6, 4);
        Serial.print(",0.0000,0.0000,0.0000,0.0000,0.0000,0.0000");
        Serial.print(",0,");
        Serial.print(state_byte);
        Serial.println(">");
    }
}
