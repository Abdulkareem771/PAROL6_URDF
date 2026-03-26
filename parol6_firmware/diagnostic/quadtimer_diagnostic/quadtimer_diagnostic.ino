/**
 * quadtimer_diagnostic.ino
 * 
 * Minimal standalone QuadTimer PWM duty-cycle verification sketch.
 * Flash this to Teensy 4.1 to verify QuadTimer capture hardware works
 * BEFORE integrating into the full firmware stack.
 *
 * Setup:
 *   ESP32 PWM pin → Teensy pin 10 (J1 encoder).
 *   Recommended ESP32 frequency: 1000 Hz, any duty 1–99%.
 *   Open serial monitor at 115200 baud.
 *
 * Expected output (duty=50%, 1kHz):
 *   Pin10 duty=0.4999 angle=3.1414 rad | Pin11 duty=0.0000 | ...
 */

#include <Arduino.h>
#include "imxrt.h"

// ---------- Supported encoder pins (must match QuadTimerEncoder.h) ----------
// J1=10, J2=11, J3=12, J4=14, J5=15, J6=18
static const uint8_t ENC_PINS[6] = {10, 11, 12, 14, 15, 18};

struct QtChannel {
    volatile IMXRT_TMR_t* tmr;
    uint8_t ch;
    uint16_t last_cntr;
    uint32_t last_cyccnt;
};

static QtChannel qt[6];

// ---------- Init one QuadTimer channel ----------------------------------
static bool qt_init(uint8_t pin, QtChannel& q) {
    if      (pin == 10) { q.tmr = &IMXRT_TMR1; q.ch = 0; CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON); }
    else if (pin == 11) { q.tmr = &IMXRT_TMR1; q.ch = 2; CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON); }
    else if (pin == 12) { q.tmr = &IMXRT_TMR1; q.ch = 1; CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON); }
    else if (pin == 14) { q.tmr = &IMXRT_TMR3; q.ch = 2; CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON); IOMUXC_QTIMER3_TIMER2_SELECT_INPUT = 1; }
    else if (pin == 15) { q.tmr = &IMXRT_TMR3; q.ch = 3; CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON); IOMUXC_QTIMER3_TIMER3_SELECT_INPUT = 1; }
    else if (pin == 18) { q.tmr = &IMXRT_TMR3; q.ch = 1; CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON); IOMUXC_QTIMER3_TIMER1_SELECT_INPUT = 0; }
    else { q.tmr = nullptr; return false; }

    *(portConfigRegister(pin)) = 1 | 0x10;   // Alt-1 (QuadTimer) + SION
    q.tmr->CH[q.ch].CTRL  = 0;
    q.tmr->CH[q.ch].CNTR  = 0;
    q.tmr->CH[q.ch].LOAD  = 0;
    q.tmr->CH[q.ch].SCTRL = 0;
    q.tmr->CH[q.ch].CSCTRL= 0;
    // Gated count mode: count IP Bus/8 clock while pin is HIGH
    q.tmr->CH[q.ch].CTRL = TMR_CTRL_CM(6) | TMR_CTRL_PCS(8+3) | TMR_CTRL_SCS(q.ch) | TMR_CTRL_LENGTH;
    q.last_cntr   = q.tmr->CH[q.ch].CNTR;
    q.last_cyccnt = ARM_DWT_CYCCNT;
    return true;
}

// ---------- Read duty cycle (0.0–1.0) -----------------------------------
static float qt_read(QtChannel& q) {
    if (!q.tmr) return -1.0f;
    uint16_t cntr  = q.tmr->CH[q.ch].CNTR;
    uint32_t cyc   = ARM_DWT_CYCCNT;
    uint16_t hi    = cntr - q.last_cntr;
    uint32_t delta = cyc  - q.last_cyccnt;
    q.last_cntr   = cntr;
    q.last_cyccnt = cyc;
    float expected = (float)delta / (F_CPU_ACTUAL / F_BUS_ACTUAL) / 8.0f;
    if (expected < 1.0f) return 0.0f;
    float d = (float)hi / expected;
    return d < 0.0f ? 0.0f : (d > 1.0f ? 1.0f : d);
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);
    Serial.println("=== QuadTimer PWM Diagnostic ===");
    Serial.println("Pins tested: 10(J1) 11(J2) 12(J3) 14(J4) 15(J5) 18(J6)");
    Serial.println();
    ARM_DEMCR |= ARM_DEMCR_TRCENA;   // enable DWT cycle counter
    ARM_DWT_CYCCNT = 0;
    for (int i = 0; i < 6; i++) {
        bool ok = qt_init(ENC_PINS[i], qt[i]);
        Serial.printf("  Pin %2d (J%d): %s\n", ENC_PINS[i], i+1, ok?"INIT OK":"UNSUPPORTED");
    }
    Serial.println();
    delay(200);  // let first measurement window settle
}

void loop() {
    static uint32_t last_print = 0;
    if (millis() - last_print < 100) return;   // print at 10 Hz
    last_print = millis();

    Serial.printf("[%5.1fs]", millis() / 1000.0f);
    for (int i = 0; i < 6; i++) {
        float d = qt_read(qt[i]);
        if (d < 0)
            Serial.printf("  J%d=N/A", i+1);
        else
            Serial.printf("  J%d=%.4f(%.2frad)", i+1, d, d * 2.0f * PI);
    }
    Serial.println();

    // PASS/FAIL check: J1 (pin10) should show non-zero if ESP32 is connected
    float j1 = qt_read(qt[0]);
    if (j1 > 0.01f && j1 < 0.99f) {
        Serial.printf("  *** J1 PASS: duty=%.4f angle=%.3f rad ***\n", j1, j1 * 2 * PI);
    } else if (j1 <= 0.01f) {
        Serial.println("  J1 reads 0 — check: pin10 connected? ESP32 PWM running?");
    }
}
