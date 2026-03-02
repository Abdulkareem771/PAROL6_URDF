/**
 * main.cpp — QuadTimer PWM Diagnostic (v3)
 *
 * Uses IntervalTimer at 1 ms to sample the QuadTimer counter correctly.
 * The 16-bit counter runs at IP Bus/8 = ~18.75 MHz.
 * In 1 ms it accumulates ~18,750 ticks — fits in uint16_t (max 65535).
 * Accumulating 100 x 1ms samples = 100ms display window is done in uint32_t.
 *
 * Setup:
 *   Connect 3.3V → Teensy pin 10 → expect J1 ≈ 1.000
 *   Connect GND  → Teensy pin 10 → expect J1 ≈ 0.000
 *   Connect ESP32 PWM (1 kHz, 50%)  → expect J1 ≈ 0.500
 */

#include <Arduino.h>
#include "imxrt.h"

// ── Encoder pin map ────────────────────────────────────────────────────────
static const uint8_t ENC_PINS[6] = {10, 11, 12, 14, 15, 18};

struct QtCh {
    volatile IMXRT_TMR_t* tmr = nullptr;
    uint8_t ch = 0;
};
static QtCh qt[6];

// ── Sampling state (written from 1ms ISR, read from loop) ─────────────────
static volatile uint16_t s_last[6]  = {0};   // last counter snapshot
static volatile uint32_t s_hi[6]    = {0};   // accumulated HIGH ticks
static volatile uint32_t s_total    = 0;      // accumulated expected ticks
static volatile float    s_duty[6]  = {0};   // published duty per channel
static volatile bool     s_ready    = false;

IntervalTimer sampleTimer;

// Expected ticks per 1 ms at PCS=11 (IP Bus / 8)
static const uint32_t TICKS_PER_MS = F_BUS_ACTUAL / 8 / 1000;   // = 18750
static const uint32_t TICKS_100MS  = TICKS_PER_MS * 100;         // display window

static void FASTRUN onSample() {
    for (int i = 0; i < 6; i++) {
        if (!qt[i].tmr) continue;
        uint16_t c   = qt[i].tmr->CH[qt[i].ch].CNTR;
        uint16_t inc = c - s_last[i];   // 16-bit wrap-safe delta
        s_hi[i]     += inc;
        s_last[i]    = c;
    }
    s_total += TICKS_PER_MS;

    if (s_total >= TICKS_100MS) {
        for (int i = 0; i < 6; i++) {
            float d = (s_total > 0) ? (float)s_hi[i] / (float)s_total : 0.f;
            s_duty[i] = constrain(d, 0.f, 1.f);
            s_hi[i]   = 0;
            // Re-sync snapshot so delta starts clean
            if (qt[i].tmr) s_last[i] = qt[i].tmr->CH[qt[i].ch].CNTR;
        }
        s_total  = 0;
        s_ready  = true;
    }
}

// ── Channel init ───────────────────────────────────────────────────────────
static bool qt_init(uint8_t pin, QtCh& q) {
    if      (pin == 10) { q.tmr = &IMXRT_TMR1; q.ch = 0; CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON); }
    else if (pin == 11) { q.tmr = &IMXRT_TMR1; q.ch = 2; CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON); }
    else if (pin == 12) { q.tmr = &IMXRT_TMR1; q.ch = 1; CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON); }
    else if (pin == 14) { q.tmr = &IMXRT_TMR3; q.ch = 2; CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON); IOMUXC_QTIMER3_TIMER2_SELECT_INPUT = 1; }
    else if (pin == 15) { q.tmr = &IMXRT_TMR3; q.ch = 3; CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON); IOMUXC_QTIMER3_TIMER3_SELECT_INPUT = 1; }
    else if (pin == 18) { q.tmr = &IMXRT_TMR3; q.ch = 1; CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON); IOMUXC_QTIMER3_TIMER1_SELECT_INPUT = 0; }
    else { q.tmr = nullptr; return false; }

    // Pad mux → Alt-1 (QuadTimer) + SION (software input on)
    *(portConfigRegister(pin)) = 1 | 0x10;

    // Reset all timer channel registers
    q.tmr->CH[q.ch].CTRL   = 0;
    q.tmr->CH[q.ch].CNTR   = 0;
    q.tmr->CH[q.ch].LOAD   = 0;
    q.tmr->CH[q.ch].COMP1  = 0xFFFF;
    q.tmr->CH[q.ch].SCTRL  = 0;
    q.tmr->CH[q.ch].CSCTRL = 0;

    // CM(3) = Gated Count: increments primary (IP Bus/8 clock) ONLY while
    // secondary input (this encoder pin, selected by SCS=ch) is HIGH.
    // No LENGTH flag — without it the counter free-runs 0→0xFFFF→wrap.
    q.tmr->CH[q.ch].CTRL = TMR_CTRL_CM(3)
                          | TMR_CTRL_PCS(8 + 3)   // IP Bus / 8 = 18.75 MHz
                          | TMR_CTRL_SCS(q.ch);    // secondary = this channel's pin
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Serial.println("\n=== QuadTimer PWM Diagnostic v3 (1ms ISR sampling) ===");
    Serial.printf("F_CPU=%lu  F_BUS=%lu  TICKS_PER_MS=%lu\n",
                  F_CPU_ACTUAL, F_BUS_ACTUAL, TICKS_PER_MS);
    Serial.println("Initialising channels...");

    for (int i = 0; i < 6; i++) {
        bool ok = qt_init(ENC_PINS[i], qt[i]);
        Serial.printf("  J%d  pin%2d : %s\n", i+1, ENC_PINS[i], ok?"INIT OK":"UNSUPPORTED");
    }

    // Prime the snapshots
    for (int i = 0; i < 6; i++)
        if (qt[i].tmr) s_last[i] = qt[i].tmr->CH[qt[i].ch].CNTR;

    // Start 1ms sampler
    sampleTimer.begin(onSample, 1000);   // 1000 µs = 1 ms

    Serial.println("\nSampling at 1ms, displaying every 100ms.");
    Serial.println("Connect 3.3V to Teensy pin 10 — expect J1 ≈ 1.000\n");
}

void loop() {
    if (!s_ready) return;
    s_ready = false;

    // Snapshot duty (written atomically enough for floats on ARM)
    float d[6];
    for (int i = 0; i < 6; i++) d[i] = s_duty[i];

    Serial.printf("[%5.1fs]", millis() / 1000.0f);
    for (int i = 0; i < 6; i++)
        Serial.printf("  J%d=%.4f(%.2frad)", i+1, d[i], d[i] * TWO_PI);
    Serial.println();

    // PASS/FAIL for J1
    if (d[0] > 0.01f && d[0] < 0.99f)
        Serial.printf("  *** J1 PASS  duty=%.4f  angle=%.3f rad ***\n", d[0], d[0]*TWO_PI);
    else if (d[0] >= 0.99f)
        Serial.println("  J1 = 1.000 → pin10 stuck HIGH (3.3V jumper connected) — TIMER WORKS!");
    else
        Serial.println("  J1 = 0 → pin10 not driven. Check wiring.");
}
