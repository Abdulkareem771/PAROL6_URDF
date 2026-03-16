/*
 * PAROL6 Real-Time Servo Control — STM32F411CE Black Pill
 *
 * Arduino IDE / STM32Duino sketch.
 *
 * Board settings in Arduino IDE:
 *   Board:        "Generic STM32F4 series"
 *   Board part:   "BlackPill F411CE"
 *   USB support:  "CDC (generic 'Serial' supersede U(S)ART)"
 *   Upload:       "STM32CubeProgrammer (DFU)"
 *
 * Architecture:
 *   6 timers (TIM1-5, TIM9) in PWM Input mode → encoder capture (ZERO ISR load)
 *   TIM11 update ISR                           → controlUpdate()  @ 500 Hz
 *   Main loop()                                → serial I/O       @ 50 Hz
 *
 * PWM Input mode:
 *   For each encoder, the timer hardware does ALL edge timing.
 *   CH1 captures the period (rising-to-rising), CH2 captures high-time.
 *   Slave reset mode resets the counter on each rising edge.
 *   The control ISR simply reads CCR1/CCR2 — no encoder interrupts needed.
 *
 * DFU upload:
 *   Hold BOOT0 → press RESET → release → upload in Arduino IDE.
 *   PA10 must be pulled to GND (Black Pill HW bug). PA10 is NOT used.
 *
 * STM32F411CE @ 96 MHz (PLL: HSE 25 MHz, PLLM=25, PLLN=192, PLLP=2, PLLQ=4)
 *   96 MHz SYSCLK, 48 MHz USB clock (exact)
 *   667 ns digital filter on encoder inputs — matches Teensy QTimer FILT
 */

#include "config.h"
#include "control.h"
#include "serial_comm.h"

// ============================================================================
// SETUP
// ============================================================================

void setup()
{
    // STM32Duino handles clock config (96 MHz), SystemInit, and USB CDC.
    // No manual PLL setup needed when using the Arduino framework.

    // Initialize serial (USB CDC 12 Mbps)
    serialCommInit();

    // Initialize control system (encoders in PWM Input mode, motors, TIM11 ISR)
    controlInit();

    // Onboard LED: PC13 (active LOW on Black Pill)
    pinMode(PC13, OUTPUT);
    digitalWrite(PC13, LOW);   // LED ON — boot indicator
    delay(200);
    digitalWrite(PC13, HIGH);  // LED OFF
}

// ============================================================================
// MAIN LOOP — serial I/O at 50 Hz
// ============================================================================

static uint32_t last_feedback_ms = 0;

void loop()
{
    // Process incoming commands (non-blocking)
    serialCommProcessIncoming();

    // Send feedback at 50 Hz
    uint32_t now = millis();
    if ((now - last_feedback_ms) >= FEEDBACK_PERIOD_MS) {
        last_feedback_ms = now;
        serialCommSendFeedback();
    }
}
