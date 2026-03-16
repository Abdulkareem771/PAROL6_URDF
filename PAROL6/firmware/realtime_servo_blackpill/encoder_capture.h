/*
 * PAROL6 Encoder Capture — STM32 PWM Input Mode (Hardware Duty Cycle)
 *
 * Each encoder gets a DEDICATED timer in PWM Input mode:
 *   CH1 captures rising edge  → CCR1 = period (ticks)
 *   CH2 captures falling edge → CCR2 = high-time (ticks)
 *   Both mapped to TI1 (same physical pin)
 *   Slave reset mode: counter resets on rising edge
 *   ICxF = 9 digital filter: 667 ns hardware glitch rejection
 *
 * The CPU never services an interrupt for encoder reading.
 * The control ISR simply polls CCR1 and CCR2 from each timer.
 *
 * duty = CCR2 / CCR1
 *
 * Timers used:
 *   TIM1  → J1 (PA8,  AF1)
 *   TIM2  → J2 (PA15, AF1)
 *   TIM3  → J3 (PA6,  AF2)
 *   TIM4  → J4 (PB6,  AF2)
 *   TIM5  → J5 (PA0,  AF2)
 *   TIM9  → J6 (PA2,  AF3)
 */

#ifndef ENCODER_CAPTURE_H
#define ENCODER_CAPTURE_H

#include "config.h"

// Initialize all 6 timers in PWM Input mode
void encoderCaptureInit(void);

// Read the latest duty cycle from hardware registers (no interrupt involved)
// Returns pulse width in timer ticks (CCR2 value = high-time)
// Also provides period in timer ticks (CCR1 value)
// Call with interrupts disabled (from control ISR context)
void encoderReadCapture(uint8_t enc_idx, uint32_t *period_ticks, uint32_t *hightime_ticks);

#endif // ENCODER_CAPTURE_H
