/*
 * PAROL6 Motor Control — LEDC PWM Step Generation
 *
 * Each motor's STEP pin is driven by an independent LEDC channel.
 * LEDC generates a 50 % duty-cycle square wave at the desired step
 * frequency — pure hardware, zero CPU overhead per pulse.
 *
 * ESP32 LEDC Core 3.x API:
 *   ledcAttach(pin, freq, resolution)       — bind pin to channel
 *   ledcWrite(pin, duty)                    — set duty (0 = stop)
 *   ledcChangeFrequency(pin, freq, res)     — change freq on the fly
 *   ledcDetach(pin)                         — release pin
 */

#include "motor.h"

// LEDC resolution (8-bit = 0..255)
#define LEDC_RES  8
#define LEDC_DUTY 128        // 50 % of 256 → clean square wave
#define LEDC_INIT_FREQ 1000  // 1 kHz initial (overwritten immediately)

// Track running state so we don't call ledcWrite(0) every control cycle
static bool motor_running[NUM_MOTORS];

// ============================================================================
// INIT
// ============================================================================

void motorsInit() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    motor_running[i] = false;

    // Direction pin
    pinMode(DIR_PINS[i], OUTPUT);
    digitalWrite(DIR_PINS[i], LOW);

    // Attach STEP pin to LEDC channel at initial frequency
    ledcAttach(STEP_PINS[i], LEDC_INIT_FREQ, LEDC_RES);
    ledcWrite(STEP_PINS[i], 0);   // Start stopped (0 % duty)
  }
}

// ============================================================================
// SET FREQUENCY  (called from control task, ~500 Hz, per motor)
// ============================================================================

void motorSetFrequency(uint8_t motor_idx, float frequency_hz) {
  if (motor_idx >= NUM_MOTORS) return;

  if (frequency_hz < 10.0f) {
    // Stop — set duty to 0 (pin stays LOW, no pulses)
    // LEDC can't reliably do < 10 Hz at 8-bit resolution
    if (motor_running[motor_idx]) {
      ledcWrite(STEP_PINS[motor_idx], 0);
      motor_running[motor_idx] = false;
    }
    return;
  }

  // Clamp to maximum step frequency
  if (frequency_hz > (float)MAX_STEP_FREQUENCY_HZ)
    frequency_hz = (float)MAX_STEP_FREQUENCY_HZ;

  // Change frequency (hardware-atomic, no ISR, no timer recreation)
  ledcChangeFrequency(STEP_PINS[motor_idx],
                      (uint32_t)frequency_hz,
                      LEDC_RES);

  if (!motor_running[motor_idx]) {
    ledcWrite(STEP_PINS[motor_idx], LEDC_DUTY);   // 50 % duty → pulses
    motor_running[motor_idx] = true;
  }
}

// ============================================================================
// DIRECTION
// ============================================================================

void motorSetDirection(uint8_t motor_idx, bool forward) {
  if (motor_idx >= NUM_MOTORS) return;
  digitalWrite(DIR_PINS[motor_idx], forward ? HIGH : LOW);
}
