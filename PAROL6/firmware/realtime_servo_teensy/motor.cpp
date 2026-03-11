/*
 * PAROL6 Motor Control — FlexPWM Step Generation (Teensy 4.1)
 *
 * Each motor's STEP pin produces a 50% duty square wave via FlexPWM.
 * Frequency changes are instant via analogWriteFrequency().
 *
 * No pin conflicts, no timer sharing issues — each STEP pin is on
 * a different FlexPWM submodule.
 */

#include "motor.h"

// 50% duty for 8-bit resolution: 128/256
#define PWM_DUTY_50 128

// Track running state so we don't call analogWrite(0) every control cycle
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

    // STEP pin — configure FlexPWM channel
    pinMode(STEP_PINS[i], OUTPUT);
    analogWriteResolution(PWM_RESOLUTION);     // 8-bit resolution
    analogWriteFrequency(STEP_PINS[i], 1000);  // Initial 1 kHz
    analogWrite(STEP_PINS[i], 0);              // Start stopped
  }

  // Proximity sensor pins — INPUT_PULLUP
  // Optocoupler pulls to GND when sensor triggers → read LOW
  for (uint8_t i = 0; i < NUM_PROX_SENSORS; i++) {
    pinMode(PROX_PINS[i], INPUT_PULLUP);
  }
}

// ============================================================================
// SET FREQUENCY  (called from control ISR, ~500 Hz, per motor)
// ============================================================================

void motorSetFrequency(uint8_t motor_idx, float frequency_hz) {
  if (motor_idx >= NUM_MOTORS) return;

  if (frequency_hz < 10.0f) {
    // Stop — set duty to 0 (pin stays LOW, no pulses)
    if (motor_running[motor_idx]) {
      analogWrite(STEP_PINS[motor_idx], 0);
      motor_running[motor_idx] = false;
    }
    return;
  }

  // Clamp to maximum step frequency
  if (frequency_hz > (float)MAX_STEP_FREQUENCY_HZ)
    frequency_hz = (float)MAX_STEP_FREQUENCY_HZ;

  // Change frequency (only affects this pin's FlexPWM submodule)
  analogWriteFrequency(STEP_PINS[motor_idx], frequency_hz);

  if (!motor_running[motor_idx]) {
    analogWrite(STEP_PINS[motor_idx], PWM_DUTY_50);  // 50% duty → pulses
    motor_running[motor_idx] = true;
  }
}

// ============================================================================
// DIRECTION
// ============================================================================

void motorSetDirection(uint8_t motor_idx, bool forward) {
  if (motor_idx >= NUM_MOTORS) return;
  digitalWriteFast(DIR_PINS[motor_idx], forward ? HIGH : LOW);
}
