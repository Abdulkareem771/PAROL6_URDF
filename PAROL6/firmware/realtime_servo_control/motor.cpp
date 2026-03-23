/*
 * PAROL6 Motor Control — LEDC PWM Step Generation
 *
 * Each motor's STEP pin is driven by an independent LEDC channel.
 * LEDC generates a 50 % duty-cycle square wave at the desired step
 * frequency — pure hardware, zero CPU overhead per pulse.
 *
 * IMPORTANT — TIMER SHARING BUG IN ARDUINO CORE 3.x:
 *   ledcAttach() calls find_matching_timer() which reuses an existing
 *   timer if freq+resolution match.  If all motors start at the same
 *   frequency they all share ONE timer, so ledcChangeFrequency() on
 *   any motor changes the frequency for ALL motors.
 *   FIX: use ledcAttachChannel() to manually assign each motor to a
 *   separate channel+timer.
 *
 * PIN CONFLICT PROTECTION:
 *   Some DIR/STEP pins share GPIO with encoder input pins.
 *   Writing to a shared pin would trigger the encoder ISR with
 *   garbage timing, corrupting position readings.
 *   motorsInit() detects these conflicts and skips those pins.
 *
 * ESP32 LEDC Core 3.x API:
 *   ledcAttachChannel(pin, freq, res, channel) — bind pin to channel
 *   ledcWrite(pin, duty)                       — set duty (0 = stop)
 *   ledcChangeFrequency(pin, freq, res)        — change freq on the fly
 */

#include "motor.h"

// LEDC resolution (8-bit = 0..255)
#define LEDC_RES  8
#define LEDC_DUTY 128        // 50 % of 256 → clean square wave

// Track running state so we don't call ledcWrite(0) every control cycle
static bool motor_running[NUM_MOTORS];

// Track which DIR/STEP pins conflict with enabled encoder pins
static bool dir_pin_conflict[NUM_MOTORS];
static bool step_pin_conflict[NUM_MOTORS];

// ============================================================================
// LEDC CHANNEL ASSIGNMENT
// ============================================================================
//
// ESP32 has 8 high-speed LEDC channels and 4 timers.
// Channel-to-timer mapping: timer = channel / 2
//   Channels 0,1 → Timer 0      Channels 4,5 → Timer 2
//   Channels 2,3 → Timer 1      Channels 6,7 → Timer 3
//
// To give each motor its OWN timer (critical for independent freq control),
// we assign motors to channels on DIFFERENT timers:
//   Motor 0 (J1) → Channel 0 (Timer 0)
//   Motor 1 (J2) → Channel 2 (Timer 1)
//   Motor 2 (J3) → Channel 4 (Timer 2)
//   Motor 3 (J4) → Channel 6 (Timer 3)
//   Motor 4 (J5) → Channel 1 (Timer 0, shares with J1 — J5 not active)
//   Motor 5 (J6) → Channel 3 (Timer 1, shares with J2 — J6 not active)
//
static const uint8_t LEDC_CHANNEL[NUM_MOTORS] = { 0, 2, 4, 6, 1, 3 };

// ============================================================================
// INIT
// ============================================================================

void motorsInit() {
  // Detect pin conflicts between motor outputs and encoder inputs
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    dir_pin_conflict[i] = false;
    step_pin_conflict[i] = false;

    for (uint8_t j = 0; j < NUM_MOTORS; j++) {
      if (!ENCODER_ENABLED[j]) continue;
      if (DIR_PINS[i] == ENCODER_PINS[j]) {
        dir_pin_conflict[i] = true;
      }
      if (STEP_PINS[i] == ENCODER_PINS[j]) {
        step_pin_conflict[i] = true;
      }
    }
  }

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    motor_running[i] = false;

    // Direction pin — SKIP if it conflicts with an encoder input
    if (!dir_pin_conflict[i]) {
      pinMode(DIR_PINS[i], OUTPUT);
      digitalWrite(DIR_PINS[i], LOW);
    }

    // STEP pin — SKIP LEDC if it conflicts with an encoder input
    // Use ledcAttachChannel() with UNIQUE initial frequencies so
    // Arduino core assigns each channel its own timer.
    if (!step_pin_conflict[i]) {
      uint32_t init_freq = 1000 + i * 100;  // 1000, 1100, 1200, ...
      ledcAttachChannel(STEP_PINS[i], init_freq, LEDC_RES, LEDC_CHANNEL[i]);
      ledcWrite(STEP_PINS[i], 0);   // Start stopped (0 % duty)
    }
  }
}

// ============================================================================
// SET FREQUENCY  (called from control task, ~500 Hz, per motor)
// ============================================================================

void motorSetFrequency(uint8_t motor_idx, float frequency_hz) {
  if (motor_idx >= NUM_MOTORS) return;
  if (step_pin_conflict[motor_idx]) return;  // Can't use this pin

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

  // Change frequency (only affects THIS motor's timer now)
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
  if (dir_pin_conflict[motor_idx]) return;  // Don't touch — used by encoder!
  digitalWrite(DIR_PINS[motor_idx], forward ? HIGH : LOW);
}
