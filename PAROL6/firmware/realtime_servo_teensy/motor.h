/*
 * PAROL6 Motor Control — FlexPWM Step Generation (Teensy 4.1)
 *
 * Each motor's STEP pin is driven by a FlexPWM submodule.
 * analogWriteFrequency() sets the PWM frequency per submodule.
 * analogWrite(pin, 128) produces a 50% duty square wave.
 *
 * Teensy 4.1 advantages over ESP32 LEDC:
 *   - 16 FlexPWM submodules (vs 4 shared timers)
 *   - Each STEP pin on different submodule = independent frequency
 *   - No pin conflict detection needed (55 GPIO, all unique)
 *   - analogWriteFrequency() change is instant, no timer recreation
 *
 * FlexPWM API:
 *   analogWriteFrequency(pin, freq)  — set frequency for the submodule
 *   analogWrite(pin, duty)           — start/stop pulses (0 = stop)
 */

#ifndef MOTOR_H
#define MOTOR_H

#include <Arduino.h>
#include "config.h"

// Initialize all motors (FlexPWM channels + direction pins)
void motorsInit();

// Set step frequency in Hz (called from control ISR at 500 Hz)
void motorSetFrequency(uint8_t motor_idx, float frequency_hz);

// Set direction (true = forward / HIGH, false = reverse / LOW)
void motorSetDirection(uint8_t motor_idx, bool forward);

#endif // MOTOR_H
