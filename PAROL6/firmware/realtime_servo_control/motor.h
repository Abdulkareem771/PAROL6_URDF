/*
 * PAROL6 Motor Control - LEDC PWM Based Step Generation
 *
 * Uses ESP32 LEDC (LED Control) peripheral for step pulse generation.
 * 8 hardware channels, independent frequencies, zero ISR overhead.
 * Frequency changes are instant via ledcChangeFrequency().
 *
 * Why LEDC instead of hw_timer:
 *   - ESP32 has only 4 hw timers but 8 LEDC channels
 *   - No ISR needed → no stack / IRAM concerns
 *   - ledcChangeFrequency() is atomic — no stop/destroy/create cycle
 */

#ifndef MOTOR_H
#define MOTOR_H

#include <Arduino.h>
#include "config.h"

// Initialize all motors (LEDC channels + direction pins)
void motorsInit();

// Set step frequency in Hz (called from control task at 500 Hz)
void motorSetFrequency(uint8_t motor_idx, float frequency_hz);

// Set direction (true = forward / HIGH, false = reverse / LOW)
void motorSetDirection(uint8_t motor_idx, bool forward);

#endif // MOTOR_H
