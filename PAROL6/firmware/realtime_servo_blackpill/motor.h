/*
 * PAROL6 Motor Control — GPIO Step Pulse Generation (STM32F411CE)
 *
 * DDS phase accumulator generates step pulses from the 500 Hz control ISR.
 * DWT cycle counter provides sub-microsecond pulse width accuracy.
 * STM32Duino compatible.
 */

#ifndef MOTOR_H
#define MOTOR_H

#include "config.h"

void motorsInit(void);
void motorSetFrequency(uint8_t motor_idx, float frequency_hz);
void motorSetDirection(uint8_t motor_idx, bool forward);

#endif // MOTOR_H
