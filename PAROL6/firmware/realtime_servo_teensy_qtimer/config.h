/*
 * PAROL6 Real-Time Servo Control — Configuration (Teensy 4.1, QTimer Edition)
 *
 * Uses hardware QTimer input capture for EMI-immune encoder reading.
 * Encoder pins MUST be on QTimer-capable pins:
 *   TMR1: Pin 10 (CH0), Pin 12 (CH1), Pin 11 (CH2)
 *   TMR3: Pin 19 (CH0), Pin 18 (CH1), Pin 14 (CH2), Pin 15 (CH3)
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================

#define NUM_MOTORS 6

// Step pins — each on a DIFFERENT FlexPWM submodule for independent frequency
const int STEP_PINS[NUM_MOTORS] = {
  2,   // J1 → FlexPWM4.2A
  4,   // J2 → FlexPWM2.2A
  6,   // J3 → FlexPWM1.3B
  8,   // J4 → FlexPWM1.3A
  5,   // J5 → FlexPWM2.0A
  7    // J6 → FlexPWM2.1A
};

// Direction pins — simple digital outputs
const int DIR_PINS[NUM_MOTORS] = {
  24,  // J1
  35,  // J2
  26,  // J3
  27,  // J4
  36,  // J5
  34   // J6
};

// ============================================================================
// ENCODER PIN ASSIGNMENTS  (QTimer-capable pins ONLY)
// ============================================================================
// *** Pin 16 and 17 are NOT QTimer-capable! ***
// *** J3 changed: 16 → 10,  J4 changed: 17 → 15 ***
const int ENCODER_PINS[NUM_MOTORS] = {
  14,  // J1 → TMR3_CH2  (GPIO_AD_B1_02, ALT1)
  12,  // J2 → TMR1_CH1  (GPIO_B0_01, ALT1)
  10,  // J3 → TMR1_CH0  (GPIO_B0_00, ALT1)  *** CHANGED from 16 ***
  15,  // J4 → TMR3_CH3  (GPIO_AD_B1_03, ALT1)  *** CHANGED from 17 ***
  18,  // J5 → TMR3_CH0  (GPIO_AD_B1_00, ALT1)
  19   // J6 → TMR3_CH1  (GPIO_AD_B1_01, ALT1)
};

// QTimer module and channel for each encoder
// 0 = TMR1, 2 = TMR3 (TMR2/TMR4 not used)
const uint8_t ENC_TMR_MODULE[NUM_MOTORS] = { 2, 0, 0, 2, 2, 2 }; // TMR3, TMR1, TMR1, TMR3, TMR3, TMR3
const uint8_t ENC_TMR_CHANNEL[NUM_MOTORS] = { 2, 1, 0, 3, 0, 1 };

// Encoder enable flags
const bool ENCODER_ENABLED[NUM_MOTORS] = {
  false,  // J1
  false,   // J2
  false,  // J3
  false,  // J4
  true,   // J5
  true   // J6
};

// ============================================================================
// INDUCTIVE PROXIMITY SENSORS (homing)
// ============================================================================

#define NUM_PROX_SENSORS 3
const int PROX_PINS[NUM_PROX_SENSORS] = { 20, 21, 22 };

// ============================================================================
// MOTOR CONFIGURATION
// ============================================================================

#define STEPS_PER_REV 200

const int MICROSTEPS[NUM_MOTORS] = { 4, 16, 16, 16, 16, 16 };

const float GEAR_RATIOS[NUM_MOTORS] = {
  20.0,  // J1: 20:1 gearbox
  20.0,  // J2
  1.0,   // J3
  4.0,   // J4
  4.0,  // J5
  10.0    // J6
};

const int MOTOR_DIR_SIGN[NUM_MOTORS] = { 1, 1, 1, 1, 1, -1 };

// ============================================================================
// ENCODER CONFIGURATION
// ============================================================================

// MT6816 PWM encoding
const float ENCODER_CLOCK_PERIOD_NS = 250.0;
const int ENCODER_START_CLOCKS = 16;
const int ENCODER_RESOLUTION = 4096;

// QTimer clock: IPG_CLK (150 MHz) / 8 = 18.75 MHz
// Tick period: 53.33 ns.  16-bit overflow: 3495 us (>> MT6816 max 1028us)
#define QTIMER_PRESCALER    8
#define QTIMER_PCS_VALUE    11   // PCS=1011 = IPG/8 in QTimer register

// Hardware input filter (FILT register)
// Rejects any edge glitch shorter than FILT_PER * FILT_CNT IPG clocks.
// FILT_PER=20 (133ns per sample), FILT_CNT=3 (5 samples) → 667ns filter
// This eliminates all MKS 24V capacitive crosstalk while easily passing
// the MT6816's 2.0us minimum LOW time.
#define QTIMER_FILT_PER     20
#define QTIMER_FILT_CNT     3

const float ENCODER_OFFSETS[NUM_MOTORS] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

const int ENCODER_DIR_SIGN[NUM_MOTORS] = { 1, 1, 1, 1, -1, 1 };

// Encoder smoothing filters
#define ENCODER_MEDIAN_FILTER true
#define ENCODER_EMA_ENABLED true
#define ENCODER_EMA_ALPHA 0.3f

// ============================================================================
// CONTROL PARAMETERS
// ============================================================================

#define CONTROL_FREQUENCY_HZ 500
#define CONTROL_PERIOD_US (1000000 / CONTROL_FREQUENCY_HZ)

#define FEEDBACK_FREQUENCY_HZ 50
#define FEEDBACK_PERIOD_US (1000000 / FEEDBACK_FREQUENCY_HZ)

const float Kp[NUM_MOTORS] = { 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
const float Kd[NUM_MOTORS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

const float MAX_JOINT_VELOCITIES[NUM_MOTORS] = { 3.0, 3.0, 6.0, 6.0, 6.0, 6.0 };

#define VELOCITY_DEADBAND 0.02f
#define POSITION_ERROR_LIMIT 0.5

// ============================================================================
// COMMUNICATION CONFIGURATION
// ============================================================================

#define SERIAL_BAUD 115200
#define COMMAND_BUFFER_SIZE 256

// ============================================================================
// TIMING CONSTRAINTS
// ============================================================================

#define MAX_JITTER_US 50
#define MIN_STEP_PULSE_WIDTH_US 5
#define MAX_STEP_FREQUENCY_HZ 20000
#define PWM_RESOLUTION 8

#endif // CONFIG_H
