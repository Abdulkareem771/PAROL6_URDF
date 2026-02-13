/*
 * PAROL6 Real-Time Servo Control - Configuration
 * 
 * Pin assignments, constants, and control parameters
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================

#define NUM_MOTORS 6

// Step/Dir pins per motor
const int STEP_PINS[NUM_MOTORS] = {5, 25, 14, 12, 13, 15};
const int DIR_PINS[NUM_MOTORS] = {2, 26, 27, 4, 16, 17};

// Encoder PWM input pins (MT6816)
const int ENCODER_PINS[NUM_MOTORS] = {27, 33, 25, 26, 34, 35};

// Encoder enable flags (set true when encoder is physically connected)
const bool ENCODER_ENABLED[NUM_MOTORS] = {
  true,   // J1: Encoder connected
  false,  // J2: Not connected yet
  false,  // J3: Not connected yet
  false,  // J4: Not connected yet
  false,  // J5: Not connected yet
  false   // J6: Not connected yet
};

// Motor configuration
#define STEPS_PER_REV 200

// Microstepping configuration per motor
const int MICROSTEPS[NUM_MOTORS] = {
  4,   // J1: 20:1 gearbox - low microsteps for speed
  16,  // J2: Direct drive - high precision
  16,  // J3: Direct drive - high precision
  16,  // J4: Direct drive - high precision
  16,  // J5: Direct drive - high precision
  16   // J6: Direct drive - high precision
};

// Gearbox ratios (motor revolutions per joint revolution)
const float GEAR_RATIOS[NUM_MOTORS] = {
  20.0,  // J1: 20:1 gearbox
  1.0,   // J2: Direct drive
  1.0,   // J3: Direct drive
  1.0,   // J4: Direct drive
  1.0,   // J5: Direct drive
  1.0    // J6: Direct drive
};

// ============================================================================
// ENCODER CONFIGURATION
// ============================================================================

// MT6816 PWM encoding
const float ENCODER_CLOCK_PERIOD_NS = 250.0;  // 250ns clock period
const int ENCODER_START_CLOCKS = 16;           // 16-clock start pattern
const int ENCODER_RESOLUTION = 4096;           // 12-bit = 4096 positions

// Encoder zero offsets (radians at motor shaft, measured during calibration)
const float ENCODER_OFFSETS[NUM_MOTORS] = {
  -5.117,  // J1: Calibrated offset
  0.0,     // J2-J6: TBD
  0.0,
  0.0,
  0.0,
  0.0
};

// ============================================================================
// CONTROL PARAMETERS
// ============================================================================

// Control loop frequency
#define CONTROL_FREQUENCY_HZ 500
#define CONTROL_PERIOD_MS (1000 / CONTROL_FREQUENCY_HZ)

// Feedback rate
#define FEEDBACK_FREQUENCY_HZ 50
#define FEEDBACK_PERIOD_MS (1000 / FEEDBACK_FREQUENCY_HZ)

// Servo gains (position + velocity feedforward)
const float Kp[NUM_MOTORS] = {
  2.0,  // J1
  2.0,  // J2
  2.0,  // J3
  2.0,  // J4
  2.0,  // J5
  2.0   // J6
};

// Velocity derivative gain (set to 0 initially - no velocity feedback term)
const float Kd[NUM_MOTORS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

// Maximum joint velocities (rad/s) - from joint_limits.yaml
const float MAX_JOINT_VELOCITIES[NUM_MOTORS] = {
  3.0,  // J1: Conservative for geared joint
  4.0,  // J2
  6.0,  // J3
  6.0,  // J4
  6.0,  // J5
  6.0   // J6
};

// Safety position error limit (radians)
#define POSITION_ERROR_LIMIT 0.5  // Trigger fault if error exceeds this

// ============================================================================
// COMMUNICATION CONFIGURATION
// ============================================================================

#define SERIAL_BAUD 115200
#define COMMAND_BUFFER_SIZE 256

// ============================================================================
// FREERTOS CONFIGURATION
// ============================================================================

// Task priorities (higher = more important)
#define CONTROL_TASK_PRIORITY 3      // Highest (after ISR)
#define SERIAL_TASK_PRIORITY 2        // Lower than control

// Stack sizes (bytes)
#define CONTROL_TASK_STACK_SIZE 4096
#define SERIAL_TASK_STACK_SIZE 4096

// ============================================================================
// TIMING CONSTRAINTS
// ============================================================================

// Maximum acceptable control loop jitter (microseconds)
#define MAX_JITTER_US 200

// Minimum step pulse width (microseconds) - for MKS SERVO42C
#define MIN_STEP_PULSE_WIDTH_US 5

// Maximum step frequency (Hz) - 20kHz = 50μs minimum period
#define MAX_STEP_FREQUENCY_HZ 20000

#endif // CONFIG_H
