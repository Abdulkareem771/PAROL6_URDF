/*
 * PAROL6 Real-Time Servo Control — Configuration (Teensy 4.1)
 *
 * Pin assignments, constants, and control parameters.
 * Teensy 4.1 (i.MXRT1062, 600 MHz ARM Cortex-M7):
 *   - 55 GPIO, all bidirectional, all interrupt-capable
 *   - 4 FlexPWM modules × 4 submodules = 16 independent frequency groups
 *   - 8 hardware UART ports
 *   - Native USB 2.0 High Speed (480 Mbps)
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================

#define NUM_MOTORS 6

// Step pins — each on a DIFFERENT FlexPWM submodule for independent frequency
//   Pin  4 → FlexPWM2.0    Pin  8 → FlexPWM1.3
//   Pin 33 → FlexPWM2.0    Pin  9 → FlexPWM2.2   ... etc.
//
// We select 6 pins across 6 different submodules:
const int STEP_PINS[NUM_MOTORS] = {
  2,   // J1 → FlexPWM4.2A
  4,   // J2 → FlexPWM2.2A
  5,   // J3 → FlexPWM1.3B
  8,   // J4 → FlexPWM1.3A
  7,   // J5 → FlexPWM2.0A
  6    // J6 → FlexPWM2.1A
};

// Direction pins — simple digital outputs
const int DIR_PINS[NUM_MOTORS] = {
  24,  // J1
  35,  // J2
  40,  // J3
  27,  // J4
  34,  // J5
  28   // J6
};

// Encoder PWM input pins (MT6816) — all support interrupts on Teensy
const int ENCODER_PINS[NUM_MOTORS] = {
  14,  // J1
  12,  // J2
  11,  // J3
  17,  // J4
  19,  // J5
  18   // J6
};

// Encoder enable flags (set true when encoder is physically connected)
const bool ENCODER_ENABLED[NUM_MOTORS] = {
  false,   // J1: Encoder connected
  false,   // J2: Encoder connected
  false,  // J3: Not connected yet
  false,  // J4: Not connected yet
  true,  // J5: Not connected yet
  false   // J6: Not connected yetparol6-ultimate.tar
};

// ============================================================================
// INDUCTIVE PROXIMITY SENSORS (homing)
// ============================================================================

#define NUM_PROX_SENSORS 3

// Proximity sensor pins — connected via optocouplers
// Optocoupler conducts → pin pulled to GND → read LOW = sensor triggered
// Configured as INPUT_PULLUP (HIGH when sensor not triggered)
const int PROX_PINS[NUM_PROX_SENSORS] = {
  20,  // Sensor 1
  21,  // Sensor 2
  22   // Sensor 3
};

// ============================================================================
// MOTOR CONFIGURATION
// ============================================================================

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
  1.0,      // J1: 20:1 gearbox
  20.0,     // J2: Direct drive
  16.5,  // J3: Direct drive
  4.0,      // J4: Direct drive
  10.0,      // J5: Direct drive
  10.0      // J6: Direct drive
};

// Motor direction sign (+1 or -1)
// If encoder shows position going OPPOSITE to motor step direction, set to -1
const int MOTOR_DIR_SIGN[NUM_MOTORS] = {
  1,   // J1: Normal
  -1,  // J2: Inverted (encoder reads opposite to step direction)
  -1,   // J3: TBD
  1,   // J4: TBD
  -1,   // J5: TBD
  1    // J6: TBD
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
  0.0,  // J1: Calibrated offset
  0.0,  // J2: TBD
  0.0,  // J3: TBD
  0.0,  // J4: TBD
  0.0,  // J5: TBD
  0.0   // J6: TBD
};

// Encoder direction sign (+1 or -1)
// If encoder position decreases when physical shaft moves in positive joint direction, set to -1
const int ENCODER_DIR_SIGN[NUM_MOTORS] = {
  1,   // J1: Normal
  1,   // J2: TBD
  1,   // J3: TBD
  1,   // J4: TBD
  -1,  // J5: Inverted to track motor direction inversion
  1    // J6: TBD
};

// Encoder smoothing filters
// Median filter: rejects single-sample PWM timing glitches (zero added lag)
#define ENCODER_MEDIAN_FILTER true
// EMA filter: smooths encoder quantization noise
// At 500 Hz, alpha=0.3 → time constant ~6ms (fast enough for control)
#define ENCODER_EMA_ENABLED true
#define ENCODER_EMA_ALPHA 0.3f

// ============================================================================
// CONTROL PARAMETERS
// ============================================================================

// Control loop frequency
#define CONTROL_FREQUENCY_HZ 500
#define CONTROL_PERIOD_US (1000000 / CONTROL_FREQUENCY_HZ)  // 2000 µs

// Feedback rate
#define FEEDBACK_FREQUENCY_HZ 50
#define FEEDBACK_PERIOD_US (1000000 / FEEDBACK_FREQUENCY_HZ) // 20000 µs

// Servo gains (position + velocity feedforward)
const float Kp[NUM_MOTORS] = {
  1.0,  // J1: Low Kp to prevent overshoot/stall
  1.0,  // J2: Low Kp to prevent overshoot/stall
  1.0,  // J3
  2.0,  // J4
  2.0,  // J5
  2.0   // J6
};

// Velocity derivative gain (set to 0 initially)
const float Kd[NUM_MOTORS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

// Maximum joint velocities (rad/s) - from joint_limits.yaml
const float MAX_JOINT_VELOCITIES[NUM_MOTORS] = {
  0.5,  // J1: Conservative — keeps motor below stall speed
  0.5,  // J2: 0.5 rad/s × 20 gear = 10 rad/s motor = ~5000 Hz steps
  0.5,  // J3: Conservative
  1.0,  // J4
  1.2,  // J5
  1.0   // J6
};

// Velocity deadband — suppress encoder noise jitter near target
#define VELOCITY_DEADBAND 0.02f  // rad/s

// Safety position error limit (radians)
#define POSITION_ERROR_LIMIT 0.5

// ============================================================================
// COMMUNICATION CONFIGURATION
// ============================================================================

// Teensy 4.1 uses native USB (480 Mbps) — no baud rate limitation.
// Serial.begin() is ignored for USB; it always runs at full speed.
// We keep a nominal value for compatibility.
#define SERIAL_BAUD 115200
#define COMMAND_BUFFER_SIZE 256

// ============================================================================
// TIMING CONSTRAINTS
// ============================================================================

// Maximum acceptable control loop jitter (microseconds)
#define MAX_JITTER_US 50  // Teensy is much more deterministic than ESP32

// Minimum step pulse width (microseconds) - for MKS SERVO42C
#define MIN_STEP_PULSE_WIDTH_US 5

// Maximum step frequency (Hz) - 20kHz = 50µs minimum period
#define MAX_STEP_FREQUENCY_HZ 20000

// PWM resolution for step generation (8-bit = 256 levels)
#define PWM_RESOLUTION 8

#endif // CONFIG_H
