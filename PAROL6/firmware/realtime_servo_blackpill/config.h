/*
 * PAROL6 Real-Time Servo Control — Configuration (STM32F411CE Black Pill)
 *
 * STM32Duino / Arduino IDE edition.
 *
 * Uses STM32 Timer PWM Input mode for encoder reading:
 *   Each encoder gets a DEDICATED timer with slave reset.
 *   Hardware measures both period (CCR1) and high-time (CCR2).
 *   Zero ISR load for encoder capture — the control ISR just polls registers.
 *
 *   TIM1  → J1 encoder (PA8)
 *   TIM2  → J2 encoder (PA15)
 *   TIM3  → J3 encoder (PA6)
 *   TIM4  → J4 encoder (PB6)
 *   TIM5  → J5 encoder (PA0)
 *   TIM9  → J6 encoder (PA2)
 *   TIM11 → 500 Hz control loop ISR
 *
 * NOTE: PA10 is NOT used — Black Pill DFU USB upload bug.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>
#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// STEP GENERATION TIMER (TIM10 at 40 kHz)
// ============================================================================
// TIM10 runs a phase-accumulator DDS at 40 kHz.
// Each tick: check 6 accumulators, toggle one GPIO if overflow.
// No busy-wait, no EMI burst, 25µs pulse width.
#define STEP_TIMER_FREQ  40000

// ============================================================================
// HARDWARE CONFIGURATION
// ============================================================================

#define NUM_MOTORS 6

// ============================================================================
// ENCODER PWM INPUT MODE — PIN ASSIGNMENTS
// ============================================================================
// Each encoder uses a dedicated timer in PWM Input mode:
//   CH1 captures rising edge (period)   → CCR1
//   CH2 captures falling edge (high-time) → CCR2
//   Both internally mapped to same TI1 input
//   Slave reset mode: counter resets on each rising edge
//   ICxF = 9 digital filter for ~667 ns glitch rejection
//
// The CPU never services an interrupt for encoder reading.
// The control ISR (500 Hz) simply reads CCR1/CCR2 from each timer.

// Timer instance pointers (set during init)
// Index: J1=0, J2=1, J3=2, J4=3, J5=4, J6=5

// Pin assignments for each encoder (TI1 input = CH1 pin)
//                                  J1    J2     J3    J4    J5    J6
#define ENC_PIN_J1    PA8    // TIM1_CH1, AF1
#define ENC_PIN_J2    PA15   // TIM2_CH1, AF1
#define ENC_PIN_J3    PA6    // TIM3_CH1, AF2
#define ENC_PIN_J4    PB6    // TIM4_CH1, AF2
#define ENC_PIN_J5    PA0    // TIM5_CH1, AF2  (shared with user button, OK)
#define ENC_PIN_J6    PA2    // TIM9_CH1, AF3

// Encoder enable flags
static const bool ENCODER_ENABLED[NUM_MOTORS] = {
  true,  // J1
  true,  // J2
  true,  // J3
  true,  // J4
  true,   // J5
  false    // J6
};

// ============================================================================
// STEP PULSE PINS (GPIO output, toggled from control ISR)
// ============================================================================

#define STEP_PIN_J1    PA7
#define STEP_PIN_J2    PA9
#define STEP_PIN_J3    PB0
#define STEP_PIN_J4    PB1
#define STEP_PIN_J5    PB8
#define STEP_PIN_J6    PB9

// Array for indexed access (set in motor.cpp init)
extern const uint32_t STEP_PINS[NUM_MOTORS];

// ============================================================================
// DIRECTION PINS (GPIO output)
// ============================================================================

#define DIR_PIN_J1     PB10
#define DIR_PIN_J2     PB12
#define DIR_PIN_J3     PB13
#define DIR_PIN_J4     PB14
#define DIR_PIN_J5     PB15
#define DIR_PIN_J6     PA5

extern const uint32_t DIR_PINS_ARR[NUM_MOTORS];

// ============================================================================
// HOMING CONFIGURATION
// ============================================================================
// Sensor types: 0 = limit switch (INPUT_PULLUP, FALLING edge)
//               1 = inductive proximity (INPUT_PULLUP, RISING edge)
//
// Homing direction: the direction the motor moves to reach the home sensor.
//   +1 = positive (encoder readings increasing)
//   -1 = negative (encoder readings decreasing)
//
// Offsets are in DEGREES for easy manual tuning. Converted to radians internally.
// The offset is the position value assigned to the joint AFTER homing completes.
// Example: if J4 homes at its max-range end and that corresponds to 108° in
//          MoveIt, set HOMING_OFFSET_DEG[3] = 108.0f.

#define DEG_TO_RAD(d) ((d) * (float)M_PI / 180.0f)

// Sensor pin for each joint (0 = no sensor / homing disabled)
//                                     J1    J2    J3    J4    J5    J6
static const uint32_t HOMING_SENSOR_PINS[NUM_MOTORS] = {
    PA1,  PB3,  PB5,  PA3,  PB4,  0
};

// Sensor type per joint: 0=limit_switch, 1=inductive_prox
static const uint8_t HOMING_SENSOR_TYPE[NUM_MOTORS] = { 1, 0, 0, 1, 0, 0 };

// Per-joint homing enable
static const bool HOMING_ENABLED[NUM_MOTORS] = {
    false,   // J1 — inductive sensor
    true,   // J2 — limit switch (may be resting on it)
    true,   // J3 — limit switch
    false,   // J4 — inductive sensor
    true,   // J5 — limit switch
    false   // J6 — no sensor yet
};

// Homing operation mode: true = one by one, false = all simultaneously
static const bool HOMING_SEQUENTIAL = true;

// Homing direction: +1 = increasing, -1 = decreasing
static const int HOMING_DIR[NUM_MOTORS] = { -1, -1, +1, -1, +1, 0 };

// Homing speed (rad/s at joint level)
static const float HOMING_SPEED[NUM_MOTORS] = {
    0.5f, 0.2f, 0.3f, 0.5f, 0.5f, 0.0f
};

// Position offset assigned after homing (DEGREES — easy to tune)
// This is what the joint position is set to when the sensor is triggered.
static const float HOMING_OFFSET_DEG[NUM_MOTORS] = {
    0.0f,    // J1
    -56.0f,  // J2
    74.0f,   // J3
    108.0f,  // J4 — homes at ~108° in MoveIt range
    120.0f,  // J5
    0.0f     // J6
};

#define HOMING_NO_READY 999.0f

// Optional: Go to a specific angle (DEGREES) after a joint finishes homing.
// If set to HOMING_NO_READY, the joint stays at its sensor triggered position.
// Useful for moving a joint out of the way before the next joint homes.
static const float HOMING_READY_POS_DEG[NUM_MOTORS] = {
    HOMING_NO_READY, // J1
    0.0f,            // J2 (moves to 0.0 deg after arriving at -56.0 deg)
    0.0f, // J3
    HOMING_NO_READY, // J4
    0.0f, // J5
    HOMING_NO_READY  // J6
};

// Homing timeout (ms)
#define HOMING_TIMEOUT_MS     15000

// Back-off speed: fraction of homing speed when backing away from a
// sensor that's already triggered at startup (e.g. J2 resting on switch)
#define HOMING_BACKOFF_SPEED_MULT  0.5f

// Settle time after sensor triggers before zeroing (ms)
#define HOMING_SETTLE_MS      100

// ============================================================================
// MOTOR CONFIGURATION
// ============================================================================

#define STEPS_PER_REV 200

static const int MICROSTEPS[NUM_MOTORS] = { 4, 16, 16, 16, 16, 16 };

static const float GEAR_RATIOS[NUM_MOTORS] = {
  6.4f,  // J1: 20:1 gearbox
  20.0f,  // J2
  18.0952381f,   // J3
  4.0f,   // J4
  4.0f,   // J5
  10.0f   // J6
};

static const int MOTOR_DIR_SIGN[NUM_MOTORS] = { 1, 1, -1, 1, 1, -1 }; //J1 , J2 , J3 Negative, J4 Positive, J5 Positive, J6 Negative

// ============================================================================
// ENCODER CONFIGURATION
// ============================================================================

// MT6816 PWM encoding parameters
#define ENCODER_CLOCK_PERIOD_NS   250.0f
#define ENCODER_START_CLOCKS      16
#define ENCODER_RESOLUTION        4096

// ---- PWM Input mode timer configuration ----
// SYSCLK = 96 MHz, APB1 timers = 96 MHz, APB2 timers = 96 MHz
// Timer prescaler = 7 (divides by 8) → 12 MHz capture clock → 83.33 ns/tick
// 16-bit timers: overflow at 65536 ticks = 5461 µs (>> MT6816 max 1028 µs)
// 32-bit timers (TIM2, TIM5): effectively infinite
//
// Digital filter: ICxF = 9 → fSAMPLING = fDTS/8, N=8
//   At 12 MHz: 8 samples × 83.33 ns = 667 ns glitch rejection
//   EXACTLY matches Teensy QTimer FILT (667 ns)
#define ENC_TIM_PRESCALER   7       // PSC value (divides by 8)
#define ENC_TIM_FILTER      9       // ICxF value
#define ENC_TIM_CLOCK_HZ    12000000  // 96 MHz / 8

// Tick → microseconds conversion
#define TICKS_TO_US  (1.0f / (float)ENC_TIM_CLOCK_HZ * 1000000.0f)

static const float ENCODER_OFFSETS[NUM_MOTORS] = { 0.0f, 0.0f, 0.0f, -1.920f, 1.0f, 0.0f };
static const int ENCODER_DIR_SIGN[NUM_MOTORS] = { 1, 1, 1, 1, 1, 1 };

// Encoder smoothing
#define ENCODER_MEDIAN_FILTER   1
#define ENCODER_EMA_ENABLED     1
#define ENCODER_EMA_ALPHA       0.3f

// ============================================================================
// CONTROL PARAMETERS
// ============================================================================

#define CONTROL_FREQUENCY_HZ  500
#define CONTROL_PERIOD_US     (1000000 / CONTROL_FREQUENCY_HZ)

#define FEEDBACK_FREQUENCY_HZ 50
#define FEEDBACK_PERIOD_MS    (1000 / FEEDBACK_FREQUENCY_HZ)

static const float Kp[NUM_MOTORS] = { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f };
static const float Kd[NUM_MOTORS] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float MAX_JOINT_VELOCITIES[NUM_MOTORS] = { 3.0f, 3.0f, 1.0f, 6.0f, 6.0f, 6.0f };

#define VELOCITY_DEADBAND       0.02f
#define POSITION_ERROR_LIMIT    0.5f

// ============================================================================
// COMMUNICATION
// ============================================================================

#define COMMAND_BUFFER_SIZE 256

// ============================================================================
// STEP PULSE TIMING
// ============================================================================

#define MIN_STEP_PULSE_WIDTH_US   5
#define MAX_STEP_FREQUENCY_HZ     20000

// ============================================================================
// MATH CONSTANTS
// ============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#define TWO_PI (2.0f * M_PI)

#endif // CONFIG_H
