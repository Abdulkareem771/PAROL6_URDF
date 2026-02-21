/*
 * PAROL6 Control Implementation (Teensy 4.1)
 *
 * Position servo with velocity feedforward.
 * Control law (per joint, @ 500 Hz):
 *   velocity_command = desired_velocity + Kp × (desired_pos − actual_pos)
 *
 * Encoder: MT6816 PWM decoding via GPIO interrupt.
 * Non-encoder motors: velocity integration for position estimate.
 *
 * Teensy differences from ESP32:
 *   - No IRAM_ATTR needed for ISRs
 *   - All pins support attachInterrupt() directly
 *   - IntervalTimer calls controlUpdate() (not FreeRTOS)
 *   - 600 MHz = more headroom for float math in ISR
 */

#include "control.h"
#include "motor.h"

// ============================================================================
// ENCODER PWM CAPTURE
// ============================================================================

static volatile uint32_t enc_rise[NUM_MOTORS] = {0};
static volatile uint32_t enc_pw[NUM_MOTORS]   = {0};

// Per-motor ISRs (minimal work — just capture timing)
#define MAKE_ENC_ISR(N)                                          \
  void enc_isr_##N() {                                           \
    if (digitalReadFast(ENCODER_PINS[N])) enc_rise[N] = micros();\
    else enc_pw[N] = micros() - enc_rise[N];                     \
  }
MAKE_ENC_ISR(0)
MAKE_ENC_ISR(1)
MAKE_ENC_ISR(2)
MAKE_ENC_ISR(3)
MAKE_ENC_ISR(4)
MAKE_ENC_ISR(5)

static void (*enc_isrs[NUM_MOTORS])() = {
  enc_isr_0, enc_isr_1, enc_isr_2,
  enc_isr_3, enc_isr_4, enc_isr_5
};

// ============================================================================
// JOINT STATE
// ============================================================================

static JointState joints[NUM_MOTORS];

// ============================================================================
// INIT
// ============================================================================

void controlInit() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    joints[i].desired_position  = 0.0f;
    joints[i].desired_velocity  = 0.0f;
    joints[i].actual_position   = 0.0f;
    joints[i].actual_velocity   = 0.0f;
    joints[i].position_error    = 0.0f;
    joints[i].velocity_command  = 0.0f;
    joints[i].motor_revolutions = 0;
    joints[i].last_motor_angle  = -1.0f;  // sentinel: anti-glitch filter skips first reading

    if (ENCODER_ENABLED[i]) {
      pinMode(ENCODER_PINS[i], INPUT);
      attachInterrupt(ENCODER_PINS[i], enc_isrs[i], CHANGE);
    }
  }
}

// ============================================================================
// COMMAND UPDATE  (from main loop — atomic float writes on Cortex-M7)
// ============================================================================

void controlSetCommand(uint8_t idx, float pos, float vel) {
  if (idx >= NUM_MOTORS) return;
  joints[idx].desired_position = pos;
  joints[idx].desired_velocity = vel;
}

// ============================================================================
// ENCODER POSITION READING
// ============================================================================

static float readEncoder(uint8_t idx) {
  // ---------- NON-ENCODER: velocity integration ----------
  if (!ENCODER_ENABLED[idx]) {
    joints[idx].actual_position +=
        joints[idx].velocity_command * ((float)CONTROL_PERIOD_US / 1000000.0f);
    return joints[idx].actual_position;
  }

  // ---------- MT6816 PWM decode ----------
  uint32_t pw = enc_pw[idx];
  if (pw == 0) return joints[idx].actual_position;  // no data yet

  float clocks = (float)pw / (ENCODER_CLOCK_PERIOD_NS / 1000.0f);
  float counts = clocks - (float)ENCODER_START_CLOCKS;
  if (counts < 0.0f)                        counts = 0.0f;
  if (counts >= (float)ENCODER_RESOLUTION)   counts = (float)(ENCODER_RESOLUTION - 1);

  float motor_ang = (counts / (float)ENCODER_RESOLUTION) * 2.0f * PI;
  motor_ang += ENCODER_OFFSETS[idx];

  // Normalise [0, 2π)
  while (motor_ang < 0.0f)        motor_ang += 2.0f * PI;
  while (motor_ang >= 2.0f * PI)  motor_ang -= 2.0f * PI;

  // ---- ANTI-GLITCH FILTER ----
  // Reject readings where the motor angle changed more than physically
  // possible.  Prevents noisy PWM captures from corrupting the multi-turn
  // counter or causing velocity spikes.
  if (joints[idx].last_motor_angle >= 0.0f) {       // skip first reading
    float delta = motor_ang - joints[idx].last_motor_angle;
    // Unwrap to [-π, +π] so boundary crossings look small
    if (delta >  PI) delta -= 2.0f * PI;
    if (delta < -PI) delta += 2.0f * PI;

    // Maximum physically possible motor angle change per control tick
    float max_change = MAX_JOINT_VELOCITIES[idx] * GEAR_RATIOS[idx]
                     * ((float)CONTROL_PERIOD_US / 1000000.0f) * 5.0f;
    if (max_change < 0.3f) max_change = 0.3f;       // min 0.3 rad threshold

    if (fabsf(delta) > max_change) {
      // Glitch! Return previous position, do NOT update last_motor_angle
      return joints[idx].actual_position;
    }
  }

  // ---------- MULTI-TURN TRACKING (all encoder-enabled motors) ----------
  // Track cumulative angle including full revolutions.
  // Works for both geared and direct-drive motors.
  // The anti-glitch filter above protects against false revolution counts.
  float delta = motor_ang - joints[idx].last_motor_angle;
  if (delta >  PI) joints[idx].motor_revolutions--;
  if (delta < -PI) joints[idx].motor_revolutions++;
  joints[idx].last_motor_angle = motor_ang;

  float total = motor_ang + (float)joints[idx].motor_revolutions * 2.0f * PI;
  return total / GEAR_RATIOS[idx];
}

// ============================================================================
// CONTROL LOOP  (called from IntervalTimer at 500 Hz)
// ============================================================================

void controlUpdate() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    JointState* j = &joints[i];

    // 1. Actual position
    j->actual_position = readEncoder(i);

    // 2. Position error
    j->position_error = j->desired_position - j->actual_position;

    // 3. Control law: velocity feedforward + position correction
    j->velocity_command = j->desired_velocity
                        + Kp[i] * j->position_error;

    // 4. Clamp velocity (safety limit)
    if (j->velocity_command >  MAX_JOINT_VELOCITIES[i])
      j->velocity_command =  MAX_JOINT_VELOCITIES[i];
    if (j->velocity_command < -MAX_JOINT_VELOCITIES[i])
      j->velocity_command = -MAX_JOINT_VELOCITIES[i];

    // 4b. Deadband — suppress encoder noise jitter when near target
    if (fabsf(j->velocity_command) < VELOCITY_DEADBAND)
      j->velocity_command = 0.0f;

    // 5. Direction (apply motor direction sign to match encoder polarity)
    bool forward = (j->velocity_command * MOTOR_DIR_SIGN[i]) >= 0.0f;
    motorSetDirection(i, forward);

    // 6. Step frequency (positive)
    float motor_vel     = fabsf(j->velocity_command) * GEAR_RATIOS[i];
    float steps_per_rev = (float)(STEPS_PER_REV * MICROSTEPS[i]);
    float step_freq     = (motor_vel * steps_per_rev) / (2.0f * PI);

    motorSetFrequency(i, step_freq);
  }
}

// ============================================================================
// STATE ACCESS
// ============================================================================

const JointState* controlGetState(uint8_t idx) {
  if (idx >= NUM_MOTORS) return nullptr;
  return &joints[idx];
}
