/*
 * PAROL6 Control Implementation
 *
 * Position servo with velocity feedforward.
 * Control law (per joint, @ 500 Hz):
 *   velocity_command = desired_velocity + Kp × (desired_pos − actual_pos)
 *
 * Encoder: MT6816 PWM decoding via GPIO interrupt.
 * Non-encoder motors: velocity integration for position estimate.
 */

#include "control.h"
#include "motor.h"

// ============================================================================
// ENCODER PWM CAPTURE
// ============================================================================

static volatile uint32_t enc_rise[NUM_MOTORS] = {0};
static volatile uint32_t enc_pw[NUM_MOTORS]   = {0};

// Per-motor ISRs (IRAM, minimal work)
#define MAKE_ENC_ISR(N)                                      \
  void IRAM_ATTR enc_isr_##N() {                             \
    if (digitalRead(ENCODER_PINS[N])) enc_rise[N] = micros();\
    else enc_pw[N] = micros() - enc_rise[N];                 \
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
    joints[i].last_motor_angle  = 0.0f;

    if (ENCODER_ENABLED[i]) {
      pinMode(ENCODER_PINS[i], INPUT);
      attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[i]),
                      enc_isrs[i], CHANGE);
    }
  }
}

// ============================================================================
// COMMAND UPDATE  (from serial task — atomic float writes)
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
    // Integrate velocity_command to estimate position
    // dt = CONTROL_PERIOD_MS / 1000
    joints[idx].actual_position +=
        joints[idx].velocity_command * ((float)CONTROL_PERIOD_MS / 1000.0f);
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

  // Multi-turn tracking
  float delta = motor_ang - joints[idx].last_motor_angle;
  if (delta >  PI) joints[idx].motor_revolutions--;
  if (delta < -PI) joints[idx].motor_revolutions++;
  joints[idx].last_motor_angle = motor_ang;

  float total = motor_ang + (float)joints[idx].motor_revolutions * 2.0f * PI;
  return total / GEAR_RATIOS[idx];
}

// ============================================================================
// CONTROL LOOP  (500 Hz FreeRTOS task)
// ============================================================================

void controlUpdate() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    JointState* j = &joints[i];

    // 1. Actual position
    j->actual_position = readEncoder(i);

    // 2. Position error
    j->position_error = j->desired_position - j->actual_position;

    // 3. Control law: velocity feedforward + position correction
    //    Large errors just produce large velocity → clamped below
    j->velocity_command = j->desired_velocity
                        + Kp[i] * j->position_error;

    // 4. Clamp velocity (THIS is the real safety limit)
    if (j->velocity_command >  MAX_JOINT_VELOCITIES[i])
      j->velocity_command =  MAX_JOINT_VELOCITIES[i];
    if (j->velocity_command < -MAX_JOINT_VELOCITIES[i])
      j->velocity_command = -MAX_JOINT_VELOCITIES[i];

    // 6. Direction
    motorSetDirection(i, j->velocity_command >= 0.0f);

    // 7. Step frequency (positive)
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
