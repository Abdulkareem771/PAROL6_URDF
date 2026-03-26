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
static volatile bool     enc_new_data[NUM_MOTORS] = {false};

// Per-motor ISRs (minimal work — just capture timing)
#define MAKE_ENC_ISR(N)                                          \
  void enc_isr_##N() {                                           \
    if (digitalReadFast(ENCODER_PINS[N])) enc_rise[N] = micros();\
    else { enc_pw[N] = micros() - enc_rise[N]; enc_new_data[N] = true; } \
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
// ENCODER SMOOTHING FILTERS
// ============================================================================

// Median-of-3 for raw pulse width (rejects PWM timing outliers)
static uint32_t pw_history[NUM_MOTORS][3] = {{0}};
static uint8_t  pw_fill[NUM_MOTORS] = {0};  // samples collected (0-3)

static uint32_t median3_u32(uint32_t a, uint32_t b, uint32_t c) {
  if (a > b) { uint32_t t = a; a = b; b = t; }
  if (b > c) { uint32_t t = b; b = c; c = t; }
  if (a > b) { uint32_t t = a; a = b; b = t; }
  return b;
}

// EMA state for final position
static float ema_position[NUM_MOTORS] = {0};
static bool  ema_initialised[NUM_MOTORS] = {false};

// ============================================================================
// JOINT STATE
// ============================================================================

static JointState joints[NUM_MOTORS];

// ============================================================================
// SAFETY INTERLOCK
// ============================================================================
static volatile bool armed = false;

void controlArm() {
  if (armed) return;
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    joints[i].desired_position = joints[i].actual_position;
    joints[i].desired_velocity = 0.0f;
  }
  armed = true;
}

bool controlIsArmed() {
  return armed;
}

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
  // 1. Grab values atomically
  uint32_t raw_pw;
  bool has_new_data;
  noInterrupts();
  raw_pw = enc_pw[idx];
  has_new_data = enc_new_data[idx];
  enc_new_data[idx] = false;
  interrupts();

  if (raw_pw == 0) return joints[idx].actual_position;  // no data yet

  // 2. Hardware Sanity Check
  // MT6816 limits: 16 clocks (4us) to 4111 clocks (1027.75us).
  // Anything outside 2us to 1050us is physically impossible (noise/bounce/preemption glitch).
  if (raw_pw < 2 || raw_pw > 1050) {
    // Return last known good position; do not corrupt multi-turn tracking!
    #if ENCODER_EMA_ENABLED
      return ema_initialised[idx] ? ema_position[idx] : joints[idx].actual_position;
    #else
      return joints[idx].actual_position;
    #endif
  }

  // --- Layer 1: Median-of-3 on raw pulse width ---
  // Catches single-sample PWM timing glitches with zero lag for consistent readings
  uint32_t pw;
#if ENCODER_MEDIAN_FILTER
  if (has_new_data) {
    uint8_t slot = pw_fill[idx] < 3 ? pw_fill[idx] : (pw_fill[idx] % 3);
    pw_history[idx][slot] = raw_pw;
    pw_fill[idx]++;
    if (pw_fill[idx] > 200) pw_fill[idx] = 3;  // prevent overflow, keep >= 3
  }

  if (pw_fill[idx] >= 3) {
    pw = median3_u32(pw_history[idx][0], pw_history[idx][1], pw_history[idx][2]);
  } else {
    pw = raw_pw;  // not enough samples yet
  }
#else
  pw = raw_pw;
#endif

  float clocks = (float)pw / (ENCODER_CLOCK_PERIOD_NS / 1000.0f);
  float counts = clocks - (float)ENCODER_START_CLOCKS;
  if (counts < 0.0f)                        counts = 0.0f;
  if (counts >= (float)ENCODER_RESOLUTION)   counts = (float)(ENCODER_RESOLUTION - 1);

  float motor_ang = (counts / (float)ENCODER_RESOLUTION) * 2.0f * PI;
  motor_ang *= ENCODER_DIR_SIGN[idx];
  motor_ang += ENCODER_OFFSETS[idx];

  // Normalise [0, 2π)
  while (motor_ang < 0.0f)        motor_ang += 2.0f * PI;
  while (motor_ang >= 2.0f * PI)  motor_ang -= 2.0f * PI;

  // ---------- MULTI-TURN TRACKING ----------
  if (joints[idx].last_motor_angle < 0.0f) {
    // First reading —  record reference
    joints[idx].last_motor_angle = motor_ang;
    // Don't return 0.0f here, let it pass through to initialize EMA!
  }

  float delta = motor_ang - joints[idx].last_motor_angle;
  if (delta >  PI) joints[idx].motor_revolutions--;
  if (delta < -PI) joints[idx].motor_revolutions++;
  joints[idx].last_motor_angle = motor_ang;

  float total = motor_ang + (float)joints[idx].motor_revolutions * 2.0f * PI;
  float raw_joint_pos = total / GEAR_RATIOS[idx];

  // --- Layer 2: EMA on final joint position ---
  // Smooths encoder quantization noise (4096 steps/rev → 0.088° steps)
#if ENCODER_EMA_ENABLED
  if (!ema_initialised[idx]) {
    ema_position[idx] = raw_joint_pos;
    ema_initialised[idx] = true;
  } else {
    ema_position[idx] = ENCODER_EMA_ALPHA * raw_joint_pos
                      + (1.0f - ENCODER_EMA_ALPHA) * ema_position[idx];
  }
  return ema_position[idx];
#else
  return raw_joint_pos;
#endif
}

// ============================================================================
// CONTROL LOOP  (called from IntervalTimer at 500 Hz)
// ============================================================================

void controlUpdate() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    JointState* j = &joints[i];

    // 1. Actual position (always read — feedback works even when disarmed)
    j->actual_position = readEncoder(i);

    // SAFETY: Skip motor output if not armed
    if (!armed) {
      j->velocity_command = 0.0f;
      j->position_error   = 0.0f;
      motorSetFrequency(i, 0.0f);
      continue;
    }

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
