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

static volatile uint32_t enc_rise_cycles[NUM_MOTORS] = {0};
static volatile uint32_t enc_pw_cycles[NUM_MOTORS]   = {0};
static volatile bool     enc_new_data[NUM_MOTORS]    = {false};
static volatile bool     enc_is_high[NUM_MOTORS]     = {false}; // Topological state lock

// High Resolution Median Filter ISR (Cycle Accurate)
// Zero-delay state machine perfectly tracks the true 2us MT6816 minimum low time.
#define MAKE_ENC_ISR(N)                                            \
  void enc_isr_##N() {                                             \
    uint32_t c = ARM_DWT_CYCCNT;                                   \
    if (digitalReadFast(ENCODER_PINS[N])) {                        \
      /* 500ns verification completely squashes MKS inductive ringing */                                      \
      if (digitalReadFast(ENCODER_PINS[N])) {                      \
        /* Topological Lockout: Real rising edges happen ONLY ~1029us apart */ \
        /* We unlock at 900us (540,000 cycles at 600MHz) */        \
        if (c - enc_rise_cycles[N] > 540000) {                     \
          enc_rise_cycles[N] = c;                                  \
          enc_is_high[N] = true;                                   \
        }                                                          \
      }                                                            \
    } else {                                                       \
      /* 500ns verification safely inside the 2.0us minimum window */\
      delayNanoseconds(500);                                       \
      if (!digitalReadFast(ENCODER_PINS[N])) {                     \
        if (enc_is_high[N]) {                                      \
          uint32_t pw_cycles = c - enc_rise_cycles[N];             \
          /* Valid MT6816 high time: 2us (1200c) to 1050us (630000c) */ \
          /* Relaxed to 1100c (1.8us) to prevent 359-degree jitter drops */ \
          if (pw_cycles >= 1100 && pw_cycles <= 640000) {          \
              enc_pw_cycles[N] = pw_cycles;                        \
              enc_new_data[N] = true;                              \
          }                                                        \
          enc_is_high[N] = false;                                  \
        }                                                          \
      }                                                            \
    }                                                              \
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
  // MUST ENABLE CYCLE COUNTER for nano-second ISR logic!
  ARM_DEMCR |= ARM_DEMCR_TRCENA;
  ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    joints[i].desired_position  = 0.0f;
    joints[i].desired_velocity  = 0.0f;
    joints[i].actual_position   = 0.0f;
    joints[i].actual_velocity   = 0.0f;
    joints[i].position_error    = 0.0f;
    joints[i].velocity_command  = 0.0f;
    joints[i].total_motor_angle = 0.0f;
    joints[i].last_motor_angle  = -1.0f;  // sentinel: skips first reading

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
  uint32_t raw_pw_cycles;
  bool has_new_data;
  noInterrupts();
  raw_pw_cycles = enc_pw_cycles[idx];
  has_new_data = enc_new_data[idx];
  enc_new_data[idx] = false;
  interrupts();

  if (raw_pw_cycles == 0) return joints[idx].actual_position;  // no data yet

  // Convert cycles to microseconds (F_CPU is typically 600,000,000, so 600 cycles = 1 us)
  float raw_pw = (float)raw_pw_cycles / (F_CPU_ACTUAL / 1000000.0f);

  // 2. Hardware Sanity Check
  // Our topological ISR guarantees that we only receive valid MT6816 pulses (edges=1).
  // Anything outside 2us to 1050us shouldn't even make it here, but handled just in case.
  if (raw_pw < 2.0f || raw_pw > 1050.0f) {
    #if ENCODER_EMA_ENABLED
      return ema_initialised[idx] ? ema_position[idx] : joints[idx].actual_position;
    #else
      return joints[idx].actual_position;
    #endif
  }

  // --- Layer 1: Median-of-3 on raw pulse width ---
  float pw;
#if ENCODER_MEDIAN_FILTER
  if (has_new_data) {
    uint8_t slot = pw_fill[idx] < 3 ? pw_fill[idx] : (pw_fill[idx] % 3);
    // Cast to uint32_t for cheap median, logic analyzer gives stable float but we scale it safely
    pw_history[idx][slot] = raw_pw_cycles;
    pw_fill[idx]++;
    if (pw_fill[idx] > 200) pw_fill[idx] = 3;  // prevent overflow, keep >= 3
  }

  if (pw_fill[idx] >= 3) {
    uint32_t median_cycles = median3_u32(pw_history[idx][0], pw_history[idx][1], pw_history[idx][2]);
    pw = (float)median_cycles / (F_CPU_ACTUAL / 1000000.0f);
  } else {
    pw = raw_pw;  // not enough samples yet
  }
#else
  pw = raw_pw;
#endif

  float clocks = pw / (ENCODER_CLOCK_PERIOD_NS / 1000.0f);
  float counts = clocks - (float)ENCODER_START_CLOCKS;
  if (counts < 0.0f)                        counts = 0.0f;
  if (counts >= (float)ENCODER_RESOLUTION)   counts = (float)(ENCODER_RESOLUTION - 1);

  float motor_ang = (counts / (float)ENCODER_RESOLUTION) * 2.0f * PI;
  motor_ang *= ENCODER_DIR_SIGN[idx];
  motor_ang += ENCODER_OFFSETS[idx];

  // Normalise [0, 2π) using fmodf to prevent infinite loops on NaN/huge values
  if (!isfinite(motor_ang)) motor_ang = 0.0f; // Reject NaN
  motor_ang = fmodf(motor_ang, 2.0f * PI);
  if (motor_ang < 0.0f) motor_ang += 2.0f * PI;

  // ---------- INITIALISATION ----------
  if (joints[idx].last_motor_angle < 0.0f) {
    // First reading —  record reference flawlessly
    joints[idx].last_motor_angle = motor_ang;
    joints[idx].total_motor_angle = motor_ang;
  } else {
    // --- VELOCITY-PREDICTIVE NYQUIST UNWRAPPER ---
    // Problem: Blind [-PI, PI) normalization assumes zero velocity.
    // When EMI corrupts a reading by >180 degrees, blind normalization
    // maps the jump as movement in the WRONG direction, permanently
    // injecting exactly ±360 degrees of error (1 full revolution).
    //
    // Fix: Normalize the RESIDUAL (measured - predicted) instead of raw ang_diff.
    // The motor's known velocity resolves which direction a jump truly represents.
    // Then reject readings where the residual is physically impossible.
    float ang_diff = motor_ang - joints[idx].last_motor_angle;

    // Predict expected movement from last frame's velocity command (in motor space)
    float expected_diff = joints[idx].velocity_command * GEAR_RATIOS[idx]
                         * ((float)CONTROL_PERIOD_US / 1000000.0f);

    // Normalize the RESIDUAL (deviation from prediction) to [-PI, PI)
    float residual = ang_diff - expected_diff;
    while (residual >  PI) residual -= 2.0f * PI;
    while (residual <= -PI) residual += 2.0f * PI;
    ang_diff = expected_diff + residual;

    // EMI REJECTION: If reading deviates >0.8 rad from prediction, it's corrupted.
    // Normal movement:     |residual| < 0.05 rad (perfect prediction match)
    // 30-frame dropout:    |residual| < 0.7 rad  (still accepted for recovery)
    // EMI false reading:   |residual| > 3.0 rad  (instantly rejected!)
    if (fabsf(residual) > 0.8f) {
      // Dead-reckon: advance tracking by predicted movement so we don't deadlock.
      // This keeps last_motor_angle close to the true physical position,
      // ensuring the NEXT valid reading always has a small residual.
      joints[idx].total_motor_angle += expected_diff;
      joints[idx].last_motor_angle += expected_diff;
      // Wrap last_motor_angle to [0, 2PI)
      joints[idx].last_motor_angle = fmodf(joints[idx].last_motor_angle, 2.0f * PI);
      if (joints[idx].last_motor_angle < 0.0f) joints[idx].last_motor_angle += 2.0f * PI;

      float dr_joint_pos = joints[idx].total_motor_angle / GEAR_RATIOS[idx];
      #if ENCODER_EMA_ENABLED
        if (ema_initialised[idx]) {
          ema_position[idx] = ENCODER_EMA_ALPHA * dr_joint_pos
                            + (1.0f - ENCODER_EMA_ALPHA) * ema_position[idx];
        }
        return ema_initialised[idx] ? ema_position[idx] : dr_joint_pos;
      #else
        return dr_joint_pos;
      #endif
    }

    // Authentic movement! Accumulate infinitely.
    joints[idx].total_motor_angle += ang_diff;
    joints[idx].last_motor_angle = motor_ang;
  }

  // The final joint position calculation:
  float raw_joint_pos = joints[idx].total_motor_angle / GEAR_RATIOS[idx];

  // --- Layer 2: EMA on final joint position ---
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
