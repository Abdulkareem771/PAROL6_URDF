/*
 * PAROL6 Control — QTimer Hardware Capture Edition
 *
 * Uses i.MXRT1062 Quad Timer (QTimer) hardware input capture instead of
 * GPIO interrupts for reading MT6816 PWM encoder signals.
 *
 * KEY ADVANTAGES over GPIO interrupt approach:
 *   1. Edge timestamps are latched in HARDWARE — immune to ISR latency
 *   2. FILT register provides hardware digital noise filtering (~667ns)
 *      that eliminates MKS 24V step-pulse crosstalk at the silicon level
 *   3. No delayNanoseconds() blocking inside ISR
 *   4. No software debounce state machines needed
 *
 * QTimer modules used:
 *   TMR1: J2 (CH1, pin 12), J3 (CH0, pin 10)
 *   TMR3: J1 (CH2, pin 14), J4 (CH3, pin 15), J5 (CH0, pin 19), J6 (CH1, pin 18)
 */

#include "control.h"
#include "motor.h"

// ============================================================================
// QTIMER ENCODER CAPTURE STATE
// ============================================================================

// Pulse width captured by QTimer ISR (in timer ticks at IPG/8 = 18.75 MHz)
static volatile uint16_t enc_pw_ticks[NUM_MOTORS]    = {0};
static volatile bool     enc_new_data[NUM_MOTORS]    = {false};
static volatile uint16_t enc_rise_ticks[NUM_MOTORS]  = {0};

// References to the two QTimer modules we use
#define TMR_MOD_TMR1  0
#define TMR_MOD_TMR3  2

// ============================================================================
// QTIMER ISR — HARDWARE CAPTURE HANDLER
// ============================================================================

// Process a single QTimer channel capture event
// `enc_idx` is the motor/encoder index (0-5)
// `ch` is a reference to the QTimer channel registers
static inline void processCapture(uint8_t enc_idx, volatile IMXRT_TMR_CH_t &ch) {
  uint16_t capt = ch.CAPT;      // Reading CAPT clears the capture flag
  uint16_t sctrl = ch.SCTRL;

  // Clear Input Edge Flag (write 0 to IEF bit, w1c-style: write 0 to clear)
  ch.SCTRL = sctrl & ~TMR_SCTRL_IEF;

  // Determine edge type from current pin state (INPUT bit)
  if (sctrl & TMR_SCTRL_INPUT) {
    // Pin is currently HIGH → this was a RISING edge
    enc_rise_ticks[enc_idx] = capt;
  } else {
    // Pin is currently LOW → this was a FALLING edge
    // Unsigned subtraction handles 16-bit counter wrap perfectly
    uint16_t pw = capt - enc_rise_ticks[enc_idx];
    enc_pw_ticks[enc_idx] = pw;
    enc_new_data[enc_idx] = true;
  }
}

// TMR1 ISR — handles J2 (CH1) and J3 (CH0)
static void tmr1_isr() {
  // Check CH0 (encoder index 2 = J3)
  if (ENCODER_ENABLED[2] && (IMXRT_TMR1.CH[0].SCTRL & TMR_SCTRL_IEF)) {
    processCapture(2, IMXRT_TMR1.CH[0]);
  }
  // Check CH1 (encoder index 1 = J2)
  if (ENCODER_ENABLED[1] && (IMXRT_TMR1.CH[1].SCTRL & TMR_SCTRL_IEF)) {
    processCapture(1, IMXRT_TMR1.CH[1]);
  }
}

// TMR3 ISR — handles J1 (CH2), J4 (CH3), J5 (CH0), J6 (CH1)
static void tmr3_isr() {
  // Check CH0 (encoder index 4 = J5)
  if (ENCODER_ENABLED[4] && (IMXRT_TMR3.CH[0].SCTRL & TMR_SCTRL_IEF)) {
    processCapture(4, IMXRT_TMR3.CH[0]);
  }
  // Check CH1 (encoder index 5 = J6)
  if (ENCODER_ENABLED[5] && (IMXRT_TMR3.CH[1].SCTRL & TMR_SCTRL_IEF)) {
    processCapture(5, IMXRT_TMR3.CH[1]);
  }
  // Check CH2 (encoder index 0 = J1)
  if (ENCODER_ENABLED[0] && (IMXRT_TMR3.CH[2].SCTRL & TMR_SCTRL_IEF)) {
    processCapture(0, IMXRT_TMR3.CH[2]);
  }
  // Check CH3 (encoder index 3 = J4)
  if (ENCODER_ENABLED[3] && (IMXRT_TMR3.CH[3].SCTRL & TMR_SCTRL_IEF)) {
    processCapture(3, IMXRT_TMR3.CH[3]);
  }
}

// ============================================================================
// QTIMER HARDWARE INITIALIZATION
// ============================================================================

// IOMUX ALT1 function selects QTimer input for each pin
static void setupQTimerPin(uint8_t enc_idx) {
  // Set pin mux to ALT1 (QTimer function) and configure pad
  // Pad config: Hysteresis ON, 100K pull-up, keeper enabled
  const uint32_t PAD_CONFIG = 0x1B000;  // HYS | PUS(100K up) | PUE | PKE

  switch (ENCODER_PINS[enc_idx]) {
    case 10: // GPIO_B0_00 → TMR1_CH0
      IOMUXC_SW_MUX_CTL_PAD_GPIO_B0_00 = 1;  // ALT1
      IOMUXC_SW_PAD_CTL_PAD_GPIO_B0_00 = PAD_CONFIG;
      break;
    case 12: // GPIO_B0_01 → TMR1_CH1
      IOMUXC_SW_MUX_CTL_PAD_GPIO_B0_01 = 1;
      IOMUXC_SW_PAD_CTL_PAD_GPIO_B0_01 = PAD_CONFIG;
      break;
    case 14: // GPIO_AD_B1_02 → TMR3_CH2
      IOMUXC_SW_MUX_CTL_PAD_GPIO_AD_B1_02 = 1;
      IOMUXC_SW_PAD_CTL_PAD_GPIO_AD_B1_02 = PAD_CONFIG;
      IOMUXC_QTIMER3_TIMER2_SELECT_INPUT = 0;  // Select this pad
      break;
    case 15: // GPIO_AD_B1_03 → TMR3_CH3
      IOMUXC_SW_MUX_CTL_PAD_GPIO_AD_B1_03 = 1;
      IOMUXC_SW_PAD_CTL_PAD_GPIO_AD_B1_03 = PAD_CONFIG;
      IOMUXC_QTIMER3_TIMER3_SELECT_INPUT = 0;
      break;
    case 18: // GPIO_AD_B1_01 → TMR3_CH1
      IOMUXC_SW_MUX_CTL_PAD_GPIO_AD_B1_01 = 1;
      IOMUXC_SW_PAD_CTL_PAD_GPIO_AD_B1_01 = PAD_CONFIG;
      IOMUXC_QTIMER3_TIMER1_SELECT_INPUT = 0;
      break;
    case 19: // GPIO_AD_B1_00 → TMR3_CH0
      IOMUXC_SW_MUX_CTL_PAD_GPIO_AD_B1_00 = 1;
      IOMUXC_SW_PAD_CTL_PAD_GPIO_AD_B1_00 = PAD_CONFIG;
      IOMUXC_QTIMER3_TIMER0_SELECT_INPUT = 1;
      break;
  }
}

static void setupQTimerChannel(volatile IMXRT_TMR_CH_t &ch, uint8_t secondary_ch) {
  ch.CTRL = 0;  // Disable before configuring

  // CTRL: Count mode 001 (count rising edges of primary = free-run from clock)
  //   PCS = 1011 (IPG bus clock / 8 = 18.75 MHz)
  //   SCS = secondary_ch (input pin for this channel)
  ch.CTRL = TMR_CTRL_CM(1)
          | TMR_CTRL_PCS(QTIMER_PCS_VALUE)
          | TMR_CTRL_SCS(secondary_ch);

  // SCTRL: Capture on BOTH edges, enable edge interrupt
  ch.SCTRL = TMR_SCTRL_CAPTURE_MODE(3)  // 11 = capture on any edge
           | TMR_SCTRL_IEFIE;           // Input Edge Flag Interrupt Enable

  // FILT: Hardware digital glitch filter
  // Requires FILT_CNT+2 consecutive samples at FILT_PER interval to accept edge
  // At IPG=150MHz, PER=20 → 133ns/sample, CNT=3 → 5 samples = 667ns filter
  ch.FILT = TMR_FILT_FILT_PER(QTIMER_FILT_PER)
          | TMR_FILT_FILT_CNT(QTIMER_FILT_CNT);

  // Load and compare: free-running, count up to 0xFFFF
  ch.LOAD = 0;
  ch.COMP1 = 0xFFFF;
  ch.COMP2 = 0xFFFF;
  ch.CNTR = 0;
}

// ============================================================================
// ENCODER SMOOTHING FILTERS
// ============================================================================

static uint16_t pw_history[NUM_MOTORS][3] = {{0}};
static uint8_t  pw_fill[NUM_MOTORS] = {0};

static uint16_t median3_u16(uint16_t a, uint16_t b, uint16_t c) {
  if (a > b) { uint16_t t = a; a = b; b = t; }
  if (b > c) { uint16_t t = b; b = c; c = t; }
  if (a > b) { uint16_t t = a; a = b; b = t; }
  return b;
}

// EMA state
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

void controlArm() { armed = true; }
bool controlIsArmed() { return armed; }

// ============================================================================
// COMMAND INTERFACE
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

  // ---------- QTimer hardware capture decode ----------
  uint16_t raw_pw_ticks;
  bool has_new_data;
  noInterrupts();
  raw_pw_ticks = enc_pw_ticks[idx];
  has_new_data = enc_new_data[idx];
  enc_new_data[idx] = false;
  interrupts();

  if (raw_pw_ticks == 0) return joints[idx].actual_position;

  // Convert QTimer ticks to microseconds
  // QTimer clock = IPG (150 MHz) / 8 = 18.75 MHz → 1 tick = 0.05333 us
  static const float TICKS_TO_US = (float)QTIMER_PRESCALER / 150000000.0f * 1000000.0f;
  float raw_pw = (float)raw_pw_ticks * TICKS_TO_US;

  // Hardware sanity check (should never fail with QTimer FILT active)
  if (raw_pw < 1.5f || raw_pw > 1100.0f) {
    #if ENCODER_EMA_ENABLED
      return ema_initialised[idx] ? ema_position[idx] : joints[idx].actual_position;
    #else
      return joints[idx].actual_position;
    #endif
  }

  // --- Median-of-3 filter on raw ticks ---
  uint16_t pw_ticks;
#if ENCODER_MEDIAN_FILTER
  if (has_new_data) {
    uint8_t slot = pw_fill[idx] < 3 ? pw_fill[idx] : (pw_fill[idx] % 3);
    pw_history[idx][slot] = raw_pw_ticks;
    pw_fill[idx]++;
    if (pw_fill[idx] > 200) pw_fill[idx] = 3;
  }
  if (pw_fill[idx] >= 3) {
    pw_ticks = median3_u16(pw_history[idx][0], pw_history[idx][1], pw_history[idx][2]);
  } else {
    pw_ticks = raw_pw_ticks;
  }
#else
  pw_ticks = raw_pw_ticks;
#endif

  // Convert median ticks to microseconds for angle calc
  float pw = (float)pw_ticks * TICKS_TO_US;

  // MT6816 angle decode
  float clocks = pw / (ENCODER_CLOCK_PERIOD_NS / 1000.0f);
  float counts = clocks - (float)ENCODER_START_CLOCKS;
  if (counts < 0.0f)                        counts = 0.0f;
  if (counts >= (float)ENCODER_RESOLUTION)   counts = (float)(ENCODER_RESOLUTION - 1);

  float motor_ang = (counts / (float)ENCODER_RESOLUTION) * 2.0f * PI;
  motor_ang *= ENCODER_DIR_SIGN[idx];
  motor_ang += ENCODER_OFFSETS[idx];

  // Normalise to [0, 2π)
  if (!isfinite(motor_ang)) motor_ang = 0.0f;
  motor_ang = fmodf(motor_ang, 2.0f * PI);
  if (motor_ang < 0.0f) motor_ang += 2.0f * PI;

  // ---------- MULTI-TURN TRACKING ----------
  if (joints[idx].last_motor_angle < 0.0f) {
    joints[idx].last_motor_angle = motor_ang;
    joints[idx].total_motor_angle = motor_ang;
  } else {
    float ang_diff = motor_ang - joints[idx].last_motor_angle;

    // Normalise to [-PI, PI) — shortest signed angular distance
    while (ang_diff >  PI) ang_diff -= 2.0f * PI;
    while (ang_diff <= -PI) ang_diff += 2.0f * PI;

    // With hardware QTimer FILT + median filter, the data is clean.
    // No software EMI rejection needed — just accumulate directly.
    joints[idx].total_motor_angle += ang_diff;
    joints[idx].last_motor_angle = motor_ang;
  }

  float raw_joint_pos = joints[idx].total_motor_angle / GEAR_RATIOS[idx];

  // --- EMA on final joint position ---
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
// CONTROL INIT
// ============================================================================

void controlInit() {
  // Zero all joint states
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    joints[i] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f};
  }

  // Enable QTimer clocks (CCM_CCGR6 bits for TMR1 and TMR3)
  CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON) | CCM_CCGR6_QTIMER3(CCM_CCGR_ON);

  // Configure QTimer channels for each enabled encoder
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    if (!ENCODER_ENABLED[i]) continue;

    // 1. Configure pin mux to ALT1 (QTimer input)
    setupQTimerPin(i);

    // 2. Configure QTimer channel
    uint8_t mod = ENC_TMR_MODULE[i];
    uint8_t ch  = ENC_TMR_CHANNEL[i];

    if (mod == TMR_MOD_TMR1) {
      setupQTimerChannel(IMXRT_TMR1.CH[ch], ch);
    } else if (mod == TMR_MOD_TMR3) {
      setupQTimerChannel(IMXRT_TMR3.CH[ch], ch);
    }
  }

  // Enable TMR1 channels that are active
  {
    uint16_t enbl = 0;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
      if (ENCODER_ENABLED[i] && ENC_TMR_MODULE[i] == TMR_MOD_TMR1) {
        enbl |= (1 << ENC_TMR_CHANNEL[i]);
      }
    }
    if (enbl) TMR1_ENBL = enbl;
  }

  // Enable TMR3 channels that are active
  {
    uint16_t enbl = 0;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
      if (ENCODER_ENABLED[i] && ENC_TMR_MODULE[i] == TMR_MOD_TMR3) {
        enbl |= (1 << ENC_TMR_CHANNEL[i]);
      }
    }
    if (enbl) TMR3_ENBL = enbl;
  }

  // Attach ISRs
  attachInterruptVector(IRQ_QTIMER1, tmr1_isr);
  NVIC_SET_PRIORITY(IRQ_QTIMER1, 16);  // High priority
  NVIC_ENABLE_IRQ(IRQ_QTIMER1);

  attachInterruptVector(IRQ_QTIMER3, tmr3_isr);
  NVIC_SET_PRIORITY(IRQ_QTIMER3, 16);
  NVIC_ENABLE_IRQ(IRQ_QTIMER3);

  // Initialize motor outputs
  motorsInit();
}

// ============================================================================
// CONTROL LOOP  (called from IntervalTimer at 500 Hz)
// ============================================================================

void controlUpdate() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    JointState* j = &joints[i];

    j->actual_position = readEncoder(i);

    if (!armed) {
      j->velocity_command = 0.0f;
      j->position_error   = 0.0f;
      motorSetFrequency(i, 0.0f);
      continue;
    }

    j->position_error = j->desired_position - j->actual_position;

    j->velocity_command = j->desired_velocity
                        + Kp[i] * j->position_error;

    if (j->velocity_command >  MAX_JOINT_VELOCITIES[i])
      j->velocity_command =  MAX_JOINT_VELOCITIES[i];
    if (j->velocity_command < -MAX_JOINT_VELOCITIES[i])
      j->velocity_command = -MAX_JOINT_VELOCITIES[i];

    if (fabsf(j->velocity_command) < VELOCITY_DEADBAND)
      j->velocity_command = 0.0f;

    bool forward = (j->velocity_command * MOTOR_DIR_SIGN[i]) >= 0.0f;
    motorSetDirection(i, forward);

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
