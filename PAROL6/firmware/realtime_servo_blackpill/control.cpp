/*
 * PAROL6 Control — STM32 PWM Input Mode Edition
 *
 * Control loop runs from TIM11 update ISR at 500 Hz.
 * Encoder reading: polls CCR1/CCR2 from each timer (PWM Input mode).
 *
 * No encoder interrupts. No DMA. Just register reads.
 * Hardware does all edge timing — we just compute duty cycle.
 *
 * MT6816 angle decode, median filter, EMA, and multi-turn tracking
 * are identical to the Teensy QTimer version.
 */

#include "control.h"
#include "motor.h"
#include "encoder_capture.h"
#include <math.h>
#include <string.h>
#include <stm32f4xx.h>

// ============================================================================
// ENCODER SMOOTHING FILTERS
// ============================================================================

static uint16_t pw_history[NUM_MOTORS][3];
uint8_t  pw_fill[NUM_MOTORS] = {0};

static uint16_t median3_u16(uint16_t a, uint16_t b, uint16_t c)
{
    if (a > b) { uint16_t t = a; a = b; b = t; }
    if (b > c) { uint16_t t = b; b = c; c = t; }
    if (a > b) { uint16_t t = a; a = b; b = t; }
    return b;
}

float ema_position[NUM_MOTORS] = {0};
bool  ema_initialised[NUM_MOTORS] = {false};

// Track previous CCR values to detect stale reads
static uint32_t prev_period[NUM_MOTORS] = {0};
static uint32_t prev_hightime[NUM_MOTORS] = {0};
static uint8_t  capture_valid_count[NUM_MOTORS] = {0};

// MT6816 expected period range at 12 MHz capture clock:
//   PWM period ~1028 µs → ~12336 ticks
//   Allow ±30% margin: 8000 to 16000 ticks (16-bit timers)
//   For 32-bit timers, same range but CCR1 can hold huge stale values
#define ENC_MIN_PERIOD_TICKS   8000
#define ENC_MAX_PERIOD_TICKS   16000

// ============================================================================
// JOINT STATE
// ============================================================================

static JointState joints[NUM_MOTORS];
static volatile bool armed = false;

void controlArm(void)
{
    // Snapshot: set desired = actual for all joints at arm time.
    // Without this, desired stays at 0 (from memset) while encoder
    // reads non-zero → huge error → motor runaway on arm.
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        joints[i].desired_position = joints[i].actual_position;
        joints[i].desired_velocity = 0.0f;
    }
    armed = true;
}
bool controlIsArmed(void) { return armed; }

void controlSetCommand(uint8_t idx, float pos, float vel)
{
    if (idx >= NUM_MOTORS) return;
    joints[idx].desired_position = pos;
    joints[idx].desired_velocity = vel;
}

// ============================================================================
// ENCODER POSITION READING — PWM Input mode edition
// ============================================================================
// Instead of reading edge timestamps and computing pulse width in software,
// we read CCR1 (period) and CCR2 (high-time) directly from the timer hardware.
// The timer handles everything: edge detection, timestamping, counter reset.

static float readEncoder(uint8_t idx)
{
    // ---------- NON-ENCODER: velocity integration ----------
    if (!ENCODER_ENABLED[idx]) {
        joints[idx].actual_position +=
            joints[idx].velocity_command * ((float)CONTROL_PERIOD_US / 1000000.0f);
        return joints[idx].actual_position;
    }

    // ---------- PWM Input mode: read hardware registers ----------
    uint32_t period_ticks, hightime_ticks;
    encoderReadCapture(idx, &period_ticks, &hightime_ticks);

    // Sanity: skip if no valid capture yet
    if (period_ticks == 0 || hightime_ticks == 0) {
        return joints[idx].actual_position;
    }

    // VALIDITY CHECK: Period must be within MT6816's expected range.
    // On first rising edge after timer start, CCR1 can capture a huge
    // stale value (time since timer init). Reject until stable.
    if (period_ticks < ENC_MIN_PERIOD_TICKS || period_ticks > ENC_MAX_PERIOD_TICKS) {
        return joints[idx].actual_position;
    }

    // STALE READ CHECK: If CCR values haven't changed since last read,
    // the encoder hasn't produced a new cycle. Skip to avoid processing
    // the same data twice (which can cause drift from floating-point noise).
    if (period_ticks == prev_period[idx] && hightime_ticks == prev_hightime[idx]) {
        // Same capture — don't re-decode (avoids FP accumulation noise)
        return joints[idx].actual_position;
    }
    prev_period[idx] = period_ticks;
    prev_hightime[idx] = hightime_ticks;

    // Require 3 consecutive valid captures before trusting the data.
    // This skips the initial transient captures after timer start.
    if (capture_valid_count[idx] < 3) {
        capture_valid_count[idx]++;
        return joints[idx].actual_position;
    }

    // Convert high-time ticks to microseconds for MT6816 angle decode
    float raw_pw = (float)hightime_ticks * TICKS_TO_US;

    // Hardware sanity check
    if (raw_pw < 1.5f || raw_pw > 1100.0f) {
#if ENCODER_EMA_ENABLED
        return ema_initialised[idx] ? ema_position[idx] : joints[idx].actual_position;
#else
        return joints[idx].actual_position;
#endif
    }

    // --- Median-of-3 filter on raw high-time ticks ---
    uint16_t pw_ticks;
    uint16_t raw_pw_ticks = (hightime_ticks > 0xFFFF) ? 0xFFFF : (uint16_t)hightime_ticks;

#if ENCODER_MEDIAN_FILTER
    {
        uint8_t slot = pw_fill[idx] < 3 ? pw_fill[idx] : (pw_fill[idx] % 3);
        pw_history[idx][slot] = raw_pw_ticks;
        pw_fill[idx]++;
        if (pw_fill[idx] > 200) pw_fill[idx] = 3;  // prevent wrap
    }
    if (pw_fill[idx] >= 3) {
        pw_ticks = median3_u16(pw_history[idx][0], pw_history[idx][1], pw_history[idx][2]);
    } else {
        pw_ticks = raw_pw_ticks;
    }
#else
    pw_ticks = raw_pw_ticks;
#endif

    // Convert median ticks to microseconds
    float pw = (float)pw_ticks * TICKS_TO_US;

    // MT6816 angle decode (identical to Teensy)
    float clocks = pw / (ENCODER_CLOCK_PERIOD_NS / 1000.0f);
    float counts = clocks - (float)ENCODER_START_CLOCKS;
    if (counts < 0.0f) counts = 0.0f;
    if (counts >= (float)ENCODER_RESOLUTION) counts = (float)(ENCODER_RESOLUTION - 1);

    float motor_ang = (counts / (float)ENCODER_RESOLUTION) * TWO_PI;
    motor_ang *= ENCODER_DIR_SIGN[idx];
    motor_ang += ENCODER_OFFSETS[idx];

    // Normalise to [0, 2π)
    if (!isfinite(motor_ang)) motor_ang = 0.0f;
    motor_ang = fmodf(motor_ang, TWO_PI);
    if (motor_ang < 0.0f) motor_ang += TWO_PI;

    // ---------- MULTI-TURN TRACKING ----------
    if (joints[idx].last_motor_angle < 0.0f) {
        joints[idx].last_motor_angle = motor_ang;
        joints[idx].total_motor_angle = motor_ang;
    } else {
        float ang_diff = motor_ang - joints[idx].last_motor_angle;
        // SYMMETRIC wrap to [-PI, PI) — both bounds use strict inequality
        // Previous code used (<= -PI) which caused systematic +2π drift
        // when ang_diff landed exactly at -π due to FP precision
        while (ang_diff >  M_PI) ang_diff -= TWO_PI;
        while (ang_diff < -M_PI) ang_diff += TWO_PI;
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
// 500 Hz CONTROL LOOP
// ============================================================================

void controlUpdate(void)
{
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        JointState *j = &joints[i];

        // 1. Read encoder (polls hardware CCR registers — no ISR involved)
        j->actual_position = readEncoder(i);

        if (!armed) {
            j->velocity_command = 0.0f;
            j->position_error   = 0.0f;
            motorSetFrequency(i, 0.0f);
            continue;
        }

        // 2. Position error
        j->position_error = j->desired_position - j->actual_position;

        // 3. Control law: velocity feedforward + proportional correction
        j->velocity_command = j->desired_velocity + Kp[i] * j->position_error;

        // 4. Velocity clamp
        if (j->velocity_command >  MAX_JOINT_VELOCITIES[i]) j->velocity_command =  MAX_JOINT_VELOCITIES[i];
        if (j->velocity_command < -MAX_JOINT_VELOCITIES[i]) j->velocity_command = -MAX_JOINT_VELOCITIES[i];

        // 4b. Deadband
        if (fabsf(j->velocity_command) < VELOCITY_DEADBAND) j->velocity_command = 0.0f;

        // 5. Direction
        bool forward = (j->velocity_command * MOTOR_DIR_SIGN[i]) >= 0.0f;
        motorSetDirection(i, forward);

        // 6. Step frequency
        float motor_vel     = fabsf(j->velocity_command) * GEAR_RATIOS[i];
        float steps_per_rev = (float)(STEPS_PER_REV * MICROSTEPS[i]);
        float step_freq     = (motor_vel * steps_per_rev) / TWO_PI;
        motorSetFrequency(i, step_freq);
    }
}

// ============================================================================
// TIM11 ISR — 500 Hz control loop (using STM32Duino HardwareTimer API)
// ============================================================================
// Using HardwareTimer instead of raw registers to avoid linker conflict
// with STM32Duino's built-in TIM1_TRG_COM_TIM11_IRQHandler.

#include <HardwareTimer.h>

static HardwareTimer *controlTimer = nullptr;

static void controlTimerCallback(void)
{
    controlUpdate();
}

static void controlTimerInit(void)
{
    controlTimer = new HardwareTimer(TIM11);
    controlTimer->setOverflow(CONTROL_FREQUENCY_HZ, HERTZ_FORMAT);
    controlTimer->attachInterrupt(controlTimerCallback);
    controlTimer->resume();

    // Set priority: below faults, above USB
    HAL_NVIC_SetPriority(TIM1_TRG_COM_TIM11_IRQn, 3, 0);
}

// ============================================================================
// INIT
// ============================================================================

void controlInit(void)
{
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        memset(&joints[i], 0, sizeof(JointState));
        joints[i].last_motor_angle = -1.0f;
    }

    encoderCaptureInit();
    motorsInit();
    controlTimerInit();
}

const JointState* controlGetState(uint8_t idx)
{
    if (idx >= NUM_MOTORS) return 0;
    return &joints[idx];
}

// ============================================================================
// RESET POSITION (called by homing after sensor trigger)
// ============================================================================
// Resets all encoder tracking state so the joint position reads as new_position.
// Must be called when the motor is STOPPED (no motion).

void controlResetPosition(uint8_t idx, float new_position)
{
    if (idx >= NUM_MOTORS) return;

    JointState *j = &joints[idx];

    // Reset multi-turn tracking: set total angle such that
    // joint position = total_motor_angle / gear_ratio = new_position
    float new_motor_angle = new_position * GEAR_RATIOS[idx];
    j->total_motor_angle = new_motor_angle;
    j->last_motor_angle = -1.0f;  // force re-init on next readEncoder()

    // Reset EMA filter state
#if ENCODER_EMA_ENABLED
    extern float ema_position[];
    extern bool  ema_initialised[];
    ema_position[idx] = new_position;
    ema_initialised[idx] = true;
#endif

    // Reset median filter state
#if ENCODER_MEDIAN_FILTER
    extern uint8_t pw_fill[];
    pw_fill[idx] = 0;
#endif

    // Set current and desired position
    j->actual_position   = new_position;
    j->desired_position  = new_position;
    j->desired_velocity  = 0.0f;
    j->velocity_command  = 0.0f;
    j->position_error    = 0.0f;
}
