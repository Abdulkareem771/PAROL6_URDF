#pragma once

#include <Arduino.h>

/**
 * @brief Phase 4: ActuatorModel — Kinematic Math Layer
 *
 * Converts kinematic velocity_command (rad/s from ControlLaw) into:
 *   - step_frequency_hz  (pulses/s for FlexPWMGenerator)
 *   - direction           (physical DIR pin state)
 *
 * =========================================================================
 * ALL HARDWARE CONSTANTS IN THIS FILE COME FROM THE AUTHORITATIVE SOURCE:
 *   "legacy code open loop/PAROL6 control board main software/src/motor_init.cpp"
 *   (the original STM32-based open-loop firmware shipped with the real robot)
 * =========================================================================
 *
 * CRITICAL PIN ZONING NOTES (see TEENSY_PIN_ZONING.md):
 *
 *   Zone 1 — LOCKED ENCODER PINS (QuadTimerEncoder.h):
 *     J1→Pin10, J2→Pin11, J3→Pin12, J4→Pin14, J5→Pin15, J6→Pin18
 *     ⛔ These pins CANNOT be used for STEP, DIR, or any other purpose.
 *
 *   Zone 2 — STEP pins (FlexPWM, each on separate submodule):
 *     Recommended: 2, 3, 4, 5, 6, 7, 8, 9
 *
 *   Zone 3 — DIR pins (pure GPIO, no timer functionality):
 *     Recommended: 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
 *
 *   The realtime_servo_teensy reference used DIR pins 24–29 which overlap
 *   Zone 2 (pins 28, 29 are FlexPWM-capable STEP candidates).
 *   That assignment is INCORRECT for our architecture.
 *   We use Zone 3 pins (30–35) for DIR to avoid any hardware contention.
 */

// ---------------------------------------------------------------------------
// Zone 2: STEP pins — each on a different FlexPWM submodule (verified)
//   Pin 2  → FlexPWM4.2A   Pin 4 → FlexPWM2.0A   Pin 5 → FlexPWM2.1A
//   Pin 6  → FlexPWM2.2A   Pin 7 → FlexPWM1.3B   Pin 8 → FlexPWM1.3A
// ---------------------------------------------------------------------------
static const uint8_t STEP_PINS[6] = { 2, 6, 7, 8, 4, 5 };

// ---------------------------------------------------------------------------
// Zone 3: DIR pins — pure GPIO, no FlexPWM or QuadTimer involvement
// ---------------------------------------------------------------------------
static const uint8_t DIR_PINS[6] = { 30, 31, 32, 33, 34, 35 };

// ---------------------------------------------------------------------------
// Motor mechanical configuration
// Source: motor_init.cpp (legacy STM32 firmware, authoritative)
//
// Gear ratios:
//   J1 = 6.4        (reduction: 96 / 15 ≈ 6.4, worm-bevel)
//   J2 = 20.0       (20:1 gearbox)
//   J3 = 18.0952381 (20 × 38/42, exact fractional ratio — DO NOT round)
//   J4 = 4.0        (4:1 gearbox)
//   J5 = 4.0        (4:1 gearbox)
//   J6 = 10.0       (10:1 gearbox)
//
// Microstepping: 32 globally (from constants.h, #define MICROSTEP 32)
//
// Direction inversion (from motor_init.cpp, `direction_reversed` field):
//   J1 → reversed = 1   (motor CW = joint CCW)
//   J2 → reversed = 0
//   J3 → reversed = 1
//   J4 → reversed = 0
//   J5 → reversed = 0
//   J6 → reversed = 1
// ---------------------------------------------------------------------------
static const float   GEAR_RATIOS[6]    = { 6.4f, 20.0f, 18.0952381f, 4.0f, 4.0f, 10.0f };
static const int     MICROSTEPS_ALL    = 32;           // Global (constants.h)
static const int     STEPS_PER_REV    = 200;
// dir_sign: +1 = not reversed, -1 = reversed
static const int     MOTOR_DIR_SIGN[6] = { -1, 1, -1, 1, 1, -1 };

// Maximum step frequency (Hz) for MKS Servo42C — validated from reference firmware
static const float   MAX_STEP_FREQ_HZ     = 20000.0f;

// Stop threshold: below this frequency, disable the PWM output entirely.
// FlexPWM is unreliable at very low frequencies with 8-bit resolution.
static const float   STOP_FREQ_THRESH_HZ  = 10.0f;

// ---------------------------------------------------------------------------
// Steps-per-radian by joint (precomputed for reference):
//   steps_per_rad = (STEPS_PER_REV * MICROSTEPS * GEAR_RATIO) / (2 * PI)
//   J1: 200 * 32 * 6.4  / 6.283 = 6513.7 steps/rad
//   J2: 200 * 32 * 20.0 / 6.283 = 20372  steps/rad
//   J3: 200 * 32 * 18.095 / 6.283 = 18415 steps/rad
//   J4: 200 * 32 * 4.0  / 6.283 = 4074   steps/rad
//   J5: 200 * 32 * 4.0  / 6.283 = 4074   steps/rad
//   J6: 200 * 32 * 10.0 / 6.283 = 10186  steps/rad
// ---------------------------------------------------------------------------

class ActuatorModel {
public:
    /**
     * @param gear_ratio  Motor revolutions per joint revolution.
     * @param microsteps  Microstep divisor configured on driver.
     * @param dir_sign    +1 = forward; -1 = reversed motor mounting.
     */
    ActuatorModel(float gear_ratio, int microsteps, int dir_sign)
        : dir_sign_(dir_sign)
    {
        // Precompute steps_per_rad once to avoid repeat multiply in 1 kHz ISR
        steps_per_rad_ = ((float)STEPS_PER_REV * (float)microsteps * gear_ratio) / (2.0f * PI);
    }

    /**
     * Converts kinematic velocity (rad/s) into stepper frequency (Hz) + DIR.
     *
     * @param velocity_rad_s    Clamped, deadbanded command from ControlLaw.
     * @param out_freq_hz       For FlexPWMGenerator::set_frequency().
     * @param out_direction     Logic state for DIR pin (true = forward after inversion).
     */
    void compute(float velocity_rad_s, float& out_freq_hz, bool& out_direction) const {
        // 1. Direction — apply physical mounting orientation correction
        bool positive = (velocity_rad_s >= 0.0f);
        out_direction = (dir_sign_ > 0) ? positive : !positive;

        // 2. Frequency (always positive)
        float f = fabsf(velocity_rad_s) * steps_per_rad_;
        if (f > MAX_STEP_FREQ_HZ) f = MAX_STEP_FREQ_HZ;
        out_freq_hz = f;
    }

    /** Returns true if frequency is below the reliable hardware threshold. */
    bool should_stop(float freq_hz) const {
        return freq_hz < STOP_FREQ_THRESH_HZ;
    }

    /**
     * Factory: returns a pre-configured ActuatorModel for axis [0..5].
     * Uses GEAR_RATIOS, MICROSTEPS_ALL, and MOTOR_DIR_SIGN from this file.
     */
    static ActuatorModel create_joint(int axis) {
        return ActuatorModel(GEAR_RATIOS[axis], MICROSTEPS_ALL, MOTOR_DIR_SIGN[axis]);
    }

private:
    int   dir_sign_;
    float steps_per_rad_;  // cached: (STEPS_PER_REV * microsteps * gear_ratio) / TWO_PI
};
