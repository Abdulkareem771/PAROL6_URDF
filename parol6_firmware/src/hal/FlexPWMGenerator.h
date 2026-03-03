#pragma once

#include <Arduino.h>

/**
 * @brief Phase 4 Stage 2 — Real Per-Axis STEP/DIR Stepper Driver
 *
 * Wraps Teensy's FlexPWM peripheral (via analogWriteFrequency/analogWrite)
 * and a GPIO DIR pin to drive one stepper motor axis.
 *
 * Design invariants:
 *  - ZERO execution inside the 1 kHz ISR (set-and-forget hardware registers).
 *  - set_frequency() and stop() update FlexPWM registers atomically.
 *  - set_direction() must be called BEFORE set_frequency() on direction change
 *    to satisfy MKS Servo42C DIR-to-STEP setup time (≥200 ns).
 *
 * Pin zones (see ActuatorModel.h and TEENSY_PIN_ZONING.md):
 *   STEP pins → Zone 2 FlexPWM-capable: 2, 4, 5, 6, 7, 8
 *   DIR  pins → Zone 3 pure GPIO:       30, 31, 32, 33, 34, 35
 */
class FlexPWMGenerator {
public:
    FlexPWMGenerator(uint8_t step_pin, uint8_t dir_pin)
        : step_pin_(step_pin), dir_pin_(dir_pin), running_(false) {}

    /**
     * Initialize PWM and DIR pins. Must be called from setup() BEFORE
     * encoders (FlexPWM CCM clock must be enabled before QuadTimer init).
     */
    void init() {
        pinMode(dir_pin_,  OUTPUT);
        pinMode(step_pin_, OUTPUT);
        digitalWriteFast(dir_pin_,  LOW);
        // Ensure PWM output starts off (analogWrite(pin, 0) = constant LOW)
        analogWriteFrequency(step_pin_, 1000);  // seed frequency (avoids 0Hz glitch)
        analogWrite(step_pin_, 0);              // output off
        running_ = false;
    }

    /**
     * Set the physical DIR pin state.
     * Call BEFORE set_frequency() when changing direction.
     * MKS Servo42C requires ≥200 ns DIR-to-STEP setup time; the ISR period
     * (1 ms) and the main-loop call sequence guarantee this.
     */
    void set_direction(bool forward) {
        digitalWriteFast(dir_pin_, forward ? HIGH : LOW);
    }

    /**
     * Start or update the STEP pulse frequency.
     * analogWriteFrequency reprograms the FlexPWM hardware timer registers
     * without CPU involvement after the call returns.
     * Duty cycle is fixed at 50% (128/256) — MKS Servo42C only needs the
     * rising edge; pulse width is irrelevant above the 2 µs minimum.
     */
    void set_frequency(float hz) {
        if (hz < 1.0f) { stop(); return; }
        // Only reprogram the hardware if the frequency changed significantly
        // (saves ~3µs per ISR cycle on identical consecutive calls)
        if (!running_ || fabsf(hz - current_freq_) > 0.5f) {
            analogWriteFrequency(step_pin_, (float)hz);
            analogWrite(step_pin_, 128);  // 50% duty cycle
            running_ = true;
            current_freq_ = hz;
        }
    }

    /**
     * Stop the STEP output (motor holds position — driver still energized).
     * analogWrite(pin, 0) forces the FlexPWM output LOW without disabling
     * the peripheral (fast to restart).
     */
    void stop() {
        if (running_) {
            analogWrite(step_pin_, 0);
            running_ = false;
            current_freq_ = 0.0f;
        }
    }

    /** True if the PWM output is currently active. */
    bool is_running() const { return running_; }

private:
    uint8_t step_pin_;
    uint8_t dir_pin_;
    bool    running_;
    float   current_freq_ = 0.0f;
};
