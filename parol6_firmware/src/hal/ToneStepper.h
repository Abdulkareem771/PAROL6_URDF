#pragma once

#include <Arduino.h>

/**
 * Fallback Stepper Generator using Arduino's built-in tone() function.
 * Used when FEATURE_HARDWARE_PWM is 0.
 * Useful for debugging hardware timers to verify base functionality.
 */
class ToneStepper {
public:
    ToneStepper(int step_pin, int dir_pin)
        : step_pin_(step_pin), dir_pin_(dir_pin), current_hz_(0.0f) {}

    void init() {
        pinMode(step_pin_, OUTPUT);
        pinMode(dir_pin_, OUTPUT);
        stop();
    }

    void set_direction(bool forward) {
        // digitalWrite is fast enough on Teensy, but digitalWriteFast removes overhead
        digitalWriteFast(dir_pin_, forward ? HIGH : LOW);
    }

    void set_frequency(float hz) {
        if (hz < 1.0f) {
            stop();
            return;
        }
        
        // Prevent constant re-triggering of the tone timer if frequency hasn't meaningfully changed
        if (fabs(hz - current_hz_) > 0.5f) {
            tone(step_pin_, (uint32_t)hz);
            current_hz_ = hz;
        }
    }

    void stop() {
        noTone(step_pin_);
        digitalWriteFast(step_pin_, LOW);
        current_hz_ = 0.0f;
    }

private:
    int step_pin_;
    int dir_pin_;
    float current_hz_;
};
