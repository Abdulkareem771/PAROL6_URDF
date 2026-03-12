#pragma once

#include <Arduino.h>

class FlexPWMGenerator {
public:
    FlexPWMGenerator(uint8_t step_pin, uint8_t dir_pin)
        : step_pin_(step_pin), dir_pin_(dir_pin), running_(false) {}

    void init() {
        pinMode(dir_pin_, OUTPUT);
        pinMode(step_pin_, OUTPUT);
        digitalWriteFast(dir_pin_, LOW);
        analogWriteFrequency(step_pin_, 1000);
        analogWrite(step_pin_, 0);
        running_ = false;
    }

    void set_direction(bool forward) {
        digitalWriteFast(dir_pin_, forward ? HIGH : LOW);
    }

    void set_frequency(float hz) {
        if (hz < 1.0f) {
            stop();
            return;
        }

        if (!running_ || fabsf(hz - current_freq_) > 0.5f) {
            analogWriteFrequency(step_pin_, hz);
            analogWrite(step_pin_, 128);
            running_ = true;
            current_freq_ = hz;
        }
    }

    void stop() {
        if (running_) {
            analogWrite(step_pin_, 0);
            running_ = false;
            current_freq_ = 0.0f;
        }
    }

private:
    uint8_t step_pin_;
    uint8_t dir_pin_;
    bool running_;
    float current_freq_ = 0.0f;
};

