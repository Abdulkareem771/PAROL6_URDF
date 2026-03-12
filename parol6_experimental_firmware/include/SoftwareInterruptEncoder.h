#pragma once

#include <Arduino.h>

#include "EncoderHAL.h"

volatile uint32_t enc_rise_times[6] = {0, 0, 0, 0, 0, 0};
volatile uint32_t enc_pulse_widths[6] = {0, 0, 0, 0, 0, 0};
volatile int enc_pins[6] = {-1, -1, -1, -1, -1, -1};

FASTRUN void enc_isr_0() {
    if (digitalRead(enc_pins[0])) {
        enc_rise_times[0] = micros();
    } else {
        enc_pulse_widths[0] = micros() - enc_rise_times[0];
    }
}
FASTRUN void enc_isr_1() {
    if (digitalRead(enc_pins[1])) {
        enc_rise_times[1] = micros();
    } else {
        enc_pulse_widths[1] = micros() - enc_rise_times[1];
    }
}
FASTRUN void enc_isr_2() {
    if (digitalRead(enc_pins[2])) {
        enc_rise_times[2] = micros();
    } else {
        enc_pulse_widths[2] = micros() - enc_rise_times[2];
    }
}
FASTRUN void enc_isr_3() {
    if (digitalRead(enc_pins[3])) {
        enc_rise_times[3] = micros();
    } else {
        enc_pulse_widths[3] = micros() - enc_rise_times[3];
    }
}
FASTRUN void enc_isr_4() {
    if (digitalRead(enc_pins[4])) {
        enc_rise_times[4] = micros();
    } else {
        enc_pulse_widths[4] = micros() - enc_rise_times[4];
    }
}
FASTRUN void enc_isr_5() {
    if (digitalRead(enc_pins[5])) {
        enc_rise_times[5] = micros();
    } else {
        enc_pulse_widths[5] = micros() - enc_rise_times[5];
    }
}

class SoftwareInterruptEncoder : public EncoderHAL {
public:
    SoftwareInterruptEncoder(int pin, int axis_index)
        : pin_(pin), idx_(axis_index), initialized_(false) {}

    void init() override {
        if (idx_ < 0 || idx_ > 5) {
            return;
        }

        enc_pins[idx_] = pin_;
        pinMode(pin_, INPUT);

        switch (idx_) {
            case 0: attachInterrupt(digitalPinToInterrupt(pin_), enc_isr_0, CHANGE); break;
            case 1: attachInterrupt(digitalPinToInterrupt(pin_), enc_isr_1, CHANGE); break;
            case 2: attachInterrupt(digitalPinToInterrupt(pin_), enc_isr_2, CHANGE); break;
            case 3: attachInterrupt(digitalPinToInterrupt(pin_), enc_isr_3, CHANGE); break;
            case 4: attachInterrupt(digitalPinToInterrupt(pin_), enc_isr_4, CHANGE); break;
            case 5: attachInterrupt(digitalPinToInterrupt(pin_), enc_isr_5, CHANGE); break;
        }

        initialized_ = true;
    }

    float read_angle() override {
        if (!initialized_) {
            return 0.0f;
        }

        uint32_t pulse_width_us;
        noInterrupts();
        pulse_width_us = enc_pulse_widths[idx_];
        interrupts();

        if (pulse_width_us == 0) {
            return 0.0f;
        }

        const float encoder_clock_period_ns = 250.0f;
        const float encoder_start_clocks = 16.0f;
        const float encoder_resolution = 4096.0f;

        float clocks = (pulse_width_us * 1000.0f) / encoder_clock_period_ns;
        float angle_clocks = clocks - encoder_start_clocks;

        if (angle_clocks < 0.0f) {
            angle_clocks = 0.0f;
        }
        if (angle_clocks > encoder_resolution - 1.0f) {
            angle_clocks = encoder_resolution - 1.0f;
        }

        return (angle_clocks / encoder_resolution) * 2.0f * PI;
    }

private:
    int pin_;
    int idx_;
    bool initialized_;
};

