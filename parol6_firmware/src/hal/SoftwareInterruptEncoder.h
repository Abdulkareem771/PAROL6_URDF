#pragma once

#include <Arduino.h>
#include "EncoderHAL.h"

#ifndef FASTRUN
#define FASTRUN
#endif

// -------------------------------------------------------------------------
// Global ISR state for SoftwareInterruptEncoder
// -------------------------------------------------------------------------
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

/**
 * SoftwareInterruptEncoder — Standard attachInterrupt fallback for MT6816.
 * Extracted exactly from the proven Tested_Working_SingleMotor_Integration branch.
 */
class SoftwareInterruptEncoder : public EncoderHAL {
public:
    SoftwareInterruptEncoder(int pin, int axis_index) 
        : pin_(pin), idx_(axis_index), initialized_(false) {}

    void init() override {
        if (idx_ < 0 || idx_ > 5) return;
        
        enc_pins[idx_] = pin_;
        pinMode(pin_, INPUT);
        
        // Attach the specific static ISR for this axis
        switch(idx_) {
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
        if (!initialized_) return 0.0f;
        
        uint32_t pulse_width_us;
        noInterrupts();
        pulse_width_us = enc_pulse_widths[idx_];
        interrupts();

        if (pulse_width_us == 0) return 0.0f; // No data yet

        // Math sourced directly from stepdir_velocity_control_encoder.ino
        const float ENCODER_CLOCK_PERIOD_NS = 250.0f;
        const float ENCODER_START_CLOCKS = 16.0f;
        const float ENCODER_RESOLUTION = 4096.0f;

        float clocks = (pulse_width_us * 1000.0f) / ENCODER_CLOCK_PERIOD_NS;
        float angle_clocks = clocks - ENCODER_START_CLOCKS;

        if (angle_clocks < 0.0f) angle_clocks = 0.0f;
        if (angle_clocks > ENCODER_RESOLUTION - 1.0f) angle_clocks = ENCODER_RESOLUTION - 1.0f;

        return (angle_clocks / ENCODER_RESOLUTION) * 2.0f * PI;
    }

private:
    int pin_;
    int idx_;
    bool initialized_;
};
