#pragma once

#include "EncoderHAL.h"
#include <Arduino.h>

/**
 * PWM Capture Encoder HAL (Phase 1.5)
 * 
 * Uses hardware interrupts (attachInterrupt) to measure the HIGH time of 
 * an incoming PWM signal. It expects a ~1 kHz PWM signal where the duty 
 * cycle linearly maps to the angle (0-2PI).
 * 
 * Compatible with synthetic ESP32 simulators and real MT6816 encoders.
 */
class PwmCaptureEncoder : public EncoderHAL {
public:
    PwmCaptureEncoder(int pwm_pin) : pin_(pwm_pin), high_time_us_(0), last_rise_time_(0) {}

    void init() override {
        pinMode(pin_, INPUT);
    }
    
    // Called by the external interrupt handlers in main.cpp
    void handle_interrupt() {
        uint32_t now = micros();
        if (digitalReadFast(pin_) == HIGH) {
            // Rising edge: Start timing
            last_rise_time_ = now;
        } else {
            // Falling edge: End timing and store High duration
            if (last_rise_time_ > 0) {
                high_time_us_ = now - last_rise_time_;
            }
        }
    }
    
    // Called at 1 kHz by control loop ISR
    float read_angle() override {
        // Read volatile variable safely
        uint32_t high_time = high_time_us_;
        
        // MT6816 typically outputs ~1kHz PWM, so a full period is roughly 1000us.
        // We calculate duty cycle (high_time / period).
        // For Phase 1.5, we assume the simulated 1kHz period.
        const float PERIOD_US = 1000.0f;
        
        float duty = (float)high_time / PERIOD_US;
        
        // Clamp bounds just in case
        if (duty < 0.0f) duty = 0.0f;
        if (duty > 1.0f) duty = 1.0f;
        
        // Map 0-100% duty to 0-2PI Rads
        return duty * 2.0f * PI; 
    }
    
private:
    int pin_;
    volatile uint32_t high_time_us_;
    volatile uint32_t last_rise_time_;
};
