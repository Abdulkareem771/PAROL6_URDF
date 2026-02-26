#pragma once

#include "EncoderHAL.h"
#include <Arduino.h>

// Temporary encoder replacement using the synthetic ESP32 simulator methodology
class MicrosEncoder : public EncoderHAL {
public:
    MicrosEncoder(int pin_a_placeholder, int pin_b_placeholder) {
        // Unused in Phase 1 fake reading
    }

    void init() override {
        start_time_ = micros();
    }
    
    // Simple mock returning a swept sine wave with synthetic noise
    float read_angle() override {
        uint32_t now = micros() - start_time_;
        float t_sec = now / 1000000.0f;
        
        // True physical path: 0.5 Hz sine wave, amplitude 2.0 radians
        float clean_pos = 2.0f * sin(2.0f * M_PI * 0.5f * t_sec);
        
        // Inject Synthetic Sensor Noise: +/- 0.025 radians
        // Note: rand() is used here ONLY for simulation. Never use rand() in a real ISR.
        float noise = ((rand() % 100) / 100.0f) * 0.05f - 0.025f;
        
        return clean_pos + noise; 
    }
    
private:
    uint32_t start_time_;
};
