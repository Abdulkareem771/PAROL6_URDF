#pragma once

#include <Arduino.h>

/**
 * @brief Phase 4: Stage 1 - Dummy FlexPWM Carrier Generator
 * 
 * This class isolates the NXP i.MXRT1062 FlexPWM hardware peripheral.
 * It is used to generate a strict, CPU-independent 50 kHz pulse train
 * on a single Zone 2 pin (Pin 2).
 * 
 * The goal of this Stage 1 implementation is to prove that enabling 
 * the FlexPWM hardware subsystem does NOT introduce any latency or jitter 
 * to the critical 1 kHz `run_control_loop_isr`.
 * 
 * Invariants Respected:
 *  - ZERO execution inside the 1 kHz ISR math loop (set & forget).
 *  - Hardware timer generation (no digitalWrite/delay in loops).
 */
class FlexPWMDriver {
public:
    FlexPWMDriver(uint8_t dummy_pin) : pin_(dummy_pin) {}

    /**
     * Initializes the pin and starts the 50 kHz dummy waveform.
     * Uses Teensy's built-in analogWriteFrequency for Stage 1 simplicity,
     * which wraps the FlexPWM registers under the hood.
     * 
     * In Stage 2, this will be replaced with direct register access 
     * to manage exact phase alignments and direction dead-times.
     */
    void init_dummy_carrier() {
        // According to TEENSY_PIN_ZONING.md Zone 2:
        // Pins 2, 3, 4, 5, 4, 7, 8, 9 are FlexPWM capable step pins.
        pinMode(pin_, OUTPUT);
        
        // 50 kHz is a high-speed stepper pulse rate (approx 10 Rev/Sec at 16 microsteps)
        analogWriteFrequency(pin_, 50000); 
        
        // 50% duty cycle. The MKServo42C only cares about the rising edge.
        // 128/256 = 50%
        analogWrite(pin_, 128);
    }

private:
    uint8_t pin_;
};
