#pragma once

#include <Arduino.h>
#include "EncoderHAL.h"
#include "imxrt.h"

/**
 * QuadTimerEncoder - Hard real-time duty cycle measurement.
 *
 * Provides ZERO-interrupt PWM pulse width integration using the i.MXRT1062
 * hardware QuadTimers. This offloads 100% of the encoder capture work from
 * the CPU, providing native 14-bit resolution jitter-free telemetry for the
 * 1 kHz control loop.
 */
class QuadTimerEncoder : public EncoderHAL {
public:
    QuadTimerEncoder(uint8_t pin) : pin_(pin) {}

    void init() override {
        // Map the user-provided physical pin to the specific hardware QuadTimer and channel.
        if (pin_ == 10) {
            tmr_ = &IMXRT_TMR1; ch_ = 0;
            CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON);
        } else if (pin_ == 11) {
            tmr_ = &IMXRT_TMR1; ch_ = 2;
            CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON);
        } else if (pin_ == 12) {
            tmr_ = &IMXRT_TMR1; ch_ = 1;
            CCM_CCGR6 |= CCM_CCGR6_QTIMER1(CCM_CCGR_ON);
        } else if (pin_ == 14) {
            tmr_ = &IMXRT_TMR3; ch_ = 2;
            CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON);
            IOMUXC_QTIMER3_TIMER2_SELECT_INPUT = 1;
        } else if (pin_ == 15) {
            tmr_ = &IMXRT_TMR3; ch_ = 3;
            CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON);
            IOMUXC_QTIMER3_TIMER3_SELECT_INPUT = 1;
        } else if (pin_ == 18) {
            tmr_ = &IMXRT_TMR3; ch_ = 1;
            CCM_CCGR6 |= CCM_CCGR6_QTIMER3(CCM_CCGR_ON);
            IOMUXC_QTIMER3_TIMER1_SELECT_INPUT = 0;
        } else {
            // Unsupported pin! 
            return;
        }

        // Configure the IO pad multiplexer to Alt-1 (QuadTimer mode) and assert SION (Software Input On)
        *(portConfigRegister(pin_)) = 1 | 0x10;

        // Reset the timer and disable it during configuration
        tmr_->CH[ch_].CTRL = 0;
        tmr_->CH[ch_].CNTR = 0;
        tmr_->CH[ch_].LOAD = 0;
        tmr_->CH[ch_].SCTRL = 0;
        tmr_->CH[ch_].CSCTRL = 0;

        // QuadTimer "Gated Count Mode":
        // This is the magic. The timer automatically counts the Primary Count Source (IP Bus clock)
        // ONLY when the Secondary Count Source (the physical pin) is HIGH.
        // It acts as a flawless hardware pulse-width integrator.
        // 
        // CM(6)   = Gated count mode
        // PCS(11) = IP Bus Clock / 8 (Prescaler 8 ensures a 1ms window won't overflow the 16-bit register)
        // SCS(ch) = Matches our physical input pin mappings
        tmr_->CH[ch_].CTRL = TMR_CTRL_CM(6) | TMR_CTRL_PCS(8 + 3) | TMR_CTRL_SCS(ch_) | TMR_CTRL_LENGTH;

        // Prime the state history
        last_cntr_ = tmr_->CH[ch_].CNTR;
        last_cyccnt_ = ARM_DWT_CYCCNT;
    }

    float read_angle() override {
        if (!tmr_) return 0.0f;

        // Atomic capture of both the QuadTimer hardware counter and the CPU cycle counter
        // This must be sampled as close together as possible.
        uint16_t current_cntr = tmr_->CH[ch_].CNTR;
        uint32_t current_cyccnt = ARM_DWT_CYCCNT;

        // Calculate delta (high time ticks) using 16-bit unsigned math which natively handles the rollover
        uint16_t high_ticks = current_cntr - last_cntr_;
        last_cntr_ = current_cntr;

        // Calculate exact CPU cycles elapsed in the identical time frame
        uint32_t delta_cyccnt = current_cyccnt - last_cyccnt_;
        last_cyccnt_ = current_cyccnt;

        // Map the CPU cycles to the expected QuadTimer clock boundaries.
        // CPU runs at F_CPU_ACTUAL (e.g. 600MHz), IP Bus runs at F_BUS_ACTUAL (e.g. 150MHz)
        // We prescaled the QuadTimer by 8.
        uint32_t cpu_to_ip_ratio = F_CPU_ACTUAL / F_BUS_ACTUAL;
        float expected_quadtimer_ticks = (float)delta_cyccnt / (float)cpu_to_ip_ratio / 8.0f;

        // Calculate exact Duty Cycle from the exact ticks captured in hardware.
        float duty_cycle = 0.0f;
        if (expected_quadtimer_ticks > 0.0f) {
            duty_cycle = (float)high_ticks / expected_quadtimer_ticks;
        }

        // Clamp noise
        if (duty_cycle > 1.0f) duty_cycle = 1.0f;
        if (duty_cycle < 0.0f) duty_cycle = 0.0f;

        // Linear mapping from 0.0-1.0 to 0-(2*PI)
        return duty_cycle * 2.0f * PI;
    }

private:
    uint8_t pin_;
    volatile IMXRT_TMR_t* tmr_ = nullptr;
    uint8_t ch_ = 0;
    
    uint16_t last_cntr_ = 0;
    uint32_t last_cyccnt_ = 0;
};
