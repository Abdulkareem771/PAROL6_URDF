#pragma once

#include <Arduino.h>
#include "EncoderHAL.h"
#include "imxrt.h"

// ---------------------------------------------------------------------------
// MT6816 PWM frame constants (MagnTek MT6816 datasheet §8.5)
//   Frame  = 4119 PWM-clock periods @ 250 ns each → 1.030 ms period (~971 Hz)
//   Data   = 4096 counts covering 0→360° (12-bit absolute)
//   Duty   = (angle_count + 1) / 4119,  angle_count ∈ [0, 4095]
//   The encoder NEVER outputs 0% or 100% duty — guard ticks pad both sides.
// ---------------------------------------------------------------------------
#ifndef MT6816_TOTAL_TICKS
#define MT6816_TOTAL_TICKS 4119.0f
#endif
#ifndef MT6816_DATA_TICKS
#define MT6816_DATA_TICKS  4096.0f
#endif

// Number of 1 kHz ISR ticks to accumulate before computing a new angle.
// With a ~1.03 ms PWM frame, 10 ticks ≈ 10 complete MT6816 frames per measurement.
// This avoids mid-frame sampling artifacts and increases high-tick resolution at
// low angles (duty ≈ 1/4119 → ~4 ticks/frame → ~40 ticks over 10 frames).
#ifndef QTIMER_ACCUMULATE_TICKS
#define QTIMER_ACCUMULATE_TICKS 10
#endif

/**
 * QuadTimerEncoder — Zero-interrupt absolute angle measurement for MT6816.
 *
 * Uses i.MXRT1062 hardware QuadTimer in Gated Count mode (CM=3) to measure
 * PWM HIGH time without any CPU interrupts. The QuadTimer counter increments
 * on IP-Bus/8 (18.75 MHz, 53.3 ns/tick) only while the encoder pin is HIGH.
 *
 * Called every ISR tick (1 kHz). Accumulates over QTIMER_ACCUMULATE_TICKS
 * ticks before computing a new angle (avoids mid-frame sampling errors).
 *
 * Compatible: MagnTek MT6816 in PWM output mode (971 Hz or 485 Hz frame rate).
 */
class QuadTimerEncoder : public EncoderHAL {
public:
    QuadTimerEncoder(uint8_t pin) : pin_(pin) {}

    void init() override {
        // Map pin → QuadTimer module + channel, enable CCM clock gate
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
            return; // Unsupported pin — tmr_ stays nullptr; read_angle() returns 0
        }

        // Pad mux: QuadTimer Alt-1 + SION (Software Input On forces pad → input path)
        *(portConfigRegister(pin_)) = 1 | 0x10;

        // Reset channel
        tmr_->CH[ch_].CTRL   = 0;
        tmr_->CH[ch_].CNTR   = 0;
        tmr_->CH[ch_].LOAD   = 0;
        tmr_->CH[ch_].SCTRL  = 0;
        tmr_->CH[ch_].CSCTRL = 0;

        // CM(3)     = Gated Count Mode: count IP-Bus/8 ONLY while input is HIGH
        // PCS(11)   = IP-Bus ÷ 8 → 18.75 MHz clock (53.3 ns/tick)
        // SCS(ch_)  = gate on this channel's secondary input (the pad signal)
        // No LENGTH bit: counter free-runs 0→0xFFFF→wrap; unsigned delta handles rollover.
        tmr_->CH[ch_].CTRL = TMR_CTRL_CM(3) | TMR_CTRL_PCS(8 + 3) | TMR_CTRL_SCS(ch_);

        // Prime state
        last_cntr_   = tmr_->CH[ch_].CNTR;
        last_cyccnt_ = ARM_DWT_CYCCNT;
        accum_high_  = 0;
        accum_total_ = 0.0f;
        accum_count_ = 0;
        last_angle_  = 0.0f;
    }

    // Called every 1 kHz ISR tick.
    // Accumulates for QTIMER_ACCUMULATE_TICKS ticks, then computes a new angle.
    // Returns cached last_angle_ between updates.
    float read_angle() override {
        if (!tmr_) return 0.0f;

        // Snapshot both counters atomically (within a few cycles of each other)
        uint16_t current_cntr   = tmr_->CH[ch_].CNTR;
        uint32_t current_cyccnt = ARM_DWT_CYCCNT;

        // HIGH-time ticks: 16-bit unsigned subtraction handles 0xFFFF→0 rollover
        accum_high_ += (uint16_t)(current_cntr - last_cntr_);
        last_cntr_   = current_cntr;

        // Total expected QuadTimer ticks: CPU-cycles → IP-Bus cycles → /8 prescaler
        uint32_t delta_cyccnt = current_cyccnt - last_cyccnt_;
        last_cyccnt_ = current_cyccnt;
        uint32_t cpu_to_ip = F_CPU_ACTUAL / F_BUS_ACTUAL;  // 600/150 = 4
        accum_total_ += (float)delta_cyccnt / (float)cpu_to_ip / 8.0f;

        accum_count_++;
        if (accum_count_ < QTIMER_ACCUMULATE_TICKS) {
            return last_angle_;  // Window not complete yet
        }

        // ----- Compute new angle from completed window -----
        float new_angle = last_angle_;
        if (accum_total_ > 0.0f) {
            float duty = (float)accum_high_ / accum_total_;
            if (duty > 1.0f) duty = 1.0f;
            if (duty < 0.0f) duty = 0.0f;

            // MT6816: duty = (count + 1) / 4119, count ∈ [0, 4095]
            // Invert: count = duty × 4119 − 1
            // Angle  = count / 4096 × 2π
            float raw_count = duty * MT6816_TOTAL_TICKS - 1.0f;
            if (raw_count < 0.0f) raw_count = 0.0f;
            new_angle = (raw_count / MT6816_DATA_TICKS) * 2.0f * PI;
        }

        // Reset window
        accum_high_  = 0;
        accum_total_ = 0.0f;
        accum_count_ = 0;
        last_angle_  = new_angle;
        return new_angle;
    }

private:
    uint8_t  pin_;
    volatile IMXRT_TMR_t* tmr_ = nullptr;
    uint8_t  ch_ = 0;

    uint16_t last_cntr_   = 0;
    uint32_t last_cyccnt_ = 0;

    uint32_t accum_high_  = 0;     // cumulative HIGH ticks over window
    float    accum_total_ = 0.0f;  // cumulative expected total ticks over window
    uint8_t  accum_count_ = 0;     // ISR ticks elapsed in current window
    float    last_angle_  = 0.0f;  // cached output between window completions
};
