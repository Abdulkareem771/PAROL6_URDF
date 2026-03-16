/*
 * PAROL6 Motor Control — TIM10 Phase-Accumulator DDS Step Pulse Generation
 *
 * Replaces busy-wait GPIO toggling with timer-driven DDS:
 *   TIM10 runs at 40 kHz (25µs per tick).
 *   Each tick: check 6 phase accumulators, set GPIO HIGH on overflow.
 *   Next tick: clear the pin LOW → 25µs pulse width.
 *
 * Zero busy-wait. Zero EMI burst. 25µs step pulses.
 *
 * The control loop (TIM11 at 500 Hz) only updates phase_increment values.
 * The actual pulse timing is handled entirely by TIM10 hardware.
 *
 * Phase accumulator: 32-bit, overflows at 2^32
 *   phase_inc = (desired_freq / 40000) × 2^32
 *   Max freq:  20 kHz (overflow every other tick)
 *   Min freq:  ~0.01 Hz (phase_inc = 1)
 *   Resolution: 40000 / 2^32 ≈ 0.00001 Hz
 */

#include "motor.h"
#include <Arduino.h>
#include <HardwareTimer.h>
#include <stm32f4xx.h>

// Pin arrays for indexed access
const uint32_t STEP_PINS[NUM_MOTORS] = {
    STEP_PIN_J1, STEP_PIN_J2, STEP_PIN_J3,
    STEP_PIN_J4, STEP_PIN_J5, STEP_PIN_J6
};

const uint32_t DIR_PINS_ARR[NUM_MOTORS] = {
    DIR_PIN_J1, DIR_PIN_J2, DIR_PIN_J3,
    DIR_PIN_J4, DIR_PIN_J5, DIR_PIN_J6
};

// Pre-computed GPIO port and pin mask for single-cycle BSRR access
static GPIO_TypeDef* step_port[NUM_MOTORS];
static uint16_t      step_mask[NUM_MOTORS];
static GPIO_TypeDef* dir_port[NUM_MOTORS];
static uint16_t      dir_mask[NUM_MOTORS];

// ============================================================================
// DDS STATE (accessed from TIM10 ISR at 40 kHz)
// ============================================================================

static volatile uint32_t phase_acc[NUM_MOTORS] = {0};  // phase accumulator
static volatile uint32_t phase_inc[NUM_MOTORS] = {0};  // phase increment (set by control loop)
static volatile bool     pulse_active[NUM_MOTORS] = {false};  // pulse HIGH flag

// Conversion constant: (2^32) / STEP_TIMER_FREQ
// For 40 kHz: 4294967296 / 40000 = 107374.1824
static const double PHASE_INC_SCALE = 4294967296.0 / (double)STEP_TIMER_FREQ;

// ============================================================================
// TIM10 ISR — 40 kHz step pulse DDS
// ============================================================================
// This ISR body is tiny: 6 additions + 6 conditionals + GPIO writes.
// At 96 MHz, ~80 cycles total = 0.83µs. CPU load = 0.83/25 = 3.3%.

static HardwareTimer *stepTimer = nullptr;

static void stepTimerISR(void)
{
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        // 1. End previous pulse (set LOW) — 25µs after HIGH was set
        if (pulse_active[i]) {
            step_port[i]->BSRR = (uint32_t)step_mask[i] << 16U;  // LOW
            pulse_active[i] = false;
        }

        // 2. Skip if motor is off
        uint32_t inc = phase_inc[i];
        if (inc == 0) continue;

        // 3. Phase accumulator: add increment, detect overflow (carry)
        uint32_t old = phase_acc[i];
        phase_acc[i] += inc;

        // Overflow means: time for a new step pulse
        if (phase_acc[i] < old) {
            step_port[i]->BSRR = step_mask[i];  // HIGH
            pulse_active[i] = true;
        }
    }
}

// ============================================================================
// INIT
// ============================================================================

void motorsInit(void)
{
    // Step pins: configure as output and pre-compute port/mask
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        pinMode(STEP_PINS[i], OUTPUT);
        digitalWrite(STEP_PINS[i], LOW);

        PinName pn = digitalPinToPinName(STEP_PINS[i]);
        step_port[i] = get_GPIO_Port(STM_PORT(pn));
        step_mask[i] = (uint16_t)(1U << STM_PIN(pn));

        phase_acc[i] = 0;
        phase_inc[i] = 0;
        pulse_active[i] = false;
    }

    // Direction pins: configure and pre-compute port/mask
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        pinMode(DIR_PINS_ARR[i], OUTPUT);
        digitalWrite(DIR_PINS_ARR[i], LOW);

        PinName pn = digitalPinToPinName(DIR_PINS_ARR[i]);
        dir_port[i] = get_GPIO_Port(STM_PORT(pn));
        dir_mask[i] = (uint16_t)(1U << STM_PIN(pn));
    }

    // Proximity sensor pins
    pinMode(PROX_PIN_1, INPUT_PULLUP);
    pinMode(PROX_PIN_2, INPUT_PULLUP);
    pinMode(PROX_PIN_3, INPUT_PULLUP);

    // Start TIM10 at 40 kHz for DDS step generation
    stepTimer = new HardwareTimer(TIM10);
    stepTimer->setOverflow(STEP_TIMER_FREQ, HERTZ_FORMAT);
    stepTimer->attachInterrupt(stepTimerISR);
    stepTimer->resume();

    // Set TIM10 ISR priority: lower than control (3), higher than USB (5)
    HAL_NVIC_SetPriority(TIM1_UP_TIM10_IRQn, 4, 0);
}

// ============================================================================
// SET FREQUENCY (called from TIM11 control ISR at 500 Hz)
// ============================================================================
// Converts desired Hz to phase increment. Lock-free: TIM10 ISR reads
// phase_inc atomically (32-bit aligned write on Cortex-M4 is atomic).

void motorSetFrequency(uint8_t motor_idx, float frequency_hz)
{
    if (motor_idx >= NUM_MOTORS) return;

    if (frequency_hz < 1.0f) {
        phase_inc[motor_idx] = 0;
        return;
    }

    if (frequency_hz > (float)MAX_STEP_FREQUENCY_HZ)
        frequency_hz = (float)MAX_STEP_FREQUENCY_HZ;

    // phase_inc = (freq / timer_freq) × 2^32
    phase_inc[motor_idx] = (uint32_t)(frequency_hz * PHASE_INC_SCALE);
}

// ============================================================================
// SET DIRECTION (called from TIM11 control ISR at 500 Hz)
// ============================================================================

void motorSetDirection(uint8_t motor_idx, bool forward)
{
    if (motor_idx >= NUM_MOTORS) return;
    if (forward) {
        dir_port[motor_idx]->BSRR = dir_mask[motor_idx];           // HIGH
    } else {
        dir_port[motor_idx]->BSRR = (uint32_t)dir_mask[motor_idx] << 16U;  // LOW
    }
}
