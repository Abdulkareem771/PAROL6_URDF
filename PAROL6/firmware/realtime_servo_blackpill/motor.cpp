/*
 * PAROL6 Motor Control — GPIO DDS Step Pulse Generation
 *
 * Software DDS from the 500 Hz control ISR.
 * Uses direct BSRR register access for minimal-latency step pulses.
 * DWT->CYCCNT for precise 5 µs pulse width.
 */

#include "motor.h"
#include <Arduino.h>
#include <stm32f4xx.h>

// Pin arrays for indexed access (Arduino pin numbers)
const uint32_t STEP_PINS[NUM_MOTORS] = {
    STEP_PIN_J1, STEP_PIN_J2, STEP_PIN_J3,
    STEP_PIN_J4, STEP_PIN_J5, STEP_PIN_J6
};

const uint32_t DIR_PINS_ARR[NUM_MOTORS] = {
    DIR_PIN_J1, DIR_PIN_J2, DIR_PIN_J3,
    DIR_PIN_J4, DIR_PIN_J5, DIR_PIN_J6
};

// Pre-computed GPIO port and pin mask for ISR-speed BSRR access
static GPIO_TypeDef* step_port[NUM_MOTORS];
static uint16_t      step_mask[NUM_MOTORS];

// DDS state
static float step_accumulator[NUM_MOTORS] = {0};

void motorsInit(void)
{
    // Step pins: configure as output and pre-compute port/mask for BSRR
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        pinMode(STEP_PINS[i], OUTPUT);
        digitalWrite(STEP_PINS[i], LOW);

        // Extract GPIO port and pin mask from STM32Duino pin number
        // STM32Duino: digitalPinToPinName() → PinName, get_GPIO_Port/STM_PIN
        PinName pn = digitalPinToPinName(STEP_PINS[i]);
        step_port[i] = get_GPIO_Port(STM_PORT(pn));
        step_mask[i] = (uint16_t)(1U << STM_PIN(pn));

        step_accumulator[i] = 0.0f;
    }

    // Direction pins
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        pinMode(DIR_PINS_ARR[i], OUTPUT);
        digitalWrite(DIR_PINS_ARR[i], LOW);
    }

    // Proximity sensor pins
    pinMode(PROX_PIN_1, INPUT_PULLUP);
    pinMode(PROX_PIN_2, INPUT_PULLUP);
    pinMode(PROX_PIN_3, INPUT_PULLUP);

    // Enable DWT cycle counter for pulse timing
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void motorSetFrequency(uint8_t motor_idx, float frequency_hz)
{
    if (motor_idx >= NUM_MOTORS) return;

    if (frequency_hz < 10.0f) {
        step_accumulator[motor_idx] = 0.0f;
        return;
    }

    if (frequency_hz > (float)MAX_STEP_FREQUENCY_HZ)
        frequency_hz = (float)MAX_STEP_FREQUENCY_HZ;

    // DDS: accumulate fractional steps
    float steps_this_period = frequency_hz / (float)CONTROL_FREQUENCY_HZ;
    step_accumulator[motor_idx] += steps_this_period;

    uint32_t pulses = (uint32_t)step_accumulator[motor_idx];
    step_accumulator[motor_idx] -= (float)pulses;

    // Emit pulses with DWT-timed 5 µs width using direct BSRR register
    // At 96 MHz: 480 cycles = 5 µs
    const uint32_t pulse_cycles = MIN_STEP_PULSE_WIDTH_US * (SystemCoreClock / 1000000);
    GPIO_TypeDef *port = step_port[motor_idx];
    uint16_t mask      = step_mask[motor_idx];

    for (uint32_t p = 0; p < pulses; p++) {
        // SET pin HIGH (atomic, single-cycle write via BSRR lower 16 bits)
        port->BSRR = mask;

        // DWT busy-wait for precise pulse width
        uint32_t start = DWT->CYCCNT;
        while ((DWT->CYCCNT - start) < pulse_cycles) {}

        // RESET pin LOW (atomic, via BSRR upper 16 bits)
        port->BSRR = (uint32_t)mask << 16U;
    }
}

void motorSetDirection(uint8_t motor_idx, bool forward)
{
    if (motor_idx >= NUM_MOTORS) return;
    digitalWrite(DIR_PINS_ARR[motor_idx], forward ? HIGH : LOW);
}
