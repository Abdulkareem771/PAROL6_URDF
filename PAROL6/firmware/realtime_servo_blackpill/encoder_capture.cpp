/*
 * PAROL6 Encoder Capture — STM32 PWM Input Mode Implementation
 *
 * Configures 6 timers in PWM Input mode using direct register access.
 * Each timer measures one MT6816 PWM encoder signal entirely in hardware.
 *
 * Register configuration per timer:
 *   SMCR: SMS=100 (reset mode), TS=101 (TI1FP1 trigger)
 *   CCMR1: CC1S=01 (IC1→TI1), CC2S=10 (IC2→TI1), IC1F/IC2F=9 (667ns filter)
 *   CCER: CC1E=1 CC1P=0 (rising), CC2E=1 CC2P=1 (falling)
 *   PSC: 7 (96 MHz / 8 = 12 MHz)
 *   ARR: 0xFFFF (16-bit) or 0xFFFFFFFF (32-bit)
 *
 * After configuration, CCR1 holds period, CCR2 holds high-time.
 * The control ISR polls these directly — zero interrupt load.
 */

#include "encoder_capture.h"
#include <stm32f4xx.h>

// ============================================================================
// TIMER INSTANCE TABLE
// ============================================================================

// Timer peripheral base addresses for each encoder
static TIM_TypeDef* const ENC_TIM[NUM_MOTORS] = {
    TIM1,   // J1
    TIM2,   // J2
    TIM3,   // J3
    TIM4,   // J4
    TIM5,   // J5
    TIM9    // J6
};

// Whether timer is 32-bit (TIM2, TIM5) or 16-bit
static const bool ENC_TIM_32BIT[NUM_MOTORS] = {
    false,  // TIM1: 16-bit
    true,   // TIM2: 32-bit
    false,  // TIM3: 16-bit
    false,  // TIM4: 16-bit
    true,   // TIM5: 32-bit
    false   // TIM9: 16-bit
};

// ============================================================================
// GPIO ALTERNATE FUNCTION CONFIGURATION
// ============================================================================

static void configureEncoderGPIO(void)
{
    // Enable GPIO clocks
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN;

    // Helper: configure pin as AF with pull-up
    // PA8 → AF1 (TIM1_CH1)
    GPIOA->MODER   &= ~(3U << (8*2));
    GPIOA->MODER   |=  (2U << (8*2));   // AF mode
    GPIOA->AFR[1]  &= ~(0xFU << ((8-8)*4));
    GPIOA->AFR[1]  |=  (1U << ((8-8)*4));  // AF1
    GPIOA->PUPDR   &= ~(3U << (8*2));
    GPIOA->PUPDR   |=  (1U << (8*2));   // Pull-up

    // PA15 → AF1 (TIM2_CH1)
    GPIOA->MODER   &= ~(3U << (15*2));
    GPIOA->MODER   |=  (2U << (15*2));
    GPIOA->AFR[1]  &= ~(0xFU << ((15-8)*4));
    GPIOA->AFR[1]  |=  (1U << ((15-8)*4));  // AF1
    GPIOA->PUPDR   &= ~(3U << (15*2));
    GPIOA->PUPDR   |=  (1U << (15*2));

    // PA6 → AF2 (TIM3_CH1)
    GPIOA->MODER   &= ~(3U << (6*2));
    GPIOA->MODER   |=  (2U << (6*2));
    GPIOA->AFR[0]  &= ~(0xFU << (6*4));
    GPIOA->AFR[0]  |=  (2U << (6*4));  // AF2
    GPIOA->PUPDR   &= ~(3U << (6*2));
    GPIOA->PUPDR   |=  (1U << (6*2));

    // PB6 → AF2 (TIM4_CH1)
    GPIOB->MODER   &= ~(3U << (6*2));
    GPIOB->MODER   |=  (2U << (6*2));
    GPIOB->AFR[0]  &= ~(0xFU << (6*4));
    GPIOB->AFR[0]  |=  (2U << (6*4));  // AF2
    GPIOB->PUPDR   &= ~(3U << (6*2));
    GPIOB->PUPDR   |=  (1U << (6*2));

    // PA0 → AF2 (TIM5_CH1)
    GPIOA->MODER   &= ~(3U << (0*2));
    GPIOA->MODER   |=  (2U << (0*2));
    GPIOA->AFR[0]  &= ~(0xFU << (0*4));
    GPIOA->AFR[0]  |=  (2U << (0*4));  // AF2
    GPIOA->PUPDR   &= ~(3U << (0*2));
    GPIOA->PUPDR   |=  (1U << (0*2));

    // PA2 → AF3 (TIM9_CH1)
    GPIOA->MODER   &= ~(3U << (2*2));
    GPIOA->MODER   |=  (2U << (2*2));
    GPIOA->AFR[0]  &= ~(0xFU << (2*4));
    GPIOA->AFR[0]  |=  (3U << (2*4));  // AF3
    GPIOA->PUPDR   &= ~(3U << (2*2));
    GPIOA->PUPDR   |=  (1U << (2*2));
}

// ============================================================================
// PWM INPUT MODE CONFIGURATION (per timer)
// ============================================================================

static void configurePWMInputMode(TIM_TypeDef *tim, bool is_32bit)
{
    // Disable timer while configuring
    tim->CR1 = 0;

    // Prescaler: 96 MHz / 8 = 12 MHz capture clock
    tim->PSC = ENC_TIM_PRESCALER;

    // Auto-reload: free-running
    tim->ARR = is_32bit ? 0xFFFFFFFF : 0xFFFF;

    // ---- CCMR1: Input capture channel mapping + digital filter ----
    // CC1S = 01: IC1 mapped to TI1 (direct)
    // CC2S = 10: IC2 mapped to TI1 (indirect — same pin as CH1)
    // IC1F = 9: digital filter (~667 ns at 12 MHz)
    // IC2F = 9: same filter on CH2
    tim->CCMR1 = (1U << 0)      // CC1S = 01 (input, IC1→TI1)
               | (ENC_TIM_FILTER << 4)   // IC1F = 9
               | (2U << 8)      // CC2S = 10 (input, IC2→TI1 indirect)
               | (ENC_TIM_FILTER << 12); // IC2F = 9

    // ---- CCER: Enable capture, set edge polarity ----
    // CC1E = 1: CH1 capture enabled
    // CC1P = 0: Rising edge on CH1 (period measurement)
    // CC2E = 1: CH2 capture enabled
    // CC2P = 1: Falling edge on CH2 (high-time measurement)
    tim->CCER = TIM_CCER_CC1E          // Enable CH1 capture
              | TIM_CCER_CC2E          // Enable CH2 capture
              | TIM_CCER_CC2P;         // CH2 falling edge (CC2P=1)
    // CC1P stays 0 (rising edge) — default

    // ---- SMCR: Slave mode = reset, trigger = TI1FP1 ----
    // SMS = 100 (reset mode): counter reset on trigger
    // TS  = 101 (TI1FP1): rising edge of filtered TI1 is the trigger
    tim->SMCR = (4U << 0)     // SMS = 100 (reset mode)
              | (5U << 4);    // TS = 101 (TI1FP1)

    // Clear counter and capture registers
    tim->CNT = 0;
    tim->CCR1 = 0;
    tim->CCR2 = 0;

    // Generate update event to load prescaler
    tim->EGR = TIM_EGR_UG;

    // Clear all flags
    tim->SR = 0;

    // Enable timer
    tim->CR1 = TIM_CR1_CEN;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void encoderCaptureInit(void)
{
    // Enable all timer clocks
    RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;    // TIM1 (APB2)
    RCC->APB1ENR |= RCC_APB1ENR_TIM2EN;    // TIM2 (APB1)
    RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;    // TIM3 (APB1)
    RCC->APB1ENR |= RCC_APB1ENR_TIM4EN;    // TIM4 (APB1)
    RCC->APB1ENR |= RCC_APB1ENR_TIM5EN;    // TIM5 (APB1)
    RCC->APB2ENR |= RCC_APB2ENR_TIM9EN;    // TIM9 (APB2)

    // Configure GPIO pins for alternate function
    configureEncoderGPIO();

    // Configure each enabled encoder's timer in PWM Input mode
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (ENCODER_ENABLED[i]) {
            configurePWMInputMode(ENC_TIM[i], ENC_TIM_32BIT[i]);
        }
    }
}

// ============================================================================
// POLLING API (called from control ISR — no interrupt involved)
// ============================================================================

void encoderReadCapture(uint8_t enc_idx, uint32_t *period_ticks, uint32_t *hightime_ticks)
{
    if (enc_idx >= NUM_MOTORS || !ENCODER_ENABLED[enc_idx]) {
        *period_ticks = 0;
        *hightime_ticks = 0;
        return;
    }

    TIM_TypeDef *tim = ENC_TIM[enc_idx];

    // Read hardware registers directly — these are continuously updated
    // by the timer hardware on each PWM cycle (slave reset mode)
    *period_ticks   = tim->CCR1;
    *hightime_ticks = tim->CCR2;
}
