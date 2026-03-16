/*
 * PAROL6 Real-Time Servo Control — STM32F411CE Black Pill
 *
 * Arduino IDE / STM32Duino sketch.
 *
 * Board settings in Arduino IDE:
 *   Board:        "Generic STM32F4 series"
 *   Board part:   "BlackPill F411CE"
 *   USB support:  "CDC (generic 'Serial' supersede U(S)ART)"
 *   Upload:       "STM32CubeProgrammer (DFU)"
 *
 * Architecture:
 *   6 timers (TIM1-5, TIM9) in PWM Input mode → encoder capture (ZERO ISR load)
 *   TIM11 update ISR                           → controlUpdate()  @ 500 Hz
 *   Main loop()                                → serial I/O       @ 50 Hz
 *
 * PWM Input mode:
 *   For each encoder, the timer hardware does ALL edge timing.
 *   CH1 captures the period (rising-to-rising), CH2 captures high-time.
 *   Slave reset mode resets the counter on each rising edge.
 *   The control ISR simply reads CCR1/CCR2 — no encoder interrupts needed.
 *
 * DFU upload:
 *   Hold BOOT0 → press RESET → release → upload in Arduino IDE.
 *   PA10 must be pulled to GND (Black Pill HW bug). PA10 is NOT used.
 *
 * STM32F411CE @ 96 MHz (PLL: HSE 25 MHz, PLLM=25, PLLN=192, PLLP=2, PLLQ=4)
 *   96 MHz SYSCLK, 48 MHz USB clock (exact)
 *   667 ns digital filter on encoder inputs — matches Teensy QTimer FILT
 */

#include "config.h"
#include "control.h"
#include "serial_comm.h"

// ============================================================================
// SETUP
extern "C" void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    // Configure the main internal regulator output voltage
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    // Initialize the CPU, AHB and APB busses clocks
    // Use HSE (25 MHz crystal on WeAct BlackPill V2)
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 25;   // 25 / 25 = 1 MHz
    RCC_OscInitStruct.PLL.PLLN = 192;  // 1 * 192 = 192 MHz VCO
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2; // 192 / 2 = 96 MHz SYSCLK
    RCC_OscInitStruct.PLL.PLLQ = 4;    // 192 / 4 = 48 MHz USB clock (EXACT)

    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    // Initialize the CPU, AHB and APB busses clocks
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK) {
        Error_Handler();
    }
}

void setup()
{
    // STM32Duino executes the overridden SystemClock_Config() automatically.

    // Initialize serial (USB CDC 12 Mbps)
    serialCommInit();

    // Initialize control system (encoders in PWM Input mode, motors, TIM11 ISR)
    controlInit();

    // Onboard LED: PC13 (active LOW on Black Pill)
    pinMode(PC13, OUTPUT);
    digitalWrite(PC13, LOW);   // LED ON — boot indicator
    delay(200);
    digitalWrite(PC13, HIGH);  // LED OFF
}

// ============================================================================
// MAIN LOOP — serial I/O at 50 Hz
// ============================================================================

static uint32_t last_feedback_ms = 0;

void loop()
{
    // Process incoming commands (non-blocking)
    serialCommProcessIncoming();

    // Send feedback at 50 Hz
    uint32_t now = millis();
    if ((now - last_feedback_ms) >= FEEDBACK_PERIOD_MS) {
        last_feedback_ms = now;
        serialCommSendFeedback();
    }
}
