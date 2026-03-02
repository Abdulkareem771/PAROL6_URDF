/**
 * main.c — MT6816 Absolute Magnetic Encoder PWM Simulator
 *
 * Simulates 6 MT6816 encoders outputting PWM signals to the Teensy QuadTimers.
 * Each channel slowly sweeps its angle so you can see the oscilloscope trace move.
 *
 * MT6816 PWM Spec:
 *   - Frame = 4119 × 250 ns = ~1029 µs → ~971 Hz
 *   - 12-bit angle: raw duty = angle_counts / 4096
 *   - Angle range 0–360° maps to ~1/4096 to 4095/4096 (never 0 or 1)
 *
 * Wiring (ESP32 GPIO → Teensy 4.1 Pin):
 *   GPIO 18 → Teensy pin 10  (J1)
 *   GPIO 19 → Teensy pin 11  (J2)
 *   GPIO 21 → Teensy pin 12  (J3)
 *   GPIO 22 → Teensy pin 14  (J4)
 *   GPIO 23 → Teensy pin 15  (J5)
 *   GPIO 25 → Teensy pin 18  (J6)
 *   GND    → GND              (REQUIRED — common ground)
 *
 * UART0 (monitor) prints current simulated angles at 1 Hz.
 */

#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include "esp_log.h"

#define TAG "MT6816_SIM"

// ── PWM hardware config ────────────────────────────────────────────────────
// MT6816 frame = 4119 × 250 ns ≈ 971 Hz.
// ESP32 LEDC at 1000 Hz with 12-bit resolution matches closely enough.
#define PWM_FREQ_HZ     1000
#define PWM_RESOLUTION  LEDC_TIMER_12_BIT   // 12 bits = 4096 steps (matches MT6816)
#define PWM_MAX_DUTY    4095                 // 2^12 - 1

// ── GPIO assignments (ESP32 → Teensy QuadTimer pin) ───────────────────────
static const int PWM_GPIOS[6] = {18, 19, 21, 22, 23, 25};
// Corresponding Teensy encoder pins:         10   11   12   14   15   18

// ── LEDC channels / timers ────────────────────────────────────────────────
static const ledc_channel_t CHANNELS[6] = {
    LEDC_CHANNEL_0, LEDC_CHANNEL_1, LEDC_CHANNEL_2,
    LEDC_CHANNEL_3, LEDC_CHANNEL_4, LEDC_CHANNEL_5,
};

// ── Simulation state ─────────────────────────────────────────────────────
// Each joint sweeps at a different speed so you see all 6 channels moving
static float sim_angle_deg[6] = {0.0f, 60.0f, 120.0f, 180.0f, 240.0f, 300.0f};
// Sweep rate in deg/s for each joint
static const float SWEEP_RATE[6] = {15.0f, 20.0f, 10.0f, 25.0f, 12.0f, 18.0f};

// ── MT6816 angle → duty count ─────────────────────────────────────────────
// MT6816 maps 0–360° to counts 1–4095 (avoids 0 and 4096 = 100%).
// duty_count = (angle_deg / 360.0) * 4096  clamped [1, 4095]
static uint32_t angle_to_duty(float angle_deg) {
    // Normalise to 0–360
    while (angle_deg < 0.0f)   angle_deg += 360.0f;
    while (angle_deg >= 360.0f) angle_deg -= 360.0f;
    uint32_t cnt = (uint32_t)((angle_deg / 360.0f) * 4096.0f);
    if (cnt < 1)    cnt = 1;
    if (cnt > 4095) cnt = 4095;
    return cnt;
}

// ── Init LEDC timer ─────────────────────────────────────────────────────
static void pwm_init(void) {
    ledc_timer_config_t timer_cfg = {
        .duty_resolution = PWM_RESOLUTION,
        .freq_hz         = PWM_FREQ_HZ,
        .speed_mode      = LEDC_HIGH_SPEED_MODE,
        .timer_num       = LEDC_TIMER_0,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&timer_cfg);

    for (int i = 0; i < 6; i++) {
        ledc_channel_config_t ch = {
            .channel    = CHANNELS[i],
            .duty       = angle_to_duty(sim_angle_deg[i]),
            .gpio_num   = PWM_GPIOS[i],
            .speed_mode = LEDC_HIGH_SPEED_MODE,
            .timer_sel  = LEDC_TIMER_0,
            .hpoint     = 0,
        };
        ledc_channel_config(&ch);
        ESP_LOGI(TAG, "J%d  GPIO%2d → Teensy pin mapped  angle=%.1f°  duty=%lu",
                 i+1, PWM_GPIOS[i], sim_angle_deg[i], angle_to_duty(sim_angle_deg[i]));
    }
}

// ── Simulation task: update angles and PWM duty every 10ms ──────────────
static void sim_task(void *arg) {
    const float DT_S = 0.010f;  // 10 ms update tick

    while (1) {
        for (int i = 0; i < 6; i++) {
            sim_angle_deg[i] += SWEEP_RATE[i] * DT_S;
            if (sim_angle_deg[i] >= 360.0f) sim_angle_deg[i] -= 360.0f;
            uint32_t duty = angle_to_duty(sim_angle_deg[i]);
            ledc_set_duty(LEDC_HIGH_SPEED_MODE, CHANNELS[i], duty);
            ledc_update_duty(LEDC_HIGH_SPEED_MODE, CHANNELS[i]);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ── Log task: print angles at 1 Hz ──────────────────────────────────────
static void log_task(void *arg) {
    while (1) {
        printf("Angles(deg): J1=%.1f  J2=%.1f  J3=%.1f  J4=%.1f  J5=%.1f  J6=%.1f\n",
               sim_angle_deg[0], sim_angle_deg[1], sim_angle_deg[2],
               sim_angle_deg[3], sim_angle_deg[4], sim_angle_deg[5]);
        printf("Duty(cnts):  J1=%lu  J2=%lu  J3=%lu  J4=%lu  J5=%lu  J6=%lu\n",
               angle_to_duty(sim_angle_deg[0]), angle_to_duty(sim_angle_deg[1]),
               angle_to_duty(sim_angle_deg[2]), angle_to_duty(sim_angle_deg[3]),
               angle_to_duty(sim_angle_deg[4]), angle_to_duty(sim_angle_deg[5]));
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "=== MT6816 Encoder PWM Simulator ===");
    ESP_LOGI(TAG, "PWM: %d Hz, 12-bit (4096 steps)", PWM_FREQ_HZ);
    ESP_LOGI(TAG, "Wiring: ESP32 GPIO → Teensy QuadTimer pin");

    pwm_init();

    xTaskCreate(sim_task, "sim",  4096, NULL, 10, NULL);
    xTaskCreate(log_task, "log",  2048, NULL,  5, NULL);
}
