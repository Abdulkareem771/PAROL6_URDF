/**
 * ESP32 Benchmark Firmware (ESP-IDF Version)
 * 
 * Tests ROS-ESP32 communication integrity
 * 
 * Protocol:
 * RX: <SEQ,J1,J2,J3,J4,J5,J6>
 * TX: <ACK,SEQ,TIMESTAMP_US>
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h> // Added for fmod
#include "freertos/FreeRTOS.h" // Restored FreeRTOS base include
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "driver/ledc.h" // Added for PWM generation
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (256)
#define TAG "BENCHMARK"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// ESP32 PWM Output Pins for Phase 1.5 testing
// These will be wired to the Teensy 4.1 interrupt pins
const int PWM_PINS[6] = { 18, 19, 21, 22, 23, 25 }; 
#define PWM_FREQ_HZ 1000 // Match typical MT6816 frequency
#define PWM_RESOLUTION LEDC_TIMER_10_BIT // 10-bit resolution (0-1023)
#define PWM_MAX_DUTY 1023

// Stats tracking
typedef struct {
    uint32_t packets_received;
    uint32_t packets_lost;
    uint32_t last_seq_num;
} comm_stats_t;

static comm_stats_t stats = {0};

// Parse incoming message: <SEQ,J1,J2,J3,J4,J5,J6>
static bool parse_message(const char* msg, uint32_t* seq, float joints[6]) {
    // Remove < and > markers
    char buffer[256];
    strncpy(buffer, msg, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    
    // Remove angle brackets
    char* start = strchr(buffer, '<');
    char* end = strchr(buffer, '>');
    
    if (!start || !end || end <= start) {
        return false;
    }
    
    *start = '\0';
    start++;
    *end = '\0';
    
    // Parse values
    char* token = strtok(start, ",");
    if (!token) return false;
    
    *seq = (uint32_t)atoi(token);
    
    for (int i = 0; i < 6; i++) {
        token = strtok(NULL, ",");
        if (!token) return false;
        joints[i] = atof(token);
    }
    
    return true;
}

static void process_command(const char* msg) {
    uint32_t seq;
    float joints[6];
    int64_t timestamp_us = esp_timer_get_time();
    
    if (!parse_message(msg, &seq, joints)) {
        ESP_LOGW(TAG, "Invalid message format");
        return;
    }
    
    // Detect packet loss
    if (stats.packets_received > 0) {
        uint32_t expected = stats.last_seq_num + 1;
        if (seq != expected) {
            stats.packets_lost += (seq - expected);
            ESP_LOGW(TAG, "Packet loss detected! Expected: %lu, Got: %lu", expected, seq);
        }
    }
    
    stats.last_seq_num = seq;
    stats.packets_received++;
    
    // Update PWM signals for the 6 joints
    // Map radians (0 to 2PI) to duty cycle (0 to 1023)
    for (int i = 0; i < 6; i++) {
        // Normalize joint angle (assuming input might be outside 0-2PI, wrap it)
        float angle = fmod(joints[i], 2.0f * PI);
        if (angle < 0) angle += 2.0f * PI;
        
        uint32_t duty = (uint32_t)((angle / (2.0f * PI)) * PWM_MAX_DUTY);
        
        // Inject tiny bit of noise (+/- 1 LSB) to simulate real hardware jitter
        int noise = (rand() % 3) - 1; // -1, 0, or 1
        int noisy_duty = duty + noise;
        if (noisy_duty < 0) noisy_duty = 0;
        if (noisy_duty > PWM_MAX_DUTY) noisy_duty = PWM_MAX_DUTY;
        
        ledc_set_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)i, noisy_duty);
        ledc_update_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)i);
    }
    
    // Send ACK
    printf("<ACK,%lu,%lld>\n", seq, timestamp_us);
    
    // Log occasionally to avoid spamming
    if (seq % 100 == 0) {
        ESP_LOGI(TAG, "SEQ:%lu PWM Duty[0]: %lu", seq, (uint32_t)((fmod(joints[0], 2.0f * PI) / (2.0f * PI)) * PWM_MAX_DUTY));
    }
}

static void uart_task(void *arg) {
    uint8_t* data = (uint8_t*) malloc(BUF_SIZE);
    char line_buffer[BUF_SIZE];
    int line_pos = 0;
    
    while (1) {
        int len = uart_read_bytes(UART_NUM, data, BUF_SIZE - 1, 20 / portTICK_PERIOD_MS);
        
        if (len > 0) {
            for (int i = 0; i < len; i++) {
                char c = (char)data[i];
                
                if (c == '\n' || c == '\r') {
                    if (line_pos > 0) {
                        line_buffer[line_pos] = '\0';
                        
                        // Process complete line
                        if (line_buffer[0] == '<') {
                            process_command(line_buffer);
                        } else if (strcmp(line_buffer, "STATS") == 0) {
                            // Print statistics
                            printf("\n=== Statistics ===\n");
                            printf("Packets Received: %lu\n", stats.packets_received);
                            printf("Packets Lost: %lu\n", stats.packets_lost);
                            if (stats.packets_received > 0) {
                                float loss_rate = (stats.packets_lost * 100.0f) / 
                                                 (stats.packets_received + stats.packets_lost);
                                printf("Loss Rate: %.2f%%\n", loss_rate);
                            }
                            printf("==================\n\n");
                        }
                        
                        line_pos = 0;
                    }
                } else if (line_pos < BUF_SIZE - 1) {
                    line_buffer[line_pos++] = c;
                }
            }
        }
    }
}

static void init_pwm() {
    // 1. Configure the PWM Timer (1 kHz, 10-bit)
    ledc_timer_config_t timer_conf = {
        .speed_mode       = LEDC_HIGH_SPEED_MODE,
        .duty_resolution  = PWM_RESOLUTION,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = PWM_FREQ_HZ,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_conf));

    // 2. Configure 6 independent PWM channels
    for (int i = 0; i < 6; i++) {
        ledc_channel_config_t ch_conf = {
            .gpio_num       = PWM_PINS[i],
            .speed_mode     = LEDC_HIGH_SPEED_MODE,
            .channel        = (ledc_channel_t)i,
            .intr_type      = LEDC_INTR_DISABLE,
            .timer_sel      = LEDC_TIMER_0,
            .duty           = 0, // Start at 0 Rads
            .hpoint         = 0
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch_conf));
    }
    ESP_LOGI(TAG, "Initialized 6 PWM channels at 1 kHz for Phase 1.5 testing.");
}

void app_main(void) {
    // Configure UART
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    
    printf("\n\n");
    printf("========================================\n");
    printf("  ESP32 Benchmark Firmware (ESP-IDF)\n");
    printf("========================================\n");
    printf("Ready to receive commands...\n");
    printf("Send 'STATS' to view statistics\n\n");
    printf("READY: ESP32_PHASE1.5_PWM_INJECTOR\n");
    
    // Initialize PWM Hardware
    init_pwm();
    
    // Create UART task
    xTaskCreate(uart_task, "uart_task", 4096, NULL, 10, NULL);
}
