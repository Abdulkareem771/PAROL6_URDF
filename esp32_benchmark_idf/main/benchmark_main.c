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
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (256)
#define TAG "BENCHMARK"

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
    
    // Send ACK
    printf("<ACK,%lu,%lld>\n", seq, timestamp_us);
    
    // Log received data
    ESP_LOGI(TAG, "SEQ:%lu J:[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]",
             seq, joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]);
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
    printf("READY: ESP32_BENCHMARK_V2\n");
    
    // Create UART task
    xTaskCreate(uart_task, "uart_task", 4096, NULL, 10, NULL);
}
