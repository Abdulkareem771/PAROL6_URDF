#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (1024)
#define TAG "FEEDBACK_FIRMWARE"

// Current joint positions (simulated - initially 0)
static float joint_positions[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static uint32_t last_seq = 0;

void setup_uart(void) {
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    
    uart_param_config(UART_NUM, &uart_config);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
    
    ESP_LOGI(TAG, "UART configured at 115200 baud");
}

void send_feedback(uint32_t seq) {
    char response[128];
    snprintf(response, sizeof(response),
             "<ACK,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f>\n",
             seq,
             joint_positions[0], joint_positions[1], joint_positions[2],
             joint_positions[3], joint_positions[4], joint_positions[5]);
    
    uart_write_bytes(UART_NUM, response, strlen(response));
}

void parse_command(const char *cmd, int len) {
    // Expected format: <SEQ,J1,J2,J3,J4,J5,J6>
    // Skip '<' and find first comma
    char buffer[128];
    if (len >= sizeof(buffer)) len = sizeof(buffer) - 1;
    memcpy(buffer, cmd, len);
    buffer[len] = '\0';
    
    // Remove '<' and '>'
    char *start = buffer;
    if (*start == '<') start++;
    char *end = strchr(start, '>');
    if (end) *end = '\0';
    
    // Parse sequence number
    char *token = strtok(start, ",");
    if (token == NULL) return;
    
    uint32_t seq = (uint32_t)atol(token);
    last_seq = seq;
    
    // Parse 6 joint positions
    int joint_idx = 0;
    while ((token = strtok(NULL, ",")) != NULL && joint_idx < 6) {
        joint_positions[joint_idx] = atof(token);
        joint_idx++;
    }
    
    // Only send ACK if we got valid data
    if (joint_idx == 6) {
        send_feedback(seq);
    }
}

void uart_task(void *arg) {
    uint8_t data[BUF_SIZE];
    static char line_buffer[128];
    static int line_idx = 0;
    
    ESP_LOGI(TAG, "READY: ESP32_FEEDBACK_V1");
    
    while (1) {
        int len = uart_read_bytes(UART_NUM, data, BUF_SIZE, 20 / portTICK_PERIOD_MS);
        
        for (int i = 0; i < len; i++) {
            char c = data[i];
            
            if (c == '<') {
                // Start of new command
                line_idx = 0;
                line_buffer[line_idx++] = c;
            } else if (c == '>') {
                // End of command
                line_buffer[line_idx++] = c;
                line_buffer[line_idx] = '\0';
                
                // Parse and respond
                parse_command(line_buffer, line_idx);
                line_idx = 0;
            } else if (line_idx > 0 && line_idx < sizeof(line_buffer) - 1) {
                // Middle of command
                line_buffer[line_idx++] = c;
            }
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting PAROL6 Feedback Firmware");
    
    setup_uart();
    
    xTaskCreate(uart_task, "uart_task", 4096, NULL, 10, NULL);
}
