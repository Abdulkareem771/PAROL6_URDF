#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "driver/gpio.h"

#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/string.h>
#include <rmw_microros/rmw_microros.h>

#define LED_PIN GPIO_NUM_2
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){printf("Failed status on line %d: %d. Aborting.\n",__LINE__,(int)temp_rc);vTaskDelete(NULL);}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){printf("Failed status on line %d: %d. Continuing.\n",__LINE__,(int)temp_rc);}}

static const char *TAG = "APP";

typedef enum {
    LED_OFF,
    LED_ON,
    LED_BLINK
} led_mode_t;

static volatile led_mode_t current_led_mode = LED_OFF;

rcl_publisher_t publisher;
rcl_subscription_t subscriber;
std_msgs__msg__String msg_req;
std_msgs__msg__String msg_resp;

void led_task(void * arg) {
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    while (1) {
        if (current_led_mode == LED_ON) {
            gpio_set_level(LED_PIN, 1);
            vTaskDelay(pdMS_TO_TICKS(100));
        } else if (current_led_mode == LED_OFF) {
            gpio_set_level(LED_PIN, 0);
            vTaskDelay(pdMS_TO_TICKS(100));
        } else if (current_led_mode == LED_BLINK) {
            gpio_set_level(LED_PIN, 1);
            vTaskDelay(pdMS_TO_TICKS(500));
            // Check if mode changed during delay to be responsive
            if (current_led_mode == LED_BLINK) {
                gpio_set_level(LED_PIN, 0);
                vTaskDelay(pdMS_TO_TICKS(500));
            }
        }
    }
}

void subscription_callback(const void * msgin) {
    const std_msgs__msg__String * msg = (const std_msgs__msg__String *)msgin;
    
    if (msg->data.data == NULL) return;

    // Create a local buffer for processing
    char cmd_buf[64];
    strncpy(cmd_buf, msg->data.data, sizeof(cmd_buf) - 1);
    cmd_buf[sizeof(cmd_buf) - 1] = '\0';

    // Trim and uppercase
    char clean_cmd[64];
    int j = 0;
    for (int i = 0; cmd_buf[i] != '\0'; i++) {
        if (!isspace((unsigned char)cmd_buf[i])) {
            clean_cmd[j++] = toupper((unsigned char)cmd_buf[i]);
        }
    }
    clean_cmd[j] = '\0';

    ESP_LOGI(TAG, "Received command: %s", clean_cmd);

    char response_buf[128];
    
    if (strcmp(clean_cmd, "ON") == 0) {
        current_led_mode = LED_ON;
        snprintf(response_buf, sizeof(response_buf), "OK:ON");
    } else if (strcmp(clean_cmd, "OFF") == 0) {
        current_led_mode = LED_OFF;
        snprintf(response_buf, sizeof(response_buf), "OK:OFF");
    } else if (strcmp(clean_cmd, "BLINK") == 0) {
        current_led_mode = LED_BLINK;
        snprintf(response_buf, sizeof(response_buf), "OK:BLINK");
    } else {
        snprintf(response_buf, sizeof(response_buf), "ERR:UNKNOWN:%s", clean_cmd);
    }

    // Publish response
    msg_resp.data.data = response_buf;
    msg_resp.data.size = strlen(response_buf);
    msg_resp.data.capacity = msg_resp.data.size + 1;

    RCSOFTCHECK(rcl_publish(&publisher, &msg_resp, NULL));
    ESP_LOGI(TAG, "Published: %s", response_buf);
}

void micro_ros_task(void * arg) {
    rcl_allocator_t allocator = rcl_get_default_allocator();
    rclc_support_t support;

    // Wait for agent connection
    while (RMW_RET_OK != rmw_uros_ping_agent(1000, 1)) {
        ESP_LOGI(TAG, "Waiting for agent connection...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    ESP_LOGI(TAG, "Agent connected!");

    // Init options
    rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
    RCCHECK(rcl_init_options_init(&init_options, allocator));

    // Init support
    RCCHECK(rclc_support_init_with_options(&support, 0, NULL, &init_options, &allocator));

    // Create node
    rcl_node_t node;
    RCCHECK(rclc_node_init_default(&node, "esp32_led_node", "", &support));

    // Create publisher
    // Using robot namespace for better integration: /parol6/esp32/led/status
    RCCHECK(rclc_publisher_init_default(
        &publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
        "/parol6/esp32/led/status"));

    // Create subscriber
    // Using robot namespace for better integration: /parol6/esp32/led/command
    RCCHECK(rclc_subscription_init_default(
        &subscriber,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
        "/parol6/esp32/led/command"));

    // Create executor
    rclc_executor_t executor;
    RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
    RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg_req, &subscription_callback, ON_NEW_DATA));

    // Allocate memory for incoming message
    msg_req.data.capacity = 100;
    msg_req.data.data = (char*) malloc(msg_req.data.capacity * sizeof(char));
    msg_req.data.size = 0;

    ESP_LOGI(TAG, "Micro-ROS loop starting");

    while (1) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
        usleep(10000);
    }

    // Cleanup (unreachable in this loop)
    RCCHECK(rcl_subscription_fini(&subscriber, &node));
    RCCHECK(rcl_publisher_fini(&publisher, &node));
    RCCHECK(rcl_node_fini(&node));
    vTaskDelete(NULL);
}

void app_main(void) {
    ESP_LOGI(TAG, "ESP32 ready");

    xTaskCreate(led_task, "led_task", 2048, NULL, 5, NULL);
    
#if defined(CONFIG_MICRO_ROS_ESP_NETIF_WLAN) || defined(CONFIG_MICRO_ROS_ESP_NETIF_ENET)
    // If using WiFi/Ethernet, we need to init netif, but user requested Serial.
    // Serial transport usually doesn't need extra init in app_main if configured via Kconfig.
#endif

    xTaskCreate(micro_ros_task, "micro_ros_task", CONFIG_ESP_MAIN_TASK_STACK_SIZE, NULL, 5, NULL);
}
