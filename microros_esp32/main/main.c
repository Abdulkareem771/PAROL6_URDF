#include <stdio.h>

/* FreeRTOS */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* micro-ROS */
#include <std_msgs/msg/int32.h>
#include <rmw_microros/rmw_microros.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>

/* Project */
#include "wifi_sta.h"
#include "esp32_wifi_transport.h"

#include "esp_log.h"

static const char *TAG = "MAIN";

void app_main(void)
{
#ifdef CONFIG_MICRO_ROS_ESP_NETIF_WLAN

    /* WiFi */
    wifi_init_sta();

    /* micro-ROS custom WiFi transport */
    rmw_uros_set_custom_transport(
        true,
        NULL,
        esp32_wifi_open,
        esp32_wifi_close,
        esp32_wifi_write,
        esp32_wifi_read);
#endif

    /* =====================================================
     * WAIT FOR micro-ROS AGENT
     * ===================================================== */
    while (rmw_uros_ping_agent(1000, 5) != RMW_RET_OK) {
        ESP_LOGW(TAG, "Waiting for micro-ROS agent...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(TAG, "micro-ROS agent connected");

    /* =====================================================
     * micro-ROS NODE
     * ===================================================== */
    rcl_allocator_t allocator = rcl_get_default_allocator();
    rclc_support_t support;
    rclc_support_init(&support, 0, NULL, &allocator);

    rcl_node_t node;
    rclc_node_init_default(&node, "esp32_node", "", &support);

    rcl_publisher_t publisher;
    rclc_publisher_init_default(
        &publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
        "esp32/chatter");

    std_msgs__msg__Int32 msg;
    msg.data = 0;

    /* =====================================================
     * PUBLISH LOOP
     * ===================================================== */
    while (1) {
        rcl_publish(&publisher, &msg, NULL);
        ESP_LOGI(TAG, "Published: %ld", msg.data);
        msg.data++;
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

