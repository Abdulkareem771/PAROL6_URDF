// main/main.c
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "driver/uart.h"

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/string.h>
#include <uxr/client/transport.h>
#include <rmw_microxrcedds_c/config.h>
#include <rmw_microros/rmw_microros.h>

// include the transport header
#include "esp32_serial_transport.h"

#define LED_GPIO GPIO_NUM_2
static const char *TAG = "APP";

// LED modes: 0=OFF, 1=ON, 2=BLINK
volatile int led_mode = 0;
volatile TickType_t last_toggle = 0;

rcl_publisher_t publisher;
rcl_subscription_t subscriber;
rclc_executor_t executor;
rclc_support_t support;
rcl_node_t node;

// Message storage for subscriber
static std_msgs__msg__String sub_msg;

void publish_status(const char *s)
{
  std_msgs__msg__String msg;
  // careful: rcl expects allocated sequence - but for simple publish we can set data pointer
  msg.data.data = (char *)s;
  msg.data.size = strlen(s);
  msg.data.capacity = msg.data.size + 1;
  rcl_ret_t ret = rcl_publish(&publisher, &msg, NULL);
  if (ret != RCL_RET_OK) {
    ESP_LOGE(TAG, "Failed to publish status: %ld", (long) ret);

  } else {
    ESP_LOGI(TAG, "Published: %s", s);
  }
}

void subscription_callback(const void *msgin)
{
  const std_msgs__msg__String *msg = (const std_msgs__msg__String *)msgin;
  if (msg->data.size == 0) return;

  char buf[64];
  size_t n = (msg->data.size < sizeof(buf)-1) ? msg->data.size : sizeof(buf)-1;
  memcpy(buf, msg->data.data, n);
  buf[n] = '\0';

  // uppercase
  for (size_t i=0;i<n;i++){
    if (buf[i] >= 'a' && buf[i] <= 'z') buf[i] = buf[i] - 'a' + 'A';
  }

  ESP_LOGI(TAG, "Received command: %s", buf);

  if (strcmp(buf, "ON") == 0) {
    led_mode = 1;
    gpio_set_level(LED_GPIO, 1);
    publish_status("OK:ON");
  } else if (strcmp(buf, "OFF") == 0) {
    led_mode = 0;
    gpio_set_level(LED_GPIO, 0);
    publish_status("OK:OFF");
  } else if (strcmp(buf, "BLINK") == 0) {
    led_mode = 2;
    last_toggle = xTaskGetTickCount();
    publish_status("OK:BLINK");
  } else {
    char err[80];
    snprintf(err, sizeof(err), "ERR:UNKNOWN:%s", buf);
    publish_status(err);
  }
}

void app_main(void)
{
  esp_log_level_set("*", ESP_LOG_INFO);
  ESP_LOGI(TAG, "ESP32 ready");

  // init LED pin
  gpio_reset_pin(LED_GPIO);
  gpio_set_direction(LED_GPIO, GPIO_MODE_OUTPUT);
  gpio_set_level(LED_GPIO, 0);

  // ---- Register custom UART transport BEFORE microROS initialization ----
#if defined(RMW_UXRCE_TRANSPORT_CUSTOM)
  static size_t uart_port = UART_NUM_0; // UART port for micro-ROS transport
  
  rmw_uros_set_custom_transport(
      true,                         // framing enabled
      (void *) &uart_port,          // args passed to transport functions
      esp32_serial_open,
      esp32_serial_close,
      esp32_serial_write,
      esp32_serial_read
  );
  
  ESP_LOGI(TAG, "Custom UART transport registered successfully");
#else
#error "micro-ROS transport misconfigured - need app-colcon.meta with RMW_UXRCE_TRANSPORT=custom"
#endif  // RMW_UXRCE_TRANSPORT_CUSTOM

  // ------------- micro-ROS initialization -------------
  rcl_allocator_t allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);

  rcl_node_options_t node_ops = rcl_node_get_default_options();
  (void) node_ops;
  rclc_node_init_default(&node, "esp32_microros_node", "", &support);

  // Publisher (status)
  rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/parol6/esp32/led/status"
  );

  // Subscriber (commands)
  rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/parol6/esp32/led/command"
  );

  // Initialize message storage for subscriber
  // We need to allocate memory for the string data
  static char msg_buffer[100];
  sub_msg.data.data = msg_buffer;
  sub_msg.data.capacity = sizeof(msg_buffer);
  sub_msg.data.size = 0;

  rclc_executor_init(&executor, &support.context, 1, &allocator);
  // Attach the real storage that callback uses
  rclc_executor_add_subscription(
    &executor,
    &subscriber,
    &sub_msg, // <--- Pass the allocated message struct here!
    &subscription_callback,
    ON_NEW_DATA
  );

  ESP_LOGI(TAG, "Entering main loop");

  // Main loop: run executor and handle blink
  const TickType_t tick_period = pdMS_TO_TICKS(50);
  while (1) {
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(50));
    if (led_mode == 2) {
      TickType_t now = xTaskGetTickCount();
      if ((now - last_toggle) >= pdMS_TO_TICKS(500)) {
        static int led_state = 0;
        led_state = !led_state;
        gpio_set_level(LED_GPIO, led_state);
        last_toggle = now;
      }
    }
    vTaskDelay(tick_period);
  }
}
