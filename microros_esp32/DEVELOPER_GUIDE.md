# Micro-ROS ESP32 Developer Guide

This guide explains how to extend this project to create your own micro-ROS applications on ESP32.

## 1. Project Structure

The main logic lives in `main/main.c`. This is where you define your ROS 2 nodes, publishers, and subscribers.

```
main/
├── main.c                  # <--- YOUR CODE GOES HERE
├── esp32_serial_transport.c # Handles UART communication (don't touch)
└── CMakeLists.txt          # Build configuration
```

## 2. Core Concepts

Micro-ROS on ESP32 uses the **rclc** (ROS Client Library for C) API. It's slightly different from C++ or Python ROS 2 but follows the same concepts.

### Key Objects
- **Support (`rclc_support_t`)**: Handles the communication layer.
- **Node (`rcl_node_t`)**: Your ROS 2 node.
- **Executor (`rclc_executor_t`)**: Manages callbacks (like `spin()` in ROS 2).
- **Allocator (`rcl_allocator_t`)**: Manages memory.

## 3. How to Add a New Publisher

To publish data (e.g., sensor readings), follow these steps in `main.c`:

### Step A: Include the Message Type
Find the message type you need. Standard messages are in `std_msgs`, `sensor_msgs`, etc.
```c
#include <std_msgs/msg/int32.h>
```

### Step B: Declare the Publisher
Add a global variable for the publisher and the message.
```c
rcl_publisher_t my_pub;
std_msgs__msg__Int32 my_msg;
```

### Step C: Initialize the Publisher
Inside `app_main()`, after node initialization:
```c
rclc_publisher_init_default(
    &my_pub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
    "/my/topic/name"
);
```

### Step D: Publish Data
You can publish data in a timer callback or in the main loop.
```c
my_msg.data = 42;
rcl_publish(&my_pub, &my_msg, NULL);
```

## 4. How to Add a New Subscriber

To receive data (e.g., motor commands), follow these steps:

### Step A: Include Message Type
```c
#include <std_msgs/msg/int32.h>
```

### Step B: Declare Subscriber and Storage
You need a subscriber object AND a place to store the incoming message.
```c
rcl_subscription_t my_sub;
std_msgs__msg__Int32 my_sub_msg; // Storage for incoming data
```

### Step C: Define the Callback
Create a function that runs when a message arrives.
```c
void my_callback(const void * msgin)
{
    const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msgin;
    printf("Received: %d\n", msg->data);
}
```

### Step D: Initialize Subscriber
Inside `app_main()`:
```c
rclc_subscription_init_default(
    &my_sub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
    "/my/input/topic"
);
```

### Step E: Add to Executor
**Crucial Step**: You must tell the executor to listen to this subscriber.
```c
// IMPORTANT: Pass the address of your storage message (&my_sub_msg)
rclc_executor_add_subscription(
    &executor,
    &my_sub,
    &my_sub_msg,
    &my_callback,
    ON_NEW_DATA
);
```

## 5. How to Add a Timer (Periodic Tasks)

If you want to publish data at a fixed rate (e.g., 10Hz), use a timer.

### Step A: Declare Timer
```c
rcl_timer_t my_timer;
```

### Step B: Define Timer Callback
```c
void timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    // Do work here (read sensor, publish)
    // ...
}
```

### Step C: Initialize Timer
Inside `app_main()`:
```c
const unsigned int timer_timeout = 100; // 100ms = 10Hz
rclc_timer_init_default(
    &my_timer,
    &support,
    RCL_MS_TO_NS(timer_timeout),
    timer_callback
);
```

### Step D: Add to Executor
You must increase the executor handle count (see below) and add the timer.
```c
rclc_executor_add_timer(&executor, &my_timer);
```

## 6. Important: Executor Configuration

The executor needs to know how many "handles" (subscribers + timers + services) it needs to manage.

In `app_main()`, look for `rclc_executor_init`:

```c
// Change the number '1' to the total number of handles
// Example: 1 subscriber + 1 timer = 2 handles
rclc_executor_init(&executor, &support.context, 2, &allocator);
```

If you forget to increase this number, your new subscriber/timer won't work!

## 7. Example: Adding a Button Publisher

Here is a complete snippet to add a button publisher to the existing code.

1. **Add Include**: `#include <std_msgs/msg/bool.h>`
2. **Global Vars**:
   ```c
   rcl_publisher_t button_pub;
   std_msgs__msg__Bool button_msg;
   #define BUTTON_GPIO GPIO_NUM_0 // Boot button
   ```
3. **Init in `app_main`**:
   ```c
   gpio_set_direction(BUTTON_GPIO, GPIO_MODE_INPUT);
   
   rclc_publisher_init_default(
       &button_pub,
       &node,
       ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
       "/parol6/esp32/button"
   );
   ```
4. **Loop Logic**:
   ```c
   // Inside while(1) loop
   button_msg.data = !gpio_get_level(BUTTON_GPIO); // Active low
   rcl_publish(&button_pub, &button_msg, NULL);
   ```

## 8. Troubleshooting New Code

- **Crash on Startup**: Did you increase the executor handle count?
- **Crash on Message**: Did you pass `&msg_storage` to `rclc_executor_add_subscription` (not NULL)?
- **Build Error**: Did you include the header file for the message type?
- **No Data**: Is the executor spinning? (`rclc_executor_spin_some`)

Happy Coding!
