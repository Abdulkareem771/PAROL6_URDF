/*
 * PAROL6 Real-Time Servo Control - Main
 * 
 * FreeRTOS task orchestration and initialization
 */

#include "config.h"
#include "motor.h"
#include "control.h"
#include "serial_comm.h"

// ============================================================================
// FREERTOS TASK HANDLES
// ============================================================================

TaskHandle_t controlTaskHandle = NULL;
TaskHandle_t serialTaskHandle = NULL;

// ============================================================================
// CONTROL TASK (500 Hz)
// ============================================================================

void controlTask(void* parameter) {
  TickType_t last_wake_time = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(CONTROL_PERIOD_MS);
  
  while (true) {
    // Run control loop
    controlUpdate();
    
    // Wait for next period (deterministic timing)
    vTaskDelayUntil(&last_wake_time, period);
  }
}

// ============================================================================
// SERIAL TASK (50 Hz)
// ============================================================================

void serialTask(void* parameter) {
  TickType_t last_wake_time = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(FEEDBACK_PERIOD_MS);
  
  while (true) {
    // Process incoming commands (non-blocking)
    serialCommProcessIncoming();
    
    // Send feedback
    serialCommSendFeedback();
    
    // Wait for next period
    vTaskDelayUntil(&last_wake_time, period);
  }
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize subsystems
  serialCommInit();
  motorsInit();
  controlInit();
  
  // Create FreeRTOS tasks
  xTaskCreate(
    controlTask,
    "Control",
    CONTROL_TASK_STACK_SIZE,
    NULL,
    CONTROL_TASK_PRIORITY,
    &controlTaskHandle
  );
  
  xTaskCreate(
    serialTask,
    "Serial",
    SERIAL_TASK_STACK_SIZE,
    NULL,
    SERIAL_TASK_PRIORITY,
    &serialTaskHandle
  );
  
  // Tasks are now running
}

// ============================================================================
// LOOP (Not used - FreeRTOS handles scheduling)
// ============================================================================

void loop() {
  // Empty - all work done in tasks
  vTaskDelay(portMAX_DELAY);
}
