/*
 * PAROL6 Serial Communication - UART Protocol Handler
 * 
 * Non-blocking serial communication with ROS2
 */

#ifndef SERIAL_COMM_H
#define SERIAL_COMM_H

#include <Arduino.h>
#include "config.h"

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

// Initialize serial communication
void serialCommInit();

// Process incoming commands (non-blocking, called from serial task)
void serialCommProcessIncoming();

// Send feedback packet (called from serial task at 50 Hz)
void serialCommSendFeedback();

#endif // SERIAL_COMM_H
