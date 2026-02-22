/*
 * PAROL6 Serial Communication — UART Protocol Handler (Teensy 4.1)
 *
 * Non-blocking serial communication with ROS2.
 * Uses native USB (480 Mbps) — no baud rate limitation.
 */

#ifndef SERIAL_COMM_H
#define SERIAL_COMM_H

#include <Arduino.h>
#include "config.h"

// Initialize serial communication (USB)
void serialCommInit();

// Process incoming commands (non-blocking, called from main loop)
void serialCommProcessIncoming();

// Send feedback packet (called from main loop at 50 Hz)
void serialCommSendFeedback();

#endif // SERIAL_COMM_H
