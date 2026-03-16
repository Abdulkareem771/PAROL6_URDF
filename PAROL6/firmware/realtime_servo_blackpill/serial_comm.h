/*
 * PAROL6 Serial Communication — USB CDC (STM32Duino)
 *
 * Uses STM32Duino's built-in Serial (USB CDC at 12 Mbps).
 * Same protocol as Teensy/ESP32.
 */

#ifndef SERIAL_COMM_H
#define SERIAL_COMM_H

#include "config.h"

void serialCommInit(void);
void serialCommProcessIncoming(void);
void serialCommSendFeedback(void);

#endif // SERIAL_COMM_H
