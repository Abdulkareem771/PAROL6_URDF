/*
 * PAROL6 Serial Communication Implementation (Teensy 4.1)
 *
 * Non-blocking USB serial protocol handler.
 * Uses native USB 2.0 High Speed (480 Mbps) — no baud rate limit.
 * Serial.begin() is ignored on Teensy USB; it always runs at full speed.
 *
 * Protocol (same as ESP32):
 *   RX: <SEQ,p0,v0,p1,v1,...,p5,v5>\n
 *   TX: <ACK,seq,p0,v0,p1,v1,...>\n
 */

#include "serial_comm.h"
#include "control.h"
#include <stdlib.h>   // strtoul, strtof
#include <string.h>   // strchr
#include <stdio.h>    // snprintf

// ============================================================================
// STATE
// ============================================================================

static char   rx_buf[COMMAND_BUFFER_SIZE];
static int    rx_pos = 0;
static uint32_t tx_seq = 0;
static uint32_t last_rx_seq = 0;

// ============================================================================
// INIT
// ============================================================================

void serialCommInit() {
  // Native USB — baud rate is ignored, always 480 Mbps
  Serial.begin(SERIAL_BAUD);
  rx_pos = 0;
}

// ============================================================================
// COMMAND PARSING
// ============================================================================

static void parseCommand(const char* cmd) {
  // Format: <SEQ,p0,v0,p1,v1,...,p5,v5>
  if (cmd[0] != '<') return;
  cmd++;

  char* end_ptr;
  uint32_t seq = strtoul(cmd, &end_ptr, 10);
  last_rx_seq = seq;
  cmd = end_ptr;

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    if (*cmd != ',') return;
    cmd++;
    float pos = strtof(cmd, &end_ptr);
    cmd = end_ptr;

    if (*cmd != ',') return;
    cmd++;
    float vel = strtof(cmd, &end_ptr);
    cmd = end_ptr;

    controlSetCommand(i, pos, vel);
  }
}

// ============================================================================
// INCOMING (non-blocking)
// ============================================================================

void serialCommProcessIncoming() {
  while (Serial.available()) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (rx_pos > 0) {
        rx_buf[rx_pos] = '\0';
        parseCommand(rx_buf);
        rx_pos = 0;
      }
    } else if (rx_pos < COMMAND_BUFFER_SIZE - 1) {
      rx_buf[rx_pos++] = c;
    } else {
      rx_pos = 0;   // overflow → reset
    }
  }
}

// ============================================================================
// FEEDBACK
// ============================================================================

void serialCommSendFeedback() {
  // Format: <ACK,seq,p0,v0,p1,v1,...>\n
  char buf[COMMAND_BUFFER_SIZE];
  int pos = 0;

  pos += snprintf(buf + pos, sizeof(buf) - pos, "<ACK,%lu",
                  (unsigned long)tx_seq++);

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    const JointState* st = controlGetState(i);
    if (st) {
      pos += snprintf(buf + pos, sizeof(buf) - pos,
                      ",%.3f,%.3f",
                      st->actual_position,
                      st->velocity_command);
    } else {
      pos += snprintf(buf + pos, sizeof(buf) - pos, ",0.000,0.000");
    }
  }

  pos += snprintf(buf + pos, sizeof(buf) - pos, ">\n");
  Serial.write(buf, pos);
}
