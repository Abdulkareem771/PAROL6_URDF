/*
 * PAROL6 Serial Communication — USB CDC (STM32Duino)
 *
 * Uses STM32Duino's Serial interface for USB CDC at 12 Mbps Full Speed.
 * Protocol identical to Teensy/ESP32 version.
 *
 * RX: <SEQ,p0,v0,p1,v1,...,p5,v5>\n
 * TX: <ACK,seq,p0,v0,p1,v1,...>\n
 */

#include "serial_comm.h"
#include "control.h"
#include <Arduino.h>
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// STATE
// ============================================================================

static char     rx_buf[COMMAND_BUFFER_SIZE];
static int      rx_pos = 0;
static uint32_t tx_seq = 0;

// ============================================================================
// INIT
// ============================================================================

void serialCommInit(void)
{
    // STM32Duino: Serial is USB CDC when board is configured
    // with "USB support: CDC (generic Serial)" in Arduino IDE
    Serial.begin();  // USB CDC ignores baud rate
    rx_pos = 0;
}

// ============================================================================
// COMMAND PARSING (identical to Teensy)
// ============================================================================

static void parseCommand(const char *cmd)
{
    if (cmd[0] != '<') return;
    cmd++;

    char *end_ptr;
    uint32_t seq = strtoul(cmd, &end_ptr, 10);
    (void)seq;
    cmd = end_ptr;

    // Grace period: let ROS receive real feedback before accepting commands
    static uint16_t arm_grace_remaining = 0;

    if (!controlIsArmed()) {
        controlArm();
        arm_grace_remaining = 100;
        return;
    }

    if (arm_grace_remaining > 0) {
        arm_grace_remaining--;
        return;
    }

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
// INCOMING (non-blocking, called from main loop)
// ============================================================================

void serialCommProcessIncoming(void)
{
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (rx_pos > 0) {
                rx_buf[rx_pos] = '\0';
                parseCommand(rx_buf);
                rx_pos = 0;
            }
        } else if (rx_pos < COMMAND_BUFFER_SIZE - 1) {
            rx_buf[rx_pos++] = c;
        } else {
            rx_pos = 0;  // overflow → reset
        }
    }
}

// ============================================================================
// FEEDBACK (called from main loop at 50 Hz)
// ============================================================================

void serialCommSendFeedback(void)
{
    char buf[COMMAND_BUFFER_SIZE];
    int pos = 0;

    pos += snprintf(buf + pos, sizeof(buf) - pos, "<ACK,%lu",
                    (unsigned long)tx_seq++);

    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        const JointState *st = controlGetState(i);
        if (st) {
            pos += snprintf(buf + pos, sizeof(buf) - pos,
                            ",%.3f,%.3f",
                            (double)st->actual_position,
                            (double)st->velocity_command);
        } else {
            pos += snprintf(buf + pos, sizeof(buf) - pos, ",0.000,0.000");
        }
    }

    pos += snprintf(buf + pos, sizeof(buf) - pos, ">\n");

    Serial.write(buf, pos);
}
