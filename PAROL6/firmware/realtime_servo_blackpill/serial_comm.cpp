/*
 * PAROL6 Serial Communication — USB CDC (STM32Duino)
 *
 * Uses STM32Duino's Serial interface for USB CDC at 12 Mbps Full Speed.
 * Protocol identical to Teensy/ESP32 version.
 *
 * RX: <SEQ,p0,v0,p1,v1,...,p5,v5>\n
 * TX: <ACK,seq,p0,v0,p1,v1,...>\n
 *
 * When DEBUG_RAW_ENCODER is defined, also prints raw CCR1/CCR2 values.
 */

#include "serial_comm.h"
#include "control.h"
#include "encoder_capture.h"
#include <Arduino.h>
#include <stdlib.h>
#include <stdio.h>

// *** ENABLE THIS FOR RAW ENCODER REGISTER DEBUG ***
#define DEBUG_RAW_ENCODER

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
    Serial.begin(115200);  // Baud rate ignored for USB CDC, but API requires it
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
    char num[16];

    // Header: <ACK,seq
    pos += snprintf(buf + pos, sizeof(buf) - pos, "<ACK,%lu",
                    (unsigned long)tx_seq++);

    // Joint data: ,pos,vel for each joint
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        const JointState *st = controlGetState(i);
        float p = st ? st->actual_position : 0.0f;
        float v = st ? st->velocity_command : 0.0f;

        buf[pos++] = ',';
        dtostrf(p, 1, 3, num);
        int len = strlen(num);
        memcpy(buf + pos, num, len);
        pos += len;

        buf[pos++] = ',';
        dtostrf(v, 1, 3, num);
        len = strlen(num);
        memcpy(buf + pos, num, len);
        pos += len;
    }

    // Trailer: >\n
    buf[pos++] = '>';
    buf[pos++] = '\n';
    Serial.write((uint8_t*)buf, pos);

#ifdef DEBUG_RAW_ENCODER
    // Print raw timer CCR1/CCR2 for encoder-enabled joints.
    // If CCR2 drifts while motor is stopped → hardware/wiring issue.
    // If CCR2 is stable but decoded position drifts → firmware bug.
    char dbg[128];
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (!ENCODER_ENABLED[i]) continue;

        uint32_t per, ht;
        encoderReadCapture(i, &per, &ht);
        const JointState *st = controlGetState(i);

        int dlen = snprintf(dbg, sizeof(dbg), "[DBG J%d CCR1=%lu CCR2=%lu pos=",
                            i + 1, (unsigned long)per, (unsigned long)ht);
        dtostrf(st ? st->actual_position : 0.0f, 1, 3, num);
        int nlen = strlen(num);
        memcpy(dbg + dlen, num, nlen);
        dlen += nlen;
        dlen += snprintf(dbg + dlen, sizeof(dbg) - dlen, " arm=%d]\n",
                         controlIsArmed() ? 1 : 0);
        Serial.write((uint8_t*)dbg, dlen);
    }
#endif
}
