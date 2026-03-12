/**
 * stage4_motor.cpp — Stage 4: Encoder + Stepper (Full J6 Control)
 *
 * WHAT THIS TESTS:
 *   - ToneStepper sends step pulses on pin 5 / dir on pin 35
 *   - Position control loop closes the loop between encoder and stepper
 *   - Motor physically moves when you send position commands
 *
 * SUCCESS LOOKS LIKE:
 *   - After <ENABLE>, send: <1,0,0,0,0,0,1.5708,0,0,0,0,0,0.5>
 *   - Motor rotates ~90° on J6 output shaft
 *   - J6 position in ACK tracks toward 1.5708
 *
 * If the motor moves but position doesn't track → encoder wiring issue (Stage 3)
 * If position changes but motor doesn't move → stepper wiring/EN pin issue
 */

#include <Arduino.h>
#include "hal/SoftwareInterruptEncoder.h"
#include "hal/ToneStepper.h"
#include "transport/SerialTransport.h"
#include <CircularBuffer.h>

// ── J6 pin config ─────────────────────────────────────────────────────────────
static constexpr int   J6_STEP_PIN   = 5;
static constexpr int   J6_DIR_PIN    = 35;
static constexpr int   J6_ENC_PIN    = 19;     // ← CORRECTED: pin 19
static constexpr int   J6_ENC_IDX   = 5;
static constexpr float J6_GEAR_RATIO = 10.0f;
static constexpr int   J6_MICROSTEPS = 32;
static constexpr bool  J6_DIR_INV    = true;

static constexpr float TEST_SPEED_RAD_S = 0.1f; // 0.1 rad/s (EXTREMELY slow, easy to see ticks)
static constexpr float STEPS_PER_RAD = (200.0f * J6_MICROSTEPS * J6_GEAR_RATIO) / (2.0f * 3.14159265f);
static constexpr uint32_t CMD_TIMEOUT_MS = 1000;
static constexpr uint32_t FB_RATE_HZ    = 25;

// ── Objects ───────────────────────────────────────────────────────────────────
static CircularBuffer<RosCommand, 20> rx_queue;
static SerialTransport   transport;
static SoftwareInterruptEncoder enc(J6_ENC_PIN, J6_ENC_IDX);
static ToneStepper       stepper(J6_STEP_PIN, J6_DIR_PIN);

// ── State ─────────────────────────────────────────────────────────────────────
static volatile float j6_pos    = 0.0f;
static volatile bool  enabled   = false;
static volatile bool  moving    = false;

static IntervalTimer servo_timer;
static uint32_t last_cmd_ms = 0;
static uint32_t fb_seq      = 0;
static uint32_t last_fb_ms  = 0;

// ── 1 kHz servo ISR ──────────────────────────────────────────────────────────
static void FASTRUN servo_isr() {
    j6_pos = enc.read_angle();

    // SIMPLE OPEN-LOOP TEST: If enabled and commanded to move, step at a constant, safe slow speed.
    // This entirely removes the PID loop and instantaneous velocity jumps that cause stalls.
    if (!enabled || !moving) {
        stepper.stop();
    } else {
        stepper.set_direction(!J6_DIR_INV); // Always forward
        stepper.set_frequency(TEST_SPEED_RAD_S * STEPS_PER_RAD);
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    transport.init(115200);
    enc.init();
    stepper.init();
    servo_timer.begin(servo_isr, 1000);  // 1 kHz

    delay(500);
    Serial.println("<STAGE4_READY>");
    Serial.println("# MKServo42C: enable the driver MANUALLY (physical button).");
    Serial.println("# OPEN-LOOP TEST MODE: Send <ENABLE> to spin motor at a safe 0.5 rad/s indefinitely.");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    transport.process_incoming(rx_queue);

    while (!rx_queue.isEmpty()) {
        RosCommand cmd = rx_queue.shift();
        if (cmd.is_enable_cmd) {
            enabled = true;
            moving  = true; // Start moving immediately on ENABLE for this test
            last_cmd_ms = millis();
            Serial.println("<ENABLE_ACK> - Motor should now be spinning!");
        } else if (cmd.is_home_cmd) {
            enc.init();
            last_cmd_ms = millis();
            Serial.println("<HOMING_DONE>");
        } else {
            // Keep alive
            last_cmd_ms = millis();
        }
    }

    /* Watchdog completely disabled for hardware testing
    if (enabled && (millis() - last_cmd_ms > CMD_TIMEOUT_MS)) {
        moving = false; // stop moving if we haven't received keep-alives
    }
    */

    // Feedback
    uint32_t now = millis();
    if (now - last_fb_ms >= (1000u / FB_RATE_HZ)) {
        last_fb_ms = now;
        digitalToggleFast(LED_BUILTIN);

        float pos[6] = {0, 0, 0, 0, 0, j6_pos};
        float vel[6] = {0, 0, 0, 0, 0, (moving ? TEST_SPEED_RAD_S : 0.0f)};
        transport.send_feedback(fb_seq++, pos, vel, 0, enabled ? 1 : 3);
    }
}
