/**
 * main_j6_test.cpp — Single-Joint (Joint 6) Integration Test Firmware
 *
 * Purpose
 * -------
 * Minimal firmware that drives ONLY Joint 6 via the same ACK packet protocol
 * the parol6_system.cpp ROS hardware driver expects.
 * Joints 1-5 are echoed at 0.0 so MoveIt stays happy.
 *
 * Flash with:
 *   pio run -e teensy41_j6_test --target upload
 *
 * Protocol (115200 USB CDC)
 * -------------------------
 * ROS → Teensy: <SEQ,p1..p6,v1..v6>
 * Also accepts: <ENABLE>  <HOME6>
 * Teensy → ROS: <ACK,seq,p1..p6,v1..v6,lim_state,state_byte>
 *   state_byte: 1=NOMINAL, 3=DISABLED
 */

#include <Arduino.h>
#include "transport/SerialTransport.h"
#include "hal/SoftwareInterruptEncoder.h"
#include "hal/ToneStepper.h"
#include "observer/AlphaBetaFilter.h"
#include "control/Interpolator.h"
#include <CircularBuffer.h>

// ──────────────────────────────────────────────────────────────────────────────
// Joint 6 hardware — axis index 5 in the PAROL6 schematic
// ──────────────────────────────────────────────────────────────────────────────
static constexpr int   J6_STEP_PIN    = 5;
static constexpr int   J6_DIR_PIN     = 35;
static constexpr int   J6_ENC_PIN     = 19;    // A-phase encoder (corrected from 18)
static constexpr int   J6_ENC_IDX    = 5;      // encoder axis slot
static constexpr float J6_GEAR_RATIO  = 10.0f;
static constexpr int   J6_MICROSTEPS  = 32;
static constexpr bool  J6_DIR_INVERT  = true;
static constexpr float J6_KP          = 2.5f;
static constexpr float J6_MAX_VEL     = 3.0f;  // rad/s at output shaft
static constexpr float J6_AB_ALPHA    = 0.15f;
static constexpr float J6_AB_BETA     = 0.008f;
static constexpr float LOOP_DT        = 0.001f; // 1 kHz ISR
static constexpr float STEPS_PER_RAD  = (200.0f * J6_MICROSTEPS * J6_GEAR_RATIO) / (2.0f * M_PI);
static constexpr uint32_t UART_BAUD   = 115200;
static constexpr uint32_t FB_RATE_HZ  = 25;
static constexpr uint32_t CMD_TIMEOUT_MS = 500;

// ──────────────────────────────────────────────────────────────────────────────
// Globals
// ──────────────────────────────────────────────────────────────────────────────
CircularBuffer<RosCommand, 20> rx_queue;
SerialTransport  transport;

SoftwareInterruptEncoder encoder_j6(J6_ENC_PIN, J6_ENC_IDX);
ToneStepper              stepper_j6(J6_STEP_PIN, J6_DIR_PIN);
AlphaBetaFilter          observer_j6(J6_AB_ALPHA, J6_AB_BETA, LOOP_DT);
LinearInterpolator       interp_j6;

volatile float j6_pos_rad    = 0.0f;
volatile float j6_vel_rad_s  = 0.0f;
volatile float j6_target     = 0.0f;
volatile bool  enabled       = false;
volatile bool  estop          = false;

static uint32_t fb_seq       = 0;
static uint32_t last_cmd_ms  = 0;

// ──────────────────────────────────────────────────────────────────────────────
// 1 kHz servo ISR
// ──────────────────────────────────────────────────────────────────────────────
IntervalTimer servo_timer;

static void FASTRUN servo_isr() {
    // 1) Read encoder
    float raw_rad = encoder_j6.read_angle();

    // 2) Update observer
    observer_j6.update(raw_rad);
    j6_pos_rad   = observer_j6.get_position();
    j6_vel_rad_s = observer_j6.get_velocity();

    // 3) PD control
    float err     = j6_target - j6_pos_rad;
    float out_vel = constrain(J6_KP * err, -J6_MAX_VEL, J6_MAX_VEL);

    // 4) Command stepper
    if (!enabled || estop) {
        stepper_j6.stop();
    } else {
        bool dir = (out_vel >= 0.0f) ^ J6_DIR_INVERT;
        stepper_j6.set_direction(dir);
        stepper_j6.set_frequency(fabsf(out_vel * STEPS_PER_RAD));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
void setup() {
    transport.init(UART_BAUD);

    encoder_j6.init();
    stepper_j6.init();
    observer_j6.set_initial_position(0.0f);
    interp_j6.reset(0.0f);

    // Start control loop
    servo_timer.begin(servo_isr, 1000);  // 1000 µs = 1 kHz

    Serial.println("<J6_TEST_FW_READY>");
    Serial.println("Send <ENABLE> to power the motor, then ROS commands.");
}

// ──────────────────────────────────────────────────────────────────────────────
static uint32_t last_fb_ms = 0;

void loop() {
    transport.process_incoming(rx_queue);

    while (!rx_queue.isEmpty()) {
        RosCommand cmd = rx_queue.shift();

        if (cmd.is_enable_cmd) {
            enabled = true;
            estop   = false;
            j6_target = j6_pos_rad;  // hold in place
            last_cmd_ms = millis();
            Serial.println("<ENABLE_ACK>");

        } else if (cmd.is_home_cmd) {
            // Zero encoder reference at current physical position
            encoder_j6.init();  // re-init clears accumulated counts
            observer_j6.set_initial_position(0.0f);
            j6_target = 0.0f;
            last_cmd_ms = millis();
            Serial.println("<HOMING_DONE>");

        } else {
            // Normal motion — only J6 (index 5). Others are ignored.
            j6_target = cmd.positions[5];
            last_cmd_ms = millis();
        }
    }

    // Watchdog: hold current position if no command arrives for 500 ms
    if (enabled && (millis() - last_cmd_ms > CMD_TIMEOUT_MS)) {
        j6_target = j6_pos_rad;
    }

    // Feedback at FB_RATE_HZ
    uint32_t now = millis();
    if (now - last_fb_ms >= (1000u / FB_RATE_HZ)) {
        last_fb_ms = now;

        float pos[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, j6_pos_rad};
        float vel[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, j6_vel_rad_s};
        uint8_t lim_state  = 0;
        uint8_t state_byte = enabled ? 1u : 3u;  // 1=NOMINAL, 3=DISABLED

        transport.send_feedback(fb_seq++, pos, vel, lim_state, state_byte);
    }
}
