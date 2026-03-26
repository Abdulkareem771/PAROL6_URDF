/*
 * PAROL6 Homing — State Machine Implementation
 *
 * Runs from the main loop (NOT from an ISR).  Each enabled joint goes
 * through:  IDLE → [BACKING_OFF] → SEEKING → ZEROING → COMPLETE
 *
 * Sensor detection uses pin-change interrupts that set a volatile flag.
 * The state machine polls the flag in homingUpdate().
 *
 * After the sensor triggers and the motor settles, the encoder position
 * is reset to the configured offset (in degrees, converted to radians).
 */

#include "homing.h"
#include "control.h"
#include "motor.h"
#include <Arduino.h>

// ============================================================================
// PER-JOINT STATE
// ============================================================================

static volatile bool sensor_triggered[NUM_MOTORS] = {false};
static HomingState   joint_state[NUM_MOTORS];
static uint32_t      state_start_ms[NUM_MOTORS];
static bool          homing_requested = false;

// ============================================================================
// SENSOR ISR CALLBACKS (one per joint, sets flag)
// ============================================================================
// STM32Duino attachInterrupt() requires a void(void) callback.
// We use a macro to generate one function per joint.

#define MAKE_SENSOR_ISR(N) \
    static void sensorISR_J##N(void) { sensor_triggered[N] = true; }

MAKE_SENSOR_ISR(0)
MAKE_SENSOR_ISR(1)
MAKE_SENSOR_ISR(2)
MAKE_SENSOR_ISR(3)
MAKE_SENSOR_ISR(4)
MAKE_SENSOR_ISR(5)

typedef void (*ISRFunc)(void);
static const ISRFunc sensorISRs[NUM_MOTORS] = {
    sensorISR_J0, sensorISR_J1, sensorISR_J2,
    sensorISR_J3, sensorISR_J4, sensorISR_J5
};

// ============================================================================
// READ SENSOR PIN (direct, non-interrupt)
// ============================================================================

static bool sensorIsActive(uint8_t idx)
{
    if (HOMING_SENSOR_PINS[idx] == 0) return false;

    bool pin_high = (digitalRead(HOMING_SENSOR_PINS[idx]) == HIGH);

    // Inductive prox: active = HIGH (metal detected → optocoupler releases → pullup → HIGH)
    // Limit switch:   active = LOW  (switch pressed → pulls to GND)
    if (HOMING_SENSOR_TYPE[idx] == 1) {
        return pin_high;   // inductive: HIGH = triggered
    } else {
        return !pin_high;  // limit switch: LOW = triggered
    }
}

// ============================================================================
// INIT
// ============================================================================

void homingInit(void)
{
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        joint_state[i] = HOMING_IDLE;
        sensor_triggered[i] = false;

        if (!HOMING_ENABLED[i] || HOMING_SENSOR_PINS[i] == 0) continue;

        // Configure pin with internal pull-up
        pinMode(HOMING_SENSOR_PINS[i], INPUT_PULLUP);

        // Attach interrupt:
        //   Inductive proximity: RISING edge (pin goes HIGH when metal detected)
        //   Limit switch:        FALLING edge (pin pulled LOW when pressed)
        int edge = (HOMING_SENSOR_TYPE[i] == 1) ? RISING : FALLING;
        attachInterrupt(digitalPinToInterrupt(HOMING_SENSOR_PINS[i]),
                        sensorISRs[i], edge);
    }
}

// ============================================================================
// START HOMING
// ============================================================================

void homingStart(void)
{
    homing_requested = true;
    bool first_started = false;

    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (!HOMING_ENABLED[i]) {
            joint_state[i] = HOMING_COMPLETE;  // skip disabled joints
            continue;
        }

        sensor_triggered[i] = false;

        if (HOMING_SEQUENTIAL && first_started) {
            joint_state[i] = HOMING_PENDING;
            continue; // Wait turn
        }

        // Check if sensor is already triggered (e.g. J2 resting on switch)
        if (sensorIsActive(i)) {
            joint_state[i] = HOMING_BACKING_OFF;
        } else {
            joint_state[i] = HOMING_SEEKING;
        }

        state_start_ms[i] = millis();
        first_started = true;
    }
}

// ============================================================================
// DRIVE JOINT (helper — sets velocity & direction for homing)
// ============================================================================
// During homing, we bypass the normal controlSetCommand() path from ROS.
// Since the motor is physically moving but ROS isn't sending new desired positions,
// a large position_error would build up. We prevent this by forcing
// desired_position = actual_position to kill the proportional term.

static void driveJoint(uint8_t idx, float velocity_rad_s)
{
    const JointState *js = controlGetState(idx);
    if (!js) return;
    
    // Force desired_position = actual_position to zero out the P-term
    controlSetCommand(idx, js->actual_position, velocity_rad_s);
}

static void stopJoint(uint8_t idx)
{
    const JointState *js = controlGetState(idx);
    if (!js) return;
    
    controlSetCommand(idx, js->actual_position, 0.0f);
}

// ============================================================================
// HELPER: START NEXT PENDING JOINT (SEQUENTIAL HOMING)
// ============================================================================

static void startNextPendingJoint(uint32_t now)
{
    if (!HOMING_SEQUENTIAL) return;

    for (uint8_t j = 0; j < NUM_MOTORS; j++) {
        if (joint_state[j] == HOMING_PENDING) {
            if (sensorIsActive(j)) {
                joint_state[j] = HOMING_BACKING_OFF;
            } else {
                joint_state[j] = HOMING_SEEKING;
            }
            state_start_ms[j] = now;
            break; // Only start one at a time
        }
    }
}

// ============================================================================
// UPDATE (call from main loop — NOT time-critical)
// ============================================================================

void homingUpdate(void)
{
    if (!homing_requested) return;

    uint32_t now = millis();

    for (uint8_t i = 0; i < NUM_MOTORS; i++) {

        switch (joint_state[i]) {

        // ----- IDLE / COMPLETE / ERROR / PENDING: nothing to do -----
        case HOMING_IDLE:
        case HOMING_COMPLETE:
        case HOMING_ERROR:
        case HOMING_PENDING:
            break;

        // ----- BACKING OFF: sensor was already triggered at start -----
        case HOMING_BACKING_OFF:
        {
            // Timeout check
            if ((now - state_start_ms[i]) > HOMING_TIMEOUT_MS) {
                stopJoint(i);
                joint_state[i] = HOMING_ERROR;
                break;
            }

            // Move in the OPPOSITE direction of homing at reduced speed
            float backoff_vel = -HOMING_DIR[i] * HOMING_SPEED[i] * HOMING_BACKOFF_SPEED_MULT;
            driveJoint(i, backoff_vel);

            // Check if sensor has been released
            if (!sensorIsActive(i)) {
                // Sensor released — now seek toward it
                stopJoint(i);
                sensor_triggered[i] = false;  // clear any interrupt flag
                joint_state[i] = HOMING_SEEKING;
                state_start_ms[i] = now;      // reset timeout
            }
            break;
        }

        // ----- SEEKING: moving toward home sensor -----
        case HOMING_SEEKING:
        {
            // Timeout check
            if ((now - state_start_ms[i]) > HOMING_TIMEOUT_MS) {
                stopJoint(i);
                joint_state[i] = HOMING_ERROR;
                break;
            }

            // Drive in homing direction
            float seek_vel = HOMING_DIR[i] * HOMING_SPEED[i];
            driveJoint(i, seek_vel);

            // Check if sensor was triggered (interrupt flag)
            if (sensor_triggered[i] || sensorIsActive(i)) {
                stopJoint(i);
                joint_state[i] = HOMING_ZEROING;
                state_start_ms[i] = now;
            }
            break;
        }

        // ----- ZEROING: sensor triggered, settling, then reset position -----
        case HOMING_ZEROING:
        {
            stopJoint(i);

            // Wait for settle time
            if ((now - state_start_ms[i]) >= HOMING_SETTLE_MS) {
                // Convert degree offset to radians
                float offset_rad = DEG_TO_RAD(HOMING_OFFSET_DEG[i]);

                // Reset the encoder/position tracking to the offset
                controlResetPosition(i, offset_rad);

                // Option: Move to ready position before finishing homing
                if (HOMING_READY_POS_DEG[i] != HOMING_NO_READY) {
                    joint_state[i] = HOMING_MOVING_TO_READY;
                    state_start_ms[i] = now;
                } else {
                    joint_state[i] = HOMING_COMPLETE;
                    startNextPendingJoint(now);
                }
            }
            break;
        }

        // ----- MOVING TO READY: homing finished, now moving to set angle -----
        case HOMING_MOVING_TO_READY:
        {
            const JointState *js = controlGetState(i);
            if (!js) break;

            float target_rad = DEG_TO_RAD(HOMING_READY_POS_DEG[i]);
            float error = target_rad - js->actual_position;

            // Stop if within tolerance (~1.1 degrees)
            if (fabsf(error) < 0.02f) {
                stopJoint(i);
                joint_state[i] = HOMING_COMPLETE;
                startNextPendingJoint(now);
            } else {
                // Drive towards target
                float vel = (error > 0.0f) ? HOMING_SPEED[i] : -HOMING_SPEED[i];
                
                // Optional: slow down as we approach to avoid overshoot
                if (fabsf(error) < 0.1f) vel *= 0.5f; 
                
                driveJoint(i, vel);
            }
            break;
        }

        } // switch
    } // for
}

// ============================================================================
// QUERIES
// ============================================================================

bool homingIsComplete(void)
{
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (HOMING_ENABLED[i] && joint_state[i] != HOMING_COMPLETE) {
            return false;
        }
    }
    return homing_requested;  // only true if homing was actually started
}

bool homingIsActive(void)
{
    if (!homing_requested) return false;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (joint_state[i] == HOMING_SEEKING ||
            joint_state[i] == HOMING_BACKING_OFF ||
            joint_state[i] == HOMING_ZEROING ||
            joint_state[i] == HOMING_MOVING_TO_READY ||
            joint_state[i] == HOMING_PENDING) {
            return true;
        }
    }
    return false;
}

HomingState homingGetState(uint8_t idx)
{
    if (idx >= NUM_MOTORS) return HOMING_IDLE;
    return joint_state[idx];
}

uint8_t homingGetStatus(void)
{
    if (!homing_requested) return 0;  // never started

    bool any_error = false;
    bool any_active = false;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (!HOMING_ENABLED[i]) continue;
        if (joint_state[i] == HOMING_ERROR) any_error = true;
        if (joint_state[i] == HOMING_SEEKING ||
            joint_state[i] == HOMING_BACKING_OFF ||
            joint_state[i] == HOMING_ZEROING ||
            joint_state[i] == HOMING_MOVING_TO_READY ||
            joint_state[i] == HOMING_PENDING) any_active = true;
    }

    if (any_error) return 3;
    if (any_active) return 1;
    return 2;  // all complete
}
