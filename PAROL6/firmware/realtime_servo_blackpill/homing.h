/*
 * PAROL6 Homing — Header
 *
 * Per-joint configurable homing toward limit switches and inductive
 * proximity sensors.  State machine runs in the main loop; sensor
 * detection is via pin-change interrupts.
 */

#ifndef HOMING_H
#define HOMING_H

#include "config.h"

// ============================================================================
// HOMING STATE (per joint)
// ============================================================================

enum HomingState : uint8_t {
    HOMING_IDLE,         // Not homing
    HOMING_BACKING_OFF,  // Sensor already triggered → move away first
    HOMING_SEEKING,      // Moving toward sensor
    HOMING_ZEROING,      // Sensor hit → settling, about to zero
    HOMING_COMPLETE,     // Homed successfully
    HOMING_ERROR         // Timeout
};

// ============================================================================
// API
// ============================================================================

void         homingInit(void);           // Configure sensor pins + interrupts
void         homingStart(void);          // Begin homing for all enabled joints
void         homingUpdate(void);         // Call from loop() — runs state machine
bool         homingIsComplete(void);     // True when all enabled joints are homed
bool         homingIsActive(void);       // True when any joint is currently homing
HomingState  homingGetState(uint8_t idx); // Per-joint state query

// Status byte for serial feedback:
//   0 = idle/not started
//   1 = homing in progress
//   2 = all homed successfully
//   3 = error on at least one joint
uint8_t      homingGetStatus(void);

#endif // HOMING_H
