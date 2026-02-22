/*
 * PAROL6 Control Task — Position Servo with Velocity Feedforward
 *
 * Implements servo control law:
 *   velocity_command = desired_velocity + Kp × (desired_pos − actual_pos)
 *
 * Identical control logic to ESP32 version.
 * Teensy differences: no IRAM_ATTR needed, all pins support interrupts.
 */

#ifndef CONTROL_H
#define CONTROL_H

#include <Arduino.h>
#include "config.h"

// ============================================================================
// JOINT STATE STRUCTURE
// ============================================================================

struct JointState {
  // Commanded states (from ROS)
  float desired_position;     // rad
  float desired_velocity;     // rad/s

  // Actual states (from encoders or step counting)
  float actual_position;      // rad
  float actual_velocity;      // rad/s (optional, for Kd term)

  // Control outputs
  float position_error;       // rad
  float velocity_command;     // rad/s

  // Multi-turn tracking (for all encoder-enabled motors)
  int motor_revolutions;
  float last_motor_angle;
};

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

// Initialize control system (encoder ISRs, state zeroing)
void controlInit();

// Update command from ROS (called from main loop)
void controlSetCommand(uint8_t joint_idx, float position, float velocity);

// Main control loop (called from IntervalTimer ISR at 500 Hz)
void controlUpdate();

// Get joint state (for feedback)
const JointState* controlGetState(uint8_t joint_idx);

#endif // CONTROL_H
