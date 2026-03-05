/*
 * PAROL6 Control Task - Position Servo with Velocity Feedforward
 * 
 * Implements servo control law:
 *   velocity_command = desired_velocity + Kp * (desired_position - actual_position)
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
  
  // Multi-turn tracking (for gearbox motors with encoders)
  int motor_revolutions;
  float last_motor_angle;
};

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

// Initialize control system
void controlInit();

// Update command from ROS (called from serial task)
void controlSetCommand(uint8_t joint_idx, float position, float velocity);

// Main control loop (called from FreeRTOS task at 500 Hz)
void controlUpdate();

// Get joint state (for debugging/feedback)
const JointState* controlGetState(uint8_t joint_idx);

#endif // CONTROL_H
