/*
 * STEP/DIR VELOCITY CONTROL
 * 
 * Philosophy:
 * - ROS sends position AND velocity updates at 100Hz
 * - ESP32 generates steps at rate matching commanded velocity
 * - MKS still handles closed-loop, but we control the velocity profile
 * 
 * This gives you full control over acceleration/velocity from ROS!
 */

#include <Arduino.h>

#define USB_BAUD 115200
#define NUM_MOTORS 6

// Step/Dir pins for each motor
const int STEP_PINS[NUM_MOTORS] = {25, 5, 14, 12, 13, 15};
const int DIR_PINS[NUM_MOTORS] = {26, 2, 27, 4, 16, 17};

// Motor configuration
#define STEPS_PER_REV 200

// Per-motor microstepping configuration
// Lower values (4, 8) for geared joints = faster motion, lower precision
// Higher values (16, 32) for direct-drive joints = slower motion, higher precision
const int MICROSTEPS[NUM_MOTORS] = {
  16,  // J1: Direct drive - high precision
  4,   // J2: 20:1 gearbox - low microsteps for speed
  16,  // J3: Direct drive - high precision
  16,  // J4: Direct drive - high precision
  16,  // J5: Direct drive - high precision
  16   // J6: Direct drive - high precision
};

// Gearbox ratios (motor revolutions per joint revolution)
// Set to 1.0 for direct drive, or actual ratio for geared joints
// Example: 20.0 means motor spins 20 times for 1 joint revolution
const float GEAR_RATIOS[NUM_MOTORS] = {
  1.0,   // J1: Direct drive (no gearbox = 1)
  20.0,  // J2: 20:1 gearbox
  1.0,   // J3: Direct drive
  1.0,   // J4: Direct drive
  1.0,   // J5: Direct drive
  1.0    // J6: Direct drive
};

// Calculate total steps per joint revolution (including gearbox and microsteps)
float getStepsPerJointRev(int motor_idx) {
  float motor_steps_per_rev = STEPS_PER_REV * MICROSTEPS[motor_idx];
  return motor_steps_per_rev * GEAR_RATIOS[motor_idx];
}

// State for all motors
float current_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
float target_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
float target_velocities[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};  // NEW: velocity commands
float actual_velocities[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};  // NEW: smoothed velocities

String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_feedback = 0;
unsigned long last_step_time[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  // Configure step/dir pins for all motors
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(STEP_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
    digitalWrite(DIR_PINS[i], LOW);
  }
}

void loop() {
  // Read commands from ROS
  while (Serial.available()) {
    char c = Serial.read();
    
    if (c == '<') {
      inputBuffer = "";
    } else if (c == '>') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
  
  // Move all motors at commanded velocity
  unsigned long now = micros();
  for (int i = 0; i < NUM_MOTORS; i++) {
    updateMotorVelocity(i, now);
  }
  
  // Send feedback at 100 Hz
  if (millis() - last_feedback >= 10) {
    last_feedback = millis();
    sendFeedback();
  }
}

void processCommand(String cmd) {
  // Parse: "SEQ,J1_pos,J1_vel,J2_pos,J2_vel,J3_pos,J3_vel,J4_pos,J4_vel,J5_pos,J5_vel,J6_pos,J6_vel"
  // Total: 1 (seq) + 12 (6 joints × 2 values) = 13 tokens
  
  int idx = cmd.indexOf(',');
  if (idx == -1) return;
  
  last_seq = cmd.substring(0, idx).toInt();
  
  // Parse all 6 joints (position + velocity pairs)
  int start = idx + 1;
  for (int i = 0; i < NUM_MOTORS; i++) {
    // Parse position
    int next = cmd.indexOf(',', start);
    if (next == -1) return;  // Need velocity after position
    
    String pos_val = cmd.substring(start, next);
    target_positions[i] = pos_val.toFloat();
    
    // Parse velocity
    start = next + 1;
    next = cmd.indexOf(',', start);
    
    String vel_val;
    if (i == NUM_MOTORS - 1) {
      // Last joint - no comma after velocity
      vel_val = cmd.substring(start);
    } else {
      // More joints to come
      if (next == -1) return;  // Missing data
      vel_val = cmd.substring(start, next);
      start = next + 1;
    }
    
    target_velocities[i] = vel_val.toFloat();
  }
}

// Per-motor dead zones based on step resolution
// Calculated as ~1 step size for each motor configuration
const float DEAD_ZONES[NUM_MOTORS] = {
  0.002,   // J1: 16 microsteps, direct (0.1125° per step)
  0.001,   // J2: 4 microsteps, 20:1 gearbox (0.0225° per step) - balanced
  0.002,   // J3: 16 microsteps, direct
  0.002,   // J4: 16 microsteps, direct
  0.002,   // J5: 16 microsteps, direct
  0.002    // J6: 16 microsteps, direct
};

void updateMotorVelocity(int motor_idx, unsigned long now) {
  // Calculate error
  float error = target_positions[motor_idx] - current_positions[motor_idx];
  
  // Use per-motor dead zone based on step resolution
  if (abs(error) < DEAD_ZONES[motor_idx]) return;
  
  // Get commanded velocity (rad/s)
  float target_velocity = target_velocities[motor_idx];
  
  // Smooth velocity changes to prevent snap (exponential smoothing)
  // Lower alpha = smoother start (prevents snap)
  // Higher alpha = faster response
  // Alpha = 0.1 means 10% new value, 90% old value per cycle
  // This creates a gentle ramp-up over ~10 cycles (100ms at 100Hz)
  float alpha = 0.2;  // Reduced from 0.5 to prevent hard snap
  actual_velocities[motor_idx] = alpha * target_velocity + (1.0 - alpha) * actual_velocities[motor_idx];
  
  float velocity = actual_velocities[motor_idx];
  
  // Allow very small velocities for smooth motion
  if (abs(velocity) < 0.001) return;  // Much smaller threshold
  
  // Convert velocity to step rate
  // velocity (rad/s) -> steps/s
  // Account for gearbox ratio
  float steps_per_joint_rev = getStepsPerJointRev(motor_idx);
  float steps_per_second = (abs(velocity) * steps_per_joint_rev) / (2.0 * PI);
  
  // Calculate time between steps (microseconds)
  unsigned long step_interval_us = (unsigned long)(1000000.0 / steps_per_second);
  
  // Limit maximum step rate (20kHz = 50us minimum interval)
  if (step_interval_us < 50) step_interval_us = 50;
  
  // Calculate how many steps we should have taken by now
  unsigned long elapsed = now - last_step_time[motor_idx];
  int steps_to_send = elapsed / step_interval_us;
  
  // Limit to prevent blocking the loop too long
  // Increased for geared joints to allow faster catch-up
  if (steps_to_send > 200) steps_to_send = 200;  // Was 50
  
  if (steps_to_send > 0) {
    // Set direction
    if (error > 0) {
      digitalWrite(DIR_PINS[motor_idx], HIGH);
    } else {
      digitalWrite(DIR_PINS[motor_idx], LOW);
    }
    
    // Generate multiple steps to catch up
    for (int i = 0; i < steps_to_send; i++) {
      digitalWrite(STEP_PINS[motor_idx], HIGH);
      delayMicroseconds(5);
      digitalWrite(STEP_PINS[motor_idx], LOW);
      delayMicroseconds(5);  // Shorter delay between steps
      
      // Update position
      float step_size = (2.0 * PI) / getStepsPerJointRev(motor_idx);
      if (error > 0) {
        current_positions[motor_idx] += step_size;
      } else {
        current_positions[motor_idx] -= step_size;
      }
    }
    
    // Update timing
    last_step_time[motor_idx] = now;
  }
}

void sendFeedback() {
  // Report current positions AND velocities for all 6 joints
  // Format: <ACK,SEQ,J1_pos,J1_vel,J2_pos,J2_vel,...>
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    if (i < NUM_MOTORS) {
      Serial.print(current_positions[i], 3);
      Serial.print(",");
      Serial.print(target_velocities[i], 3);  // Report commanded velocity
    } else {
      Serial.print("0.000,0.000");
    }
  }
  
  Serial.print(">");
  Serial.println();
  Serial.flush();
}

/*
 * DESIGN NOTES:
 * 
 * Velocity Control:
 * - ROS sends position updates at 100Hz
 * - ESP32 calculates velocity from position changes
 * - Steps are generated at rate matching velocity
 * - Smooth acceleration controlled by ROS trajectory
 * 
 * Advantages:
 * - Full control over velocity profile from ROS
 * - MKS acceleration disabled (or minimized)
 * - Smoother motion following ROS trajectory exactly
 * 
 * Disadvantages:
 * - More complex than simple position following
 * - Requires careful tuning of step rates
 * - Still limited by 100Hz update rate from ROS
 */
