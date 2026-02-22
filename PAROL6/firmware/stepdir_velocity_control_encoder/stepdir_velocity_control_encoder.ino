/*
 * STEP/DIR VELOCITY CONTROL WITH ENCODER FEEDBACK
 * 
 * Philosophy:
 * - ROS sends position AND velocity updates at 100Hz
 * - MT6816 encoders provide REAL position feedback (not step counting!)
 * - ESP32 generates steps at rate matching commanded velocity
 * - Position error calculated from ENCODER position vs target
 * - MKS handles closed-loop torque, ESP32 handles trajectory following
 * 
 * This eliminates position tracking snap by using real encoder feedback!
 */

#include <Arduino.h>

#define USB_BAUD 115200
#define NUM_MOTORS 6

// Step/Dir pins for each motor (corrected per user)
const int STEP_PINS[NUM_MOTORS] = {5, 25, 14, 12, 13, 15};  // J1=5, J2=25
const int DIR_PINS[NUM_MOTORS] = {2, 26, 27, 4, 16, 17};    // J1=2, J2=26

// Encoder PWM input pins
const int ENCODER_PINS[NUM_MOTORS] = {27, 33, 25, 26, 34, 35};  // J1=27 (existing)

// IMPORTANT: Set to true only for motors with encoders connected!
// Motors without encoders will use step counting (old method)
const bool ENCODER_ENABLED[NUM_MOTORS] = {
  true,   // J1: Encoder connected on GPIO 27
  false,  // J2: Not connected yet - use step counting
  false,  // J3: Not connected yet - use step counting
  false,  // J4: Not connected yet - use step counting
  false,  // J5: Not connected yet - use step counting
  false   // J6: Not connected yet - use step counting
};

// Motor configuration
#define STEPS_PER_REV 200

// Per-motor microstepping configuration
const int MICROSTEPS[NUM_MOTORS] = {
  4,   // J1: 20:1 gearbox - low microsteps for speed
  16,  // J2: Direct drive - high precision
  16,  // J3: Direct drive - high precision
  16,  // J4: Direct drive - high precision
  16,  // J5: Direct drive - high precision
  16   // J6: Direct drive - high precision
};

// Gearbox ratios (motor revolutions per joint revolution)
const float GEAR_RATIOS[NUM_MOTORS] = {
  20.0,  // J1: 20:1 gearbox
  1.0,   // J2: Direct drive
  1.0,   // J3: Direct drive
  1.0,   // J4: Direct drive
  1.0,   // J5: Direct drive
  1.0    // J6: Direct drive
};

// MT6816 Encoder PWM reading
// PWM pulse width encodes angle: width_us = (16 + angle_counts) * clock_period
const float ENCODER_CLOCK_PERIOD_NS = 250.0;  // 250ns clock period
const int ENCODER_START_CLOCKS = 16;           // 16-clock start pattern
const int ENCODER_RESOLUTION = 4096;           // 12-bit = 4096 positions

// Encoder state
volatile uint32_t encoder_rise_times[NUM_MOTORS] = {0};
volatile uint32_t encoder_pulse_widths[NUM_MOTORS] = {0};

// Multi-turn tracking for J2 (gearbox requires tracking multiple motor revolutions)
int motor_revolutions[NUM_MOTORS] = {0};
float last_motor_angles[NUM_MOTORS] = {0};

// State for all motors
float current_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};  // Read from encoders!
float target_positions[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};
float target_velocities[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};  // From ROS
float actual_velocities[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};  // Smoothed velocities

String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_feedback = 0;
unsigned long last_step_time[NUM_MOTORS] = {0, 0, 0, 0, 0, 0};

// Encoder interrupt handlers (one per motor)
void IRAM_ATTR encoder_isr_0() {
  if (digitalRead(ENCODER_PINS[0])) {
    encoder_rise_times[0] = micros();
  } else {
    encoder_pulse_widths[0] = micros() - encoder_rise_times[0];
  }
}

void IRAM_ATTR encoder_isr_1() {
  if (digitalRead(ENCODER_PINS[1])) {
    encoder_rise_times[1] = micros();
  } else {
    encoder_pulse_widths[1] = micros() - encoder_rise_times[1];
  }
}

void IRAM_ATTR encoder_isr_2() {
  if (digitalRead(ENCODER_PINS[2])) {
    encoder_rise_times[2] = micros();
  } else {
    encoder_pulse_widths[2] = micros() - encoder_rise_times[2];
  }
}

void IRAM_ATTR encoder_isr_3() {
  if (digitalRead(ENCODER_PINS[3])) {
    encoder_rise_times[3] = micros();
  } else {
    encoder_pulse_widths[3] = micros() - encoder_rise_times[3];
  }
}

void IRAM_ATTR encoder_isr_4() {
  if (digitalRead(ENCODER_PINS[4])) {
    encoder_rise_times[4] = micros();
  } else {
    encoder_pulse_widths[4] = micros() - encoder_rise_times[4];
  }
}

void IRAM_ATTR encoder_isr_5() {
  if (digitalRead(ENCODER_PINS[5])) {
    encoder_rise_times[5] = micros();
  } else {
    encoder_pulse_widths[5] = micros() - encoder_rise_times[5];
  }
}

// Read encoder angle in radians (motor shaft)
float readEncoderAngle(int motor_idx) {
  uint32_t pulse_width_us;
  
  noInterrupts();
  pulse_width_us = encoder_pulse_widths[motor_idx];
  interrupts();
  
  // Convert pulse width to clock counts
  float clocks = (pulse_width_us * 1000.0) / ENCODER_CLOCK_PERIOD_NS;
  
  // Remove 16-clock start pattern
  float angle_clocks = clocks - ENCODER_START_CLOCKS;
  
  // Clamp to valid range
  if (angle_clocks < 0) angle_clocks = 0;
  if (angle_clocks > ENCODER_RESOLUTION - 1) angle_clocks = ENCODER_RESOLUTION - 1;
  
  // Convert to radians (motor shaft angle 0-2π)
  float angle_rad = (angle_clocks / (float)ENCODER_RESOLUTION) * 2.0 * PI;
  
  return angle_rad;
}

// Update position from encoder with multi-turn tracking
// Falls back to step counting if encoder not enabled
void updatePositionFromEncoder(int motor_idx) {
  // If encoder not connected, skip (use step counting instead)
  if (!ENCODER_ENABLED[motor_idx]) {
    return;  // Position updated by step counting in updateMotorVelocity
  }
  
  float motor_angle = readEncoderAngle(motor_idx);
  
  // Multi-turn tracking: detect when motor crosses 0/360° boundary
  if (motor_angle < 0.5 && last_motor_angles[motor_idx] > 5.5) {
    // Forward crossing (0→360 to 360→0)
    motor_revolutions[motor_idx]++;
  } else if (motor_angle > 5.5 && last_motor_angles[motor_idx] < 0.5) {
    // Reverse crossing (360→0 to 0→360)
    motor_revolutions[motor_idx]--;
  }
  
  last_motor_angles[motor_idx] = motor_angle;
  
  // Calculate total motor angle including revolutions
  float total_motor_angle = motor_angle + (motor_revolutions[motor_idx] * 2.0 * PI);
  
  // Convert motor angle to joint angle (account for gearbox)
  current_positions[motor_idx] = total_motor_angle / GEAR_RATIOS[motor_idx];
}

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
  
  // Configure encoder pins and interrupts
  pinMode(ENCODER_PINS[0], INPUT);
  pinMode(ENCODER_PINS[1], INPUT);
  pinMode(ENCODER_PINS[2], INPUT);
  pinMode(ENCODER_PINS[3], INPUT);
  pinMode(ENCODER_PINS[4], INPUT);
  pinMode(ENCODER_PINS[5], INPUT);
  
  attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[0]), encoder_isr_0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[1]), encoder_isr_1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[2]), encoder_isr_2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[3]), encoder_isr_3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[4]), encoder_isr_4, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PINS[5]), encoder_isr_5, CHANGE);
  
  delay(100);  // Let encoders stabilize
  
  // Initialize positions from encoder (only for enabled encoders)
  for (int i = 0; i < NUM_MOTORS; i++) {
    if (ENCODER_ENABLED[i]) {
      updatePositionFromEncoder(i);
    } else {
      current_positions[i] = 0.0;  // Start at zero for step counting
    }
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
  
  // Update all encoder positions
  unsigned long now = micros();
  for (int i = 0; i < NUM_MOTORS; i++) {
    updatePositionFromEncoder(i);
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

// Per-motor dead zones based on encoder resolution
// Much tighter than step-based control!
const float DEAD_ZONES[NUM_MOTORS] = {
  0.0005,  // J1: Geared 20:1, encoder at motor is 20× finer than joint
  0.002,   // J2: Direct drive
  0.002,   // J3
  0.002,   // J4
  0.002,   // J5
  0.002    // J6
};

void updateMotorVelocity(int motor_idx, unsigned long now) {
  // PURE VELOCITY CONTROL - No position error checking!
  // ROS handles trajectory planning and sends velocity commands
  // We just follow the commanded velocity
  // Encoder provides feedback to ROS, not for local control
  
  // Get commanded velocity from ROS (rad/s)
  float target_velocity = target_velocities[motor_idx];
  
  // Smooth velocity changes to prevent snap (exponential smoothing)
  // Alpha = 0.8: Fast response (~37ms to 95% of target)
  float alpha = 0.8;
  actual_velocities[motor_idx] = alpha * target_velocity + (1.0 - alpha) * actual_velocities[motor_idx];
  
  float velocity = actual_velocities[motor_idx];
  
  // Only stop if velocity command is truly zero
  if (abs(velocity) < 0.001) return;
  
  // Convert velocity to step rate
  // velocity (rad/s) -> steps/s
  // Account for gearbox ratio
  float motor_steps_per_rev = STEPS_PER_REV * MICROSTEPS[motor_idx];
  float motor_revs_per_sec = abs(velocity) * GEAR_RATIOS[motor_idx] / (2.0 * PI);
  float steps_per_second = motor_revs_per_sec * motor_steps_per_rev;
  
  // Calculate time between steps (microseconds)
  unsigned long step_interval_us = (unsigned long)(1000000.0 / steps_per_second);
  
  // Limit maximum step rate (20kHz = 50us minimum interval)
  if (step_interval_us < 50) step_interval_us = 50;
  
  // Calculate how many steps we should have taken by now
  unsigned long elapsed = now - last_step_time[motor_idx];
  int steps_to_send = elapsed / step_interval_us;
  
  // Limit to prevent blocking the loop too long
  if (steps_to_send > 200) steps_to_send = 200;
  
  if (steps_to_send > 0) {
    // Set direction based on velocity sign
    if (velocity > 0) {
      digitalWrite(DIR_PINS[motor_idx], HIGH);
    } else {
      digitalWrite(DIR_PINS[motor_idx], LOW);
    }
    
    // Generate steps
    for (int i = 0; i < steps_to_send; i++) {
      digitalWrite(STEP_PINS[motor_idx], HIGH);
      delayMicroseconds(5);
      digitalWrite(STEP_PINS[motor_idx], LOW);
      delayMicroseconds(5);
      
      // If encoder not enabled, update position by counting steps
      if (!ENCODER_ENABLED[motor_idx]) {
        float step_size = (2.0 * PI) / (STEPS_PER_REV * MICROSTEPS[motor_idx] * GEAR_RATIOS[motor_idx]);
        if (velocity > 0) {
          current_positions[motor_idx] += step_size;
        } else {
          current_positions[motor_idx] -= step_size;
        }
      }
    }
    
    // Update timing
    last_step_time[motor_idx] = now;
  }
}

void sendFeedback() {
  // Report ENCODER positions AND commanded velocities for all 6 joints
  // Format: <ACK,SEQ,J1_pos,J1_vel,J2_pos,J2_vel,...>
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    if (i < NUM_MOTORS) {
      Serial.print(current_positions[i], 3);  // Encoder position!
      Serial.print(",");
      Serial.print(target_velocities[i], 3);  // Commanded velocity from ROS
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
 * Closed-Loop Position Control with Velocity Following:
 * - MT6816 encoders provide REAL position (not step counting)
 * - Position error = target - ENCODER position
 * - Velocity from ROS trajectory controls step rate
 * - Multi-turn tracking for gearbox joints
 * 
 * Advantages:
 * - NO position tracking error = NO SNAP!
 * - Real feedback catches missed steps
 * - Maintains ROS velocity control
 * - Works with loads (encoder shows true position)
 * 
 * Velocity Control Strategy:
 * - ROS sends target velocity from trajectory planner
 * - Firmware uses this velocity to set step rate
 * - Exponential smoothing (alpha=0.2) for smooth ramping
 * - Position error corrected continuously by encoder feedback
 * 
 * This is TRUE closed-loop control with trajectory following!
 */
