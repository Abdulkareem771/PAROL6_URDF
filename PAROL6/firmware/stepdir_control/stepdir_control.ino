/*
 * STEP/DIR CONTROL - FOLLOWING ROS TRAJECTORY
 * 
 * Philosophy:
 * - ROS sends position updates at 20Hz
 * - ESP32 generates steps to follow those positions
 * - Motor tracks ROS's planned trajectory exactly
 * - No independent trajectory generation on ESP32
 * 
 * Hardware Setup:
 * - MKS must be in CR_vFOC (Step/Dir closed-loop) mode
 * - ESP32 GPIO 25 -> MKS STEP pin
 * - ESP32 GPIO 26 -> MKS DIR pin
 * - ESP32 GND -> MKS GND
 */

#include <Arduino.h>

#define USB_BAUD 115200

// Number of motors
#define NUM_MOTORS 6

// Step/Dir pins for each motor
const int STEP_PINS[NUM_MOTORS] = {25, 5, 14, 12, 13, 15};   // Configurable step pins (GPIO 5 instead of 32)
const int DIR_PINS[NUM_MOTORS] = {26, 2, 27, 4, 16, 17};     // Configurable dir pins (GPIO 2 instead of 33)

// Motor configuration
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define MOTOR_STEPS_PER_REV (STEPS_PER_REV * MICROSTEPS)  // 3200

// State for all motors
float current_positions[NUM_MOTORS] = {0, 0,0,0,0,0};  // Current positions in radians
float target_positions[NUM_MOTORS] = {0, 0,0,0,0,0};   // Target positions from ROS
String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_feedback = 0;

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
  
  // Move all motors to match target positions
  for (int i = 0; i < NUM_MOTORS; i++) {
    updateMotorPosition(i);
  }
  
  // Send feedback at 100 Hz
  unsigned long now = millis();
  if (now - last_feedback >= 10) {
    last_feedback = now;
    sendFeedback();
  }
}

void processCommand(String cmd) {
  // Parse: "SEQ,J1,J2,J3,J4,J5,J6"
  int idx = cmd.indexOf(',');
  if (idx == -1) return;
  
  last_seq = cmd.substring(0, idx).toInt();
  
  // Parse all 6 joint positions and set as targets
  int start = idx + 1;
  for (int i = 0; i < NUM_MOTORS; i++) {
    int next = cmd.indexOf(',', start);
    if (next == -1 && i < NUM_MOTORS - 1) return;
    
    String val = (next == -1) ? cmd.substring(start) : cmd.substring(start, next);
    target_positions[i] = val.toFloat();
    
    start = next + 1;
  }
}

void updateMotorPosition(int motor_idx) {
  // Calculate error for this motor
  float error = target_positions[motor_idx] - current_positions[motor_idx];
  
  // Dead zone
  if (abs(error) < 0.001) return;  // ~0.06 degrees
  
  // Convert error to steps
  float error_steps = (error * MOTOR_STEPS_PER_REV) / (2.0 * PI);
  int steps_needed = (int)abs(error_steps);
  
  if (steps_needed == 0) return;
  
  // Set direction for this motor
  if (error > 0) {
    digitalWrite(DIR_PINS[motor_idx], HIGH);
  } else {
    digitalWrite(DIR_PINS[motor_idx], LOW);
  }
  
  // Generate steps quickly to catch up
  // At 50us per step = 20,000 steps/sec max
  int steps_to_send = min(steps_needed, 100);  // Max 100 steps per update
  
  for (int i = 0; i < steps_to_send; i++) {
    digitalWrite(STEP_PINS[motor_idx], HIGH);
    delayMicroseconds(5);
    digitalWrite(STEP_PINS[motor_idx], LOW);
    delayMicroseconds(45);  // 50us total = 20kHz step rate
  }
  
  // Update position - TRUST MKS closed-loop to reach commanded position
  // We report the position we commanded, not encoder feedback
  float step_size = (2.0 * PI) / MOTOR_STEPS_PER_REV;
  float position_increment = step_size * steps_to_send;
  
  if (error > 0) {
    current_positions[motor_idx] += position_increment;
  } else {
    current_positions[motor_idx] -= position_increment;
  }
}

void sendFeedback() {
  // Report current positions for all 6 joints (ROS2 expects 6 values)
  // Controlled motors report actual position, others report 0.0
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  // Always send 6 joint positions
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    if (i < NUM_MOTORS) {
      Serial.print(current_positions[i], 3);  // Actual position for controlled motors
    } else {
      Serial.print("0.000");  // Zero for uncontrolled motors
    }
  }
  
  Serial.print(">");
  Serial.println();
  Serial.flush();
}

/*
 * DESIGN NOTES:
 * 
 * Trajectory Following:
 * - ROS sends position updates at 20Hz (every 50ms)
 * - ESP32 generates steps to match commanded position
 * - Motor follows ROS's planned trajectory exactly
 * - No independent speed control on ESP32
 * 
 * Step Generation:
 * - Fast step rate (20kHz) to quickly catch up to target
 * - Limited to 100 steps per update cycle (5ms max)
 * - Smooth because ROS sends smooth trajectory
 * 
 * Position Tracking:
 * - ESP32 tracks actual step count
 * - Reports position based on steps generated
 * - Trust MKS closed-loop to reach each step
 */
