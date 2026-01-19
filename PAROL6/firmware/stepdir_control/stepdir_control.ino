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

// Step/Dir pins
#define STEP_PIN 25
#define DIR_PIN 26

// Motor configuration
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define MOTOR_STEPS_PER_REV (STEPS_PER_REV * MICROSTEPS)  // 3200

// State
float current_position = 0.0;  // Current position in radians
float target_position = 0.0;   // Target position from ROS
float current_positions[6] = {0, 0, 0, 0, 0, 0};  // All 6 joints
String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_feedback = 0;

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  // Configure step/dir pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);
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
  
  // Move motor to match target position
  updateMotorPosition();
  
  // Send feedback at 100 Hz (ultra-smooth)
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
  
  // Parse all 6 joint positions
  int start = idx + 1;
  for (int i = 0; i < 6; i++) {
    int next = cmd.indexOf(',', start);
    if (next == -1 && i < 5) return;
    
    String val = (next == -1) ? cmd.substring(start) : cmd.substring(start, next);
    current_positions[i] = val.toFloat();
    
    // Update target for J1
    if (i == 0) {
      target_position = current_positions[0];
    }
    
    start = next + 1;
  }
}

void updateMotorPosition() {
  // Calculate error
  float error = target_position - current_position;
  
  // Dead zone
  if (abs(error) < 0.001) return;  // ~0.06 degrees
  
  // Convert error to steps
  float error_steps = (error * MOTOR_STEPS_PER_REV) / (2.0 * PI);
  int steps_needed = (int)abs(error_steps);
  
  if (steps_needed == 0) return;
  
  // Set direction
  if (error > 0) {
    digitalWrite(DIR_PIN, HIGH);
  } else {
    digitalWrite(DIR_PIN, LOW);
  }
  
  // Generate steps quickly to catch up
  // At 50us per step = 20,000 steps/sec max
  for (int i = 0; i < steps_needed && i < 100; i++) {  // Max 100 steps per update
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(5);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(45);  // 50us total = 20kHz step rate
    
    // Update position
    float step_size = (2.0 * PI) / MOTOR_STEPS_PER_REV;
    if (error > 0) {
      current_position += step_size;
    } else {
      current_position -= step_size;
    }
  }
}

void sendFeedback() {
  // Report current position
  Serial.print("<ACK,");
  Serial.print(last_seq);
  Serial.print(",");
  Serial.print(current_position, 3);
  Serial.print(",");
  Serial.print(current_positions[1], 2);
  Serial.print(",");
  Serial.print(current_positions[2], 2);
  Serial.print(",");
  Serial.print(current_positions[3], 2);
  Serial.print(",");
  Serial.print(current_positions[4], 2);
  Serial.print(",");
  Serial.print(current_positions[5], 2);
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
