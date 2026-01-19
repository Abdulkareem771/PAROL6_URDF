/*
 * OPEN-LOOP POSITION CONTROL - Trust MKS Closed-Loop System
 * 
 * Philosophy:
 * - MKS Servo42C has built-in closed-loop control
 * - We send position commands
 * - MKS handles the rest (encoder, PID, etc.)
 * - We report commanded position back to ROS
 * 
 * This is the correct approach for a closed-loop stepper driver!
 */

#include "SERVO42C.h"

#define USB_BAUD 115200
#define MOTOR_BAUD 115200
#define MOTOR_RX_PIN 16
#define MOTOR_TX_PIN 17
#define MOTOR_ADDR 0xE0

// Motor configuration
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define MOTOR_PULSES_PER_REV (STEPS_PER_REV * MICROSTEPS)  // 3200

SERVO42C motor(Serial2, MOTOR_ADDR);

// State
float current_positions[6] = {0, 0, 0, 0, 0, 0};
String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_feedback = 0;
unsigned long last_motor_command = 0;  // For rate limiting motor commands

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  // Startup messages disabled - keep serial clean for ROS
  // Serial.println("=== PAROL6 Open-Loop Control ===");
  // Serial.println("Trusting MKS closed-loop system");
  // Serial.println();
  
  // Initialize motor serial
  Serial2.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);
  delay(100);
  
  motor.begin(MOTOR_BAUD);
  delay(200);
  
  // Configure for UART mode
  motor.setWorkMode(MODE_UART);
  delay(100);
  motor.uartEnable(true);
  delay(100);
  motor.setMicrostep(MICROSTEPS);
  delay(50);
  
  // Serial.println("READY - Send commands: <SEQ,J1,J2,J3,J4,J5,J6>");
  // Serial.println("Example: <0,1.5,0,0,0,0,0>");
  // Serial.println();
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
  
  // Send feedback at 75 Hz (optimal: smooth + reliable)
  unsigned long now = millis();
  if (now - last_feedback >= 13) {  // ~13.33ms = 75Hz
    last_feedback = now;
    sendFeedback();
    Serial.flush();
  }
}

void processCommand(String cmd) {
  // Parse: "SEQ,J1,J2,J3,J4,J5,J6"
  int idx = cmd.indexOf(',');
  if (idx == -1) return;
  
  last_seq = cmd.substring(0, idx).toInt();
  
  // Parse 6 joint positions
  int start = idx + 1;
  for (int i = 0; i < 6; i++) {
    int next = cmd.indexOf(',', start);
    if (next == -1 && i < 5) return;
    
    String val = (next == -1) ? cmd.substring(start) : cmd.substring(start, next);
    float new_target = val.toFloat();
    
    // Only control J1 (motor 1)
    // With 100Hz updates, use very small threshold for ultra-smooth motion
    if (i == 0 && abs(new_target - current_positions[0]) > 0.001) {  // ~0.06 degrees
      moveMotor(new_target);
    }
    
    current_positions[i] = new_target;
    start = next + 1;
  }
}

void moveMotor(float target_rad) {
  // Convert radians to motor pulses
  float target_deg = target_rad * (180.0 / PI);
  float pulses_per_deg = (float)MOTOR_PULSES_PER_REV / 360.0;
  int32_t target_pulses = (int32_t)(target_deg * pulses_per_deg);
  
  // Get current position
  float current_deg = current_positions[0] * (180.0 / PI);
  int32_t current_pulses = (int32_t)(current_deg * pulses_per_deg);
  
  int32_t delta = target_pulses - current_pulses;
  
  if (abs(delta) < 5) {
    return;  // Already close enough
  }
  
  // Send position command
  RunDirection dir = (delta > 0) ? RUN_FWD : RUN_REV;
  uint32_t pulses = abs(delta);
  uint8_t speed = 15;  // Slower speed for smoother movement
  uint8_t status;
  
  motor.uartRunPulses(speed, dir, pulses, status);
}

void sendFeedback() {
  // Report commanded position (trust MKS reached it)
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    Serial.print(current_positions[i], 2);
  }
  
  Serial.println(">");
}

/*
 * DESIGN PHILOSOPHY:
 * 
 * The MKS Servo42C is a CLOSED-LOOP stepper driver.
 * It has:
 * - Built-in encoder
 * - Built-in PID controller
 * - Position tracking
 * 
 * We don't need to:
 * - Read the encoder ourselves
 * - Calculate position errors
 * - Implement our own control loop
 * 
 * We just:
 * - Send position commands
 * - Trust the MKS will reach them
 * - Report commanded position to ROS
 * 
 * This is how closed-loop steppers are meant to be used!
 * 
 * TESTING:
 * 1. Upload firmware
 * 2. Send: <0,1.5,0,0,0,0,0>
 * 3. Motor moves to 1.5 rad
 * 4. Send: <1,0.0,0,0,0,0,0>
 * 5. Motor returns to zero
 * 
 * For ROS integration:
 * - ROS sends target positions
 * - We command the motor
 * - We report back the commanded position
 * - ROS uses this for visualization and planning
 * 
 * The MKS handles the actual position control!
 */
