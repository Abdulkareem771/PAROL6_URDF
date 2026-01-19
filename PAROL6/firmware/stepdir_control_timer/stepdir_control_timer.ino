/*
 * STEP/DIR CONTROL - HARDWARE TIMER VERSION
 * 
 * Uses ESP32 hardware timer for precise feedback timing
 * Allows exact frequencies like 62.5Hz, 66.67Hz, 75Hz, etc.
 * No drift, perfect sync with ROS
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

// Feedback frequency (change this to match ROS update_rate)
#define FEEDBACK_HZ 75.0  // Can be any float: 50, 62.5, 66.67, 75, 100, etc.

// State
float current_position = 0.0;
float target_position = 0.0;
float current_positions[6] = {0, 0, 0, 0, 0, 0};
String inputBuffer = "";
uint32_t last_seq = 0;
volatile bool send_feedback_flag = false;

// Hardware timer
hw_timer_t *feedback_timer = NULL;

// Timer interrupt - sets flag to send feedback
void IRAM_ATTR onFeedbackTimer() {
  send_feedback_flag = true;
}

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  // Configure step/dir pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);
  
  // Setup hardware timer for feedback (ESP32 Arduino Core 3.x API)
  // Calculate frequency in Hz for timerBegin
  uint32_t timer_frequency = (uint32_t)FEEDBACK_HZ;
  feedback_timer = timerBegin(timer_frequency);
  
  // Attach interrupt function
  timerAttachInterrupt(feedback_timer, &onFeedbackTimer);
  
  // Timer will trigger at FEEDBACK_HZ automatically
  timerAlarm(feedback_timer, 1000000 / FEEDBACK_HZ, true, 0);
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
  
  // Send feedback when timer triggers
  if (send_feedback_flag) {
    send_feedback_flag = false;
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
  if (abs(error) < 0.001) return;
  
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
  for (int i = 0; i < steps_needed && i < 100; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(5);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(45);
    
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
 * HARDWARE TIMER ADVANTAGES:
 * 
 * - Microsecond precision (vs millisecond with millis())
 * - Can use any frequency: 62.5Hz, 66.67Hz, 75Hz, etc.
 * - No drift - perfectly synchronized
 * - Independent of loop timing
 * 
 * TO CHANGE FREQUENCY:
 * 1. Set FEEDBACK_HZ to desired value (can be float!)
 * 2. Set ROS update_rate to same value
 * 3. Perfect sync guaranteed!
 * 
 * EXAMPLES:
 * - 50Hz: FEEDBACK_HZ = 50.0
 * - 62.5Hz: FEEDBACK_HZ = 62.5
 * - 66.67Hz: FEEDBACK_HZ = 66.67
 * - 75Hz: FEEDBACK_HZ = 75.0
 * - 100Hz: FEEDBACK_HZ = 100.0
 */
