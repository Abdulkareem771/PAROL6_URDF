/*
 * SIMPLIFIED VERSION - Works like minimal test
 * 
 * This version:
 * - Receives ONE command from ROS
 * - Calculates the move needed
 * - Sends it ONCE
 * - Waits for completion
 * - Does NOT continuously update
 */

#include "SERVO42C.h"

#define USB_BAUD 115200
#define MOTOR_BAUD 38400
#define MOTOR_RX_PIN 16
#define MOTOR_TX_PIN 17
#define MOTOR_ADDR 0xE0

// CRITICAL: Encoder vs Motor pulses are DIFFERENT!
// Encoder: 16384 counts per revolution (for reading position)
#define ENCODER_CPR 16384
#define ENCODER_TO_RAD(enc) ((float)((enc) * (2.0 * PI) / ENCODER_CPR))

// Motor: 200 steps/rev × 16 microsteps = 3200 pulses/rev (for commanding)
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define MOTOR_PULSES_PER_REV (STEPS_PER_REV * MICROSTEPS)  // = 3200
#define RAD_TO_MOTOR_PULSES(rad) ((int32_t)((rad) * MOTOR_PULSES_PER_REV / (2.0 * PI)))

#define UPDATE_INTERVAL_MS 100   // 10 Hz for feedback

SERVO42C motor(Serial2, MOTOR_ADDR);

float target_positions[6] = {0, 0, 0, 0, 0, 0};
float current_positions[6] = {0, 0, 0, 0, 0, 0};

String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_update = 0;

bool motor_initialized = false;
bool move_commanded = false;  // Track if we've sent a move for current target
int32_t last_target_pulses = 0;  // Track in MOTOR pulses, not encoder counts

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  Serial.println("=== PAROL6 Simple Position Control ===");
  
  Serial2.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);
  delay(100);
  
  motor.begin(MOTOR_BAUD);
  delay(200);
  
  motor.setWorkMode(MODE_UART);
  delay(100);
  motor.uartEnable(true);
  delay(100);
  motor.setMicrostep(16);
  delay(50);
  
  // Read initial position
  EncoderReading enc;
  for (int retry = 0; retry < 5; retry++) {
    if (motor.readEncoder(enc)) {
      int32_t initial_encoder = (enc.carry * ENCODER_CPR) + enc.value;
      current_positions[0] = ENCODER_TO_RAD(initial_encoder);
      target_positions[0] = current_positions[0];  // Start at current position
      last_target_pulses = RAD_TO_MOTOR_PULSES(current_positions[0]);  // Track in motor pulses
      
      Serial.print("✓ Motor initialized at ");
      Serial.print(current_positions[0], 3);
      Serial.println(" rad");
      
      motor_initialized = true;
      break;
    }
    delay(200);
  }
  
  if (!motor_initialized) {
    Serial.println("✗ Motor init FAILED!");
  }
  
  Serial.println("READY - Send commands like: <0,1.5,0,0,0,0,0>");
}

void loop() {
  // Read commands
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
  
  // Update at 10 Hz
  unsigned long now = millis();
  if (now - last_update >= UPDATE_INTERVAL_MS) {
    last_update = now;
    updateMotor();
    sendFeedback();
  }
}

void processCommand(String cmd) {
  int idx = cmd.indexOf(',');
  if (idx == -1) return;
  
  last_seq = cmd.substring(0, idx).toInt();
  
  int start = idx + 1;
  for (int i = 0; i < 6; i++) {
    int next = cmd.indexOf(',', start);
    if (next == -1 && i < 5) return;
    
    String val = (next == -1) ? cmd.substring(start) : cmd.substring(start, next);
    target_positions[i] = val.toFloat();
    start = next + 1;
  }
  
  // New target received - reset move flag
  int32_t new_target_pulses = RAD_TO_MOTOR_PULSES(target_positions[0]);
  if (new_target_pulses != last_target_pulses) {
    move_commanded = false;
    Serial.print("📥 New target: ");
    Serial.print(target_positions[0], 3);
    Serial.print(" rad (");
    Serial.print(new_target_pulses);
    Serial.println(" motor pulses)");
  }
}

void updateMotor() {
  if (!motor_initialized) {
    for (int i = 0; i < 6; i++) {
      current_positions[i] = target_positions[i];
    }
    return;
  }
  
  // Read current position
  EncoderReading enc;
  if (motor.readEncoder(enc)) {
    int32_t current_encoder = (enc.carry * ENCODER_CPR) + enc.value;
    current_positions[0] = ENCODER_TO_RAD(current_encoder);
    
    // Convert BOTH to motor pulses for delta calculation
    int32_t current_motor_pulses = RAD_TO_MOTOR_PULSES(current_positions[0]);
    int32_t target_motor_pulses = RAD_TO_MOTOR_PULSES(target_positions[0]);
    int32_t delta = target_motor_pulses - current_motor_pulses;
    
    // If target changed and we haven't moved yet
    if (!move_commanded && abs(delta) > 10) {  // 10 pulse deadband
      // Send ONE move command
      RunDirection dir = (delta > 0) ? RUN_FWD : RUN_REV;
      uint32_t pulses = abs(delta);
      uint8_t speed = 20;  // Slower for better control
      uint8_t status;
      
      Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      Serial.print("🎯 Moving: ");
      Serial.print(current_positions[0], 3);
      Serial.print(" → ");
      Serial.print(target_positions[0], 3);
      Serial.print(" rad");
      Serial.print("\n   Current: ");
      Serial.print(current_motor_pulses);
      Serial.print(" pulses");
      Serial.print("\n   Target:  ");
      Serial.print(target_motor_pulses);
      Serial.print(" pulses");
      Serial.print("\n   Delta:   ");
      Serial.print(delta);
      Serial.println(" pulses");
      
      motor.uartRunPulses(speed, dir, pulses, status);
      
      Serial.print("   Status: ");
      Serial.println(status);
      Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      
      move_commanded = true;
      last_target_pulses = target_motor_pulses;
    }
  }
  
  // Joints 2-6: echo
  for (int i = 1; i < 6; i++) {
    current_positions[i] = target_positions[i];
  }
}

void sendFeedback() {
  Serial.print("<ACK,");
  Serial.print(last_seq);
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    Serial.print(current_positions[i], 2);
  }
  Serial.println(">");
}

/*
 * NOTES:
 * - This sends ONE move command per target change
 * - Motor will move and stop
 * - For continuous tracking, you'd need velocity control or a PID loop
 * - This is suitable for point-to-point moves, not continuous trajectories
 * 
 * TESTING:
 * 1. Upload and open Serial Monitor
 * 2. Send: <0,1.5,0,0,0,0,0>
 * 3. Motor should move to 1.5 rad and STOP
 * 4. Send: <1,0.0,0,0,0,0,0>
 * 5. Motor should return to zero
 */
