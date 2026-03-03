/*
 * PAROL6 ESP32 Firmware - Single Motor Test (FIXED)
 * 
 * FIX: Motor won't move until it receives first ROS command
 * 
 * Hardware:
 *   - ESP32 USB ←→ PC (ROS @ 115200 baud)
 *   - ESP32 Serial2 (GPIO16 TX, GPIO17 RX) ←→ MKS Servo42C (@ 38400 baud)
 */

#include "SERVO42C.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

#define USB_BAUD 115200
#define MOTOR_BAUD 38400

// Motor Serial Port
#define MOTOR_RX_PIN 16
#define MOTOR_TX_PIN 17

// Motor Parameters
#define MOTOR_ADDR 0xE0
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define GEAR_RATIO 1.0

// Conversion (MKS encoder: 16384 counts/rev)
#define ENCODER_CPR 16384
#define RAD_TO_ENCODER(rad) ((int32_t)((rad) * ENCODER_CPR / (2.0 * PI)))
#define ENCODER_TO_RAD(enc) ((float)((enc) * (2.0 * PI) / ENCODER_CPR))

// Timing
#define UPDATE_INTERVAL_MS 40   // 25 Hz

// ============================================================================
// GLOBALS
// ============================================================================

// Use Serial2 directly (pre-defined on ESP32)
SERVO42C motor(Serial2, MOTOR_ADDR);

// Joint States
float target_positions[6] = {0, 0, 0, 0, 0, 0};
float current_positions[6] = {0, 0, 0, 0, 0, 0};

// Communication
String inputBuffer = "";
uint32_t last_seq = 0;
unsigned long last_update = 0;

// Status
bool motor_initialized = false;
bool first_command_received = false;  // NEW: Don't move until commanded
int32_t initial_encoder = 0;          // NEW: Store initial position

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  Serial.println("==============================================");
  Serial.println("  PAROL6 Single Motor Test - FIXED");
  Serial.println("==============================================");
  
  // Motor Serial - Initialize Serial2 hardware BEFORE library
  Serial.println("Initializing Serial2 for motor...");
  Serial2.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);
  delay(100);
  
  // Initialize SERVO42C library (will call Serial2.begin again, but that's OK)
  Serial.println("Initializing MKS Servo42C library...");
  motor.begin(MOTOR_BAUD);
  delay(200);
  
  // Configure motor
  Serial.println("Configuring motor...");
  motor.setWorkMode(MODE_UART);
  delay(100);
  
  // CRITICAL: Enable UART control (from working example)
  motor.uartEnable(true);
  delay(100);
  
  motor.setEnablePin(EN_ALWAYS);
  motor.setMotorType(MOTOR_1_8_DEG);
  motor.setMicrostep(16);
  motor.setCurrent(50);  // 50% current for safety
  
  delay(100);
  
  // Read initial encoder position (with retries - motor needs time to respond)
  Serial.println("Reading initial encoder position...");
  EncoderReading enc;
  bool encoder_ok = false;
  
  for (int retry = 0; retry < 5; retry++) {
    if (motor.readEncoder(enc)) {
      initial_encoder = (enc.carry * ENCODER_CPR) + enc.value;
      current_positions[0] = ENCODER_TO_RAD(initial_encoder);
      
      Serial.print("✓ Encoder OK (attempt ");
      Serial.print(retry + 1);
      Serial.print("): ");
      Serial.print(initial_encoder);
      Serial.print(" counts = ");
      Serial.print(current_positions[0], 3);
      Serial.println(" rad");
      
      motor_initialized = true;
      encoder_ok = true;
      break;
    } else {
      Serial.print("  Retry ");
      Serial.print(retry + 1);
      Serial.println("/5...");
      delay(200);  // Wait before retry
    }
  }
  
  if (!encoder_ok) {
    Serial.println("✗ Encoder read FAILED after 5 attempts!");
    Serial.println("  Check: wiring, baud rate, motor power");
    motor_initialized = false;
  }
  
  // IMPORTANT: Set target to current position (don't move!)
  target_positions[0] = current_positions[0];
  
  Serial.println("----------------------------------------------");
  Serial.println("Configuration:");
  Serial.print("  Steps/rev: "); Serial.println(STEPS_PER_REV);
  Serial.print("  Microsteps: "); Serial.println(MICROSTEPS);
  Serial.print("  Initial position: "); Serial.print(current_positions[0], 3); Serial.println(" rad");
  Serial.println("----------------------------------------------");
  Serial.println("READY - Motor will NOT move until ROS command");
  Serial.println();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

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
  
  // Update at 25 Hz
  unsigned long now = millis();
  if (now - last_update >= UPDATE_INTERVAL_MS) {
    last_update = now;
    
    // DEBUG: Show what we're doing
    static int loop_counter = 0;
    if (loop_counter++ % 25 == 0) {  // Every second
      Serial.print("⏱️ Loop running | first_cmd=");
      Serial.print(first_command_received ? "YES" : "NO");
      Serial.print(" | motor_init=");
      Serial.println(motor_initialized ? "YES" : "NO");
    }
    
    if (first_command_received) {
      Serial.println("🔄 Calling updateMotor()...");
      updateMotor();  // Only update if we've received a command
    } else {
      readEncoder();  // Just read position, don't move
    }
    
    sendFeedback();
  }
}

// ============================================================================
// COMMAND PROCESSING
// ============================================================================

void processCommand(String cmd) {
  // Expected: "SEQ,J1,J2,J3,J4,J5,J6"
  
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  Serial.print("📥 RECEIVED: <");
  Serial.print(cmd);
  Serial.println(">");
  
  int idx = cmd.indexOf(',');
  if (idx == -1) {
    Serial.println("❌ ERROR: No comma found!");
    return;
  }
  
  last_seq = cmd.substring(0, idx).toInt();
  Serial.print("   SEQ: ");
  Serial.println(last_seq);
  
  // Parse joint positions
  int start = idx + 1;
  for (int i = 0; i < 6; i++) {
    int next = cmd.indexOf(',', start);
    if (next == -1 && i < 5) {
      Serial.println("❌ ERROR: Missing joint data!");
      return;
    }
    
    String val = (next == -1) ? cmd.substring(start) : cmd.substring(start, next);
    target_positions[i] = val.toFloat();
    
    if (i == 0) {  // Only print J1 (the one we control)
      Serial.print("   J1 target: ");
      Serial.print(target_positions[i], 4);
      Serial.println(" rad");
    }
    
    start = next + 1;
  }
  
  // Mark that we've received first command
  if (!first_command_received) {
    first_command_received = true;
    Serial.println("✅ First command received - motor control ENABLED");
  }
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

// ============================================================================
// READ ENCODER ONLY
// ============================================================================

void readEncoder() {
  if (!motor_initialized) return;
  
  EncoderReading enc;
  if (motor.readEncoder(enc)) {
    int32_t encoder_value = (enc.carry * ENCODER_CPR) + enc.value;
    current_positions[0] = ENCODER_TO_RAD(encoder_value);
  }
  
  // Joints 2-6: no motors
  for (int i = 1; i < 6; i++) {
    current_positions[i] = target_positions[i];
  }
}

// ============================================================================
// MOTOR UPDATE (ONLY AFTER FIRST COMMAND)
// ============================================================================

void updateMotor() {
  if (!motor_initialized) {
    // No motor - echo targets
    for (int i = 0; i < 6; i++) {
      current_positions[i] = target_positions[i];
    }
    return;
  }
  
  // Read current position
  EncoderReading enc;
  int32_t current_encoder = 0;
  
  if (motor.readEncoder(enc)) {
    current_encoder = (enc.carry * ENCODER_CPR) + enc.value;
    current_positions[0] = ENCODER_TO_RAD(current_encoder);
    
    // Debug: Show encoder reading
    static int debug_counter = 0;
    if (debug_counter++ % 10 == 0) {  // Every 10 updates (~400ms)
      Serial.print("📊 Encoder: carry=");
      Serial.print(enc.carry);
      Serial.print(" value=");
      Serial.print(enc.value);
      Serial.print(" → ");
      Serial.print(current_encoder);
      Serial.print(" counts = ");
      Serial.print(current_positions[0], 4);
      Serial.println(" rad");
    }
  } else {
    Serial.println("⚠️ Encoder read FAILED!");
    return;
  }
  
  // Calculate target in encoder counts
  int32_t target_encoder = RAD_TO_ENCODER(target_positions[0]);
  int32_t delta = target_encoder - current_encoder;
  
  // State machine to prevent sending multiple move commands
  static bool move_in_progress = false;
  static unsigned long move_start_time = 0;
  static int32_t last_target = 0;
  
  // Check if target changed (new command)
  if (target_encoder != last_target) {
    move_in_progress = false;  // Reset for new target
    last_target = target_encoder;
  }
  
  // Deadband: only move if error > 50 counts (~0.02 radians)
  if (abs(delta) > 50) {
    
    // Only send command if no move in progress
    if (!move_in_progress) {
      RunDirection dir = (delta > 0) ? RUN_FWD : RUN_REV;
      uint32_t pulses = abs(delta);
      uint8_t speed = 30;  // Slow speed for testing
      uint8_t status;
      
      Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      Serial.print("🎯 STARTING MOVE:");
      Serial.print("\n   Current: ");
      Serial.print(current_encoder);
      Serial.print(" counts (");
      Serial.print(current_positions[0], 4);
      Serial.print(" rad)");
      Serial.print("\n   Target:  ");
      Serial.print(target_encoder);
      Serial.print(" counts (");
      Serial.print(target_positions[0], 4);
      Serial.print(" rad)");
      Serial.print("\n   Delta:   ");
      Serial.print(delta);
      Serial.print(" counts");
      Serial.print("\n   Direction: ");
      Serial.println(dir == RUN_FWD ? "FORWARD" : "REVERSE");
      Serial.print("   Pulses: ");
      Serial.print(pulses);
      Serial.print(" @ speed ");
      Serial.println(speed);
      
      motor.uartRunPulses(speed, dir, pulses, status);
      
      Serial.print("   Status: ");
      Serial.println(status);
      Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      
      move_in_progress = true;
      move_start_time = millis();
    } else {
      // Move in progress - check timeout
      if (millis() - move_start_time > 5000) {  // 5 second timeout
        Serial.println("⚠️ Move timeout - resetting");
        move_in_progress = false;
      }
    }
  } else {
    // Within deadband - move complete
    if (move_in_progress) {
      Serial.println("✅ Move complete - within deadband");
      move_in_progress = false;
    }
  }
  
  // Joints 2-6: echo
  for (int i = 1; i < 6; i++) {
    current_positions[i] = target_positions[i];
  }
}

// ============================================================================
// FEEDBACK
// ============================================================================

void sendFeedback() {
  Serial.print("<ACK,");
  Serial.print(last_seq);
  
  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    Serial.print(current_positions[i], 2);
  }
  
  Serial.println(">");
}

// ============================================================================
// NOTES
// ============================================================================

/*
 * CHANGES FROM PREVIOUS VERSION:
 *   1. Motor doesn't move until first ROS command received
 *   2. Initial position is read and set as target
 *   3. Slower speed (30 instead of 50) for safer testing
 *   4. Larger deadband (50 counts) to prevent oscillation
 *   5. Simplified encoder conversion
 * 
 * TESTING:
 *   1. Upload firmware
 *   2. Motor should NOT move on startup
 *   3. Send test: <0,0.1,0,0,0,0,0>
 *   4. Motor should move smoothly to 0.1 rad
 *   5. Send test: <1,0.0,0,0,0,0,0>
 *   6. Motor should return to zero
 */
