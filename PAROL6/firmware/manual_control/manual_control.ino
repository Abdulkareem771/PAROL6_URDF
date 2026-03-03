/*
 * MANUAL CONTROL - Like minimal test
 * 
 * This version:
 * - Waits for command
 * - Sends it ONCE
 * - NEVER re-sends
 * - Just reads encoder for feedback
 * 
 * This matches your working minimal test behavior
 */

#include "SERVO42C.h"

#define USB_BAUD 115200
#define MOTOR_BAUD 38400
#define MOTOR_RX_PIN 16
#define MOTOR_TX_PIN 17
#define MOTOR_ADDR 0xE0

// Motor pulses (for commanding)
#define STEPS_PER_REV 200
#define MICROSTEPS 16
#define MOTOR_PULSES_PER_REV (STEPS_PER_REV * MICROSTEPS)  // 3200

// Encoder (for reading)
#define ENCODER_CPR 16384
#define ENCODER_TO_RAD(enc) ((float)((enc) * (2.0 * PI) / ENCODER_CPR))

SERVO42C motor(Serial2, MOTOR_ADDR);

String inputBuffer = "";
uint32_t last_seq = 0;

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  Serial.println("=== MANUAL MOTOR CONTROL ===");
  Serial.println("Commands: <SEQ,RAD,0,0,0,0,0>");
  Serial.println("Example: <0,1.5,0,0,0,0,0>");
  Serial.println();
  
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
  
  Serial.println("READY - Motor will move when you send commands");
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
  
  // Read encoder every 500ms
  static unsigned long last_read = 0;
  if (millis() - last_read >= 500) {
    last_read = millis();
    readAndPrintEncoder();
  }
}

void processCommand(String cmd) {
  int idx = cmd.indexOf(',');
  if (idx == -1) {
    Serial.println("ERROR: Invalid format");
    return;
  }
  
  last_seq = cmd.substring(0, idx).toInt();
  
  int start = idx + 1;
  int next = cmd.indexOf(',', start);
  if (next == -1) {
    Serial.println("ERROR: Missing position");
    return;
  }
  
  float target_rad = cmd.substring(start, next).toFloat();
  
  // Calculate pulses needed
  float target_deg = target_rad * (180.0 / PI);
  float pulses_per_deg = (float)MOTOR_PULSES_PER_REV / 360.0;
  int32_t target_pulses = (int32_t)(target_deg * pulses_per_deg);
  
  // Read current position
  EncoderReading enc;
  float current_rad = 0;
  int32_t current_pulses = 0;
  
  if (motor.readEncoder(enc)) {
    // CRITICAL: Properly handle encoder with carry
    // Encoder: 16384 counts/rev
    // Motor: 3200 pulses/rev
    // Ratio: 16384 / 3200 = 5.12 encoder counts per motor pulse
    
    int64_t total_encoder_counts = ((int64_t)enc.carry * ENCODER_CPR) + enc.value;
    current_rad = (float)(total_encoder_counts * (2.0 * PI) / ENCODER_CPR);
    
    // Convert encoder counts to motor pulses
    // motor_pulses = encoder_counts / 5.12
    current_pulses = (int32_t)(total_encoder_counts * MOTOR_PULSES_PER_REV / ENCODER_CPR);
  }
  
  int32_t delta = target_pulses - current_pulses;
  
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  Serial.print("📥 Command received:");
  Serial.print("\n   Target: ");
  Serial.print(target_rad, 3);
  Serial.print(" rad (");
  Serial.print(target_pulses);
  Serial.print(" pulses)");
  Serial.print("\n   Current: ");
  Serial.print(current_rad, 3);
  Serial.print(" rad (");
  Serial.print(current_pulses);
  Serial.print(" pulses)");
  Serial.print("\n   Delta: ");
  Serial.print(delta);
  Serial.println(" pulses");
  
  if (abs(delta) < 10) {
    Serial.println("✓ Already at target!");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    sendACK(target_rad);
    return;
  }
  
  // Send move command
  RunDirection dir = (delta > 0) ? RUN_FWD : RUN_REV;
  uint32_t pulses = abs(delta);
  uint8_t speed = 15;  // Very slow for testing
  uint8_t status;
  
  Serial.print("🎯 Moving ");
  Serial.print(dir == RUN_FWD ? "FORWARD" : "REVERSE");
  Serial.print(" ");
  Serial.print(pulses);
  Serial.println(" pulses...");
  
  motor.uartRunPulses(speed, dir, pulses, status);
  
  Serial.print("   Status: ");
  Serial.println(status);
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  
  sendACK(target_rad);
}

void readAndPrintEncoder() {
  EncoderReading enc;
  if (motor.readEncoder(enc)) {
    int64_t total_encoder_counts = ((int64_t)enc.carry * ENCODER_CPR) + enc.value;
    float rad = (float)(total_encoder_counts * (2.0 * PI) / ENCODER_CPR);
    
    Serial.print("📊 Position: ");
    Serial.print(rad, 3);
    Serial.print(" rad (carry=");
    Serial.print(enc.carry);
    Serial.print(" value=");
    Serial.print(enc.value);
    Serial.print(" total=");
    Serial.print((long)total_encoder_counts);
    Serial.println(")");
  }
}

void sendACK(float pos) {
  Serial.print("<ACK,");
  Serial.print(last_seq);
  Serial.print(",");
  Serial.print(pos, 2);
  Serial.println(",0.00,0.00,0.00,0.00,0.00>");
}

/*
 * TESTING:
 * 1. Upload and open Serial Monitor
 * 2. Motor should NOT move on startup
 * 3. Send: <0,0.5,0,0,0,0,0>
 * 4. Motor should move to ~0.5 rad and STOP
 * 5. Wait a few seconds, watch encoder readings
 * 6. Send: <1,0.0,0,0,0,0,0>
 * 7. Motor should return to zero
 * 
 * If motor still overshoots, the issue is with the MKS driver itself
 * or the microstepping/steps-per-rev settings don't match
 */
