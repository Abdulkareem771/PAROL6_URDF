/*
 * MKS Servo42C Encoder Reading Test (vFOC Mode)
 * 
 * This script continuously reads encoder values from the MKS motor
 * while it's in CR_vFOC (Step/Dir) mode.
 * 
 * Setup:
 * 1. Set MKS to CR_vFOC mode using the screen/UART config
 * 2. Upload this sketch
 * 3. Open Serial Monitor at 115200 baud
 * 4. Manually rotate the motor shaft
 * 5. Watch encoder values update
 */

#include "SERVO42C.h"

#define USB_BAUD 115200
#define MOTOR_BAUD 115200  // MKS UART baud rate
#define MOTOR_RX_PIN 16
#define MOTOR_TX_PIN 17
#define MOTOR_ADDR 0xE0

SERVO42C motor(Serial2, MOTOR_ADDR);

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial) delay(10);
  
  Serial.println("=== MKS Servo42C Encoder Test (vFOC Mode) ===");
  Serial.println("Make sure motor is in CR_vFOC mode!");
  Serial.println();
  
  // Initialize motor serial
  Serial2.begin(MOTOR_BAUD, SERIAL_8N1, MOTOR_RX_PIN, MOTOR_TX_PIN);
  delay(100);
  
  motor.begin(MOTOR_BAUD);
  delay(200);
  
  Serial.println("Starting encoder readings...");
  Serial.println("Rotate the motor shaft manually to see values change");
  Serial.println();
}

void loop() {
  // Read encoder
  EncoderReading enc;
  bool success = motor.readEncoder(enc);
  
  if (success) {
    // Calculate total position in encoder counts
    int64_t total_counts = ((int64_t)enc.carry * 65536) + enc.value;
    
    // Convert to degrees (16384 counts per revolution)
    float degrees = (total_counts * 360.0) / 16384.0;
    
    // Convert to radians
    float radians = degrees * (PI / 180.0);
    
    // Print results
    Serial.print("Encoder - Carry: ");
    Serial.print(enc.carry);
    Serial.print("  Value: ");
    Serial.print(enc.value);
    Serial.print("  Total: ");
    Serial.print(total_counts);
    Serial.print("  Deg: ");
    Serial.print(degrees, 2);
    Serial.print("°  Rad: ");
    Serial.println(radians, 4);
  } else {
    Serial.println("❌ Failed to read encoder!");
  }
  
  delay(100);  // Read at 10 Hz
}
