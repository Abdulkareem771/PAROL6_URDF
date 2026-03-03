/*
 * MINIMAL TEST - Based on your working Arduino Mega example
 * 
 * This is the SIMPLEST possible test:
 * 1. Send ONE test command on startup
 * 2. Read encoder continuously
 * 3. No ROS, no complex logic
 */

#include "SERVO42C.h"

// Use Serial2 for motor (ESP32)
SERVO42C motor(Serial2, 0xE0);

void setup() {
    Serial.begin(115200);
    Serial.println("=== MINIMAL MOTOR TEST ===");
    
    // Initialize motor serial (MUST match motor baud)
    Serial2.begin(38400, SERIAL_8N1, 16, 17);  // RX=16, TX=17
    delay(100);
    
    motor.begin(38400);
    delay(200);
    
    Serial.println("Setting work mode to UART...");
    motor.setWorkMode(MODE_UART);
    delay(100);
    
    Serial.println("Enabling UART control...");
    motor.uartEnable(true);  // CRITICAL!
    delay(100);
    
    Serial.println("Setting microstep to 16...");
    motor.setMicrostep(16);
    delay(50);
    
    Serial.println("Sending test move: 20 degrees...");
    
    // Calculate pulses for 20 degrees
    // 200 steps/rev * 16 microsteps = 3200 pulses/rev
    // 3200 / 360 = 8.889 pulses/degree
    float pulsesPerDegree = (200.0 * 16) / 360.0;
    uint32_t pulses = (uint32_t)(180.0 * pulsesPerDegree);
    
    Serial.print("Pulses: ");
    Serial.println(pulses);
    
    uint8_t status;
    motor.uartRunPulses(10, RUN_FWD, pulses, status);
    
    Serial.print("Status: ");
    Serial.println(status);
    Serial.println("=== Motor should move now! ===");
}

void loop() {
    EncoderReading enc;
    
    if (motor.readEncoder(enc)) {
        float angle = enc.value * (360.0f / 65536.0f);
        
        Serial.print("carry=");
        Serial.print(enc.carry);
        Serial.print(" value=");
        Serial.print(enc.value);
        Serial.print(" angle=");
        Serial.println(angle, 2);
    } else {
        Serial.println("Encoder read FAILED!");
    }
    
    delay(500);  // Read every 500ms
}
