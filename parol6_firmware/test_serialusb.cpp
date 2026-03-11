#include <Arduino.h>
void setup() {
    Serial.println("Serial exists.");
    // Force compiler to evaluate SerialUSB
    typeof(SerialUSB)* test_ptr = &SerialUSB;
}
void loop() {}
