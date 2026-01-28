#include "SERVO42C.h"

SERVO42C motor(Serial1, 0xE0);
float set_angle = 20.1;

void setup() {
    Serial.begin(115200);

    // MUST match motor menu baud
    motor.begin(38400);

    delay(200);

    // Switch to UART mode
    motor.setWorkMode(MODE_UART);
    delay(100);

    // ENABLE motor in UART mode (CRITICAL)
    motor.uartEnable(true);
    delay(100);
    int MicroStep= 16;
    // Known microstep
    motor.setMicrostep(MicroStep);
     delay(50);
    // motor.setACC(0x11E);
    //  delay(50);
    // // motor.setKd(0x300);
    // //  delay(50);
    // // motor.setKi(0x1E);
    // //  delay(50);
    // // motor.setKp(0x400);
    // // delay(50);
   

    Serial.println("Sending pulse move...");

    // 125 degrees @ 1.8°, 128 microstep
    float pulsesPerDegree = (200.0 * MicroStep) / 360.0;
    uint32_t pulses = (uint32_t)(set_angle * pulsesPerDegree);

    uint8_t status;
    motor.uartRunPulses(10, RUN_REV, pulses, status);
   
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
    }

    delay(50);
}
