#include <Arduino.h>

namespace {
constexpr uint8_t LED_PIN = PC13;
constexpr unsigned long BLINK_MS = 500;
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
}

void loop() {
  digitalWrite(LED_PIN, LOW);
  delay(BLINK_MS);
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_MS);
}
