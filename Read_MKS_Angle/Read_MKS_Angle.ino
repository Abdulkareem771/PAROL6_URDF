const int pwmPin = 27;

volatile uint32_t rise_t = 0;
volatile uint32_t pulse_high_us = 0;

void IRAM_ATTR pwm_isr() {
  static uint32_t last_fall = 0;

  if (digitalRead(pwmPin)) {
    rise_t = micros();
  } else {
    uint32_t now = micros();
    pulse_high_us = now - rise_t; 
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(pwmPin, INPUT);
  attachInterrupt(digitalPinToInterrupt(pwmPin), pwm_isr, CHANGE);
}

void loop() {
  uint32_t us;

  noInterrupts();
  us = pulse_high_us;
  interrupts();

  // =====================================================
  // 1. Convert high pulse width to number of clocks
  // Try 125 ns first (common on MKS boards)
  // =====================================================

  float clock_period_ns = 250.0;  // or 250.0 if needed
  float clocks = (us * 1000.0) / clock_period_ns;

  // =====================================================
  // 2. Remove the 16-clock start pattern
  // =====================================================

  float angle_clocks = clocks - 16.0;

  // Clamp
  if (angle_clocks < 0) angle_clocks = 0;
  if (angle_clocks > 4095) angle_clocks = 4095;

  // =====================================================
  // 3. Convert to degrees
  // =====================================================

  float angle_deg = angle_clocks * (360.0 / 4096.0);

  Serial.println(angle_deg);

  delay(5);
}
