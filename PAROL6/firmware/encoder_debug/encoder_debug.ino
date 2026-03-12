/*
 * PAROL6 Encoder + Motor Debug — Teensy 4.1
 *
 * Reads MT6816 encoders AND can step motors on command.
 * Open Arduino Serial Monitor — type commands:
 *
 *   '2' = Step J2 motor 200 steps (1 rev at no microstepping)
 *   '3' = Step J3 motor 200 steps
 *   's' = Stop all motors
 *   'r' = Reset revolution counters
 *
 * Encoder readings update at 10 Hz.
 * Motors step at a slow, safe speed (200 Hz = 1 rev/sec at 200 steps/rev).
 */

#define NUM_MOTORS 6

// ─── PIN CONFIG (matching realtime_servo_teensy_old) ────────────────────────
const int STEP_PINS[NUM_MOTORS] = {2, 4, 5, 8, 7, 8};
const int DIR_PINS[NUM_MOTORS]  = {24, 35, 35, 27, 40, 29};
const int ENC_PINS[NUM_MOTORS]  = {14, 12, 12, 17, 19, 18};

const bool ENC_ACTIVE[NUM_MOTORS] = {false, false, false, false, true, false};

// Gear ratios (for display only)
const float GEAR[NUM_MOTORS] = {1.0, 20.0, 16.5, 4.0, 10.0, 4.0};

// ─── MT6816 CONSTANTS ───────────────────────────────────────────────────────
const float CLOCK_NS     = 250.0;
const int   START_CLOCKS = 16;
const int   RESOLUTION   = 4096;

// ─── ENCODER ISR ────────────────────────────────────────────────────────────
static volatile uint32_t rise_us[NUM_MOTORS] = {0};
static volatile uint32_t pw_us[NUM_MOTORS]   = {0};
static volatile uint32_t isr_count[NUM_MOTORS] = {0};

#define MAKE_ISR(N) \
  void isr_##N() { \
    if (digitalReadFast(ENC_PINS[N])) rise_us[N] = micros(); \
    else { pw_us[N] = micros() - rise_us[N]; isr_count[N]++; } \
  }
MAKE_ISR(0) MAKE_ISR(1) MAKE_ISR(2)
MAKE_ISR(3) MAKE_ISR(4) MAKE_ISR(5)

static void (*isrs[NUM_MOTORS])() = {
  isr_0, isr_1, isr_2, isr_3, isr_4, isr_5
};

// ─── MULTI-TURN STATE ───────────────────────────────────────────────────────
static float last_motor_deg[NUM_MOTORS];
static int   revolutions[NUM_MOTORS];
static bool  first_reading[NUM_MOTORS];

// ─── MOTOR STEPPING STATE ───────────────────────────────────────────────────
static int  stepping_motor = -1;     // Which motor is stepping (-1 = none)
static int  steps_remaining = 0;
static elapsedMicros step_timer;
#define STEP_INTERVAL_US 5000       // 200 Hz step rate = slow, safe speed
#define STEP_PULSE_US    10         // 10µs pulse width

// ─── SETUP ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("=== PAROL6 Encoder + Motor Debug ===");
  Serial.println("Commands: '2'=step J2, '3'=step J3, 's'=stop, 'r'=reset revs");
  Serial.println();

  // Init encoders
  for (int i = 0; i < NUM_MOTORS; i++) {
    revolutions[i] = 0;
    last_motor_deg[i] = -999.0;
    first_reading[i] = true;

    if (ENC_ACTIVE[i]) {
      pinMode(ENC_PINS[i], INPUT);
      attachInterrupt(ENC_PINS[i], isrs[i], CHANGE);
    }
  }

  // Init motor pins (only J2 and J3 for safety)
  for (int i = 1; i <= 2; i++) {  // J2=index 1, J3=index 2
    pinMode(STEP_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
    digitalWrite(DIR_PINS[i], LOW);  // Forward direction
  }

  Serial.println("Joint | PulseWidth(us) | ISR_Count | MotorAngle(deg) | Revs | JointAngle(deg)");
  Serial.println("------+----------------+-----------+-----------------+------+----------------");
}

// ─── LOOP ───────────────────────────────────────────────────────────────────
void loop() {
  // ── Handle serial commands ──
  if (Serial.available()) {
    char c = Serial.read();
    switch (c) {
      case '2':
        stepping_motor = 1;  // J2 = index 1
        steps_remaining = 200;
        step_timer = 0;
        Serial.println(">>> Stepping J2: 200 steps...");
        break;
      case '3':
        stepping_motor = 2;  // J3 = index 2
        steps_remaining = 200;
        step_timer = 0;
        Serial.println(">>> Stepping J3: 200 steps...");
        break;
      case 's':
        stepping_motor = -1;
        steps_remaining = 0;
        Serial.println(">>> STOPPED");
        break;
      case 'r':
        for (int i = 0; i < NUM_MOTORS; i++) {
          revolutions[i] = 0;
          first_reading[i] = true;
        }
        Serial.println(">>> Revolution counters RESET");
        break;
    }
  }

  // ── Step motor if active ──
  if (stepping_motor >= 0 && steps_remaining > 0 && step_timer >= STEP_INTERVAL_US) {
    step_timer -= STEP_INTERVAL_US;
    
    // Single step pulse
    digitalWriteFast(STEP_PINS[stepping_motor], HIGH);
    delayMicroseconds(STEP_PULSE_US);
    digitalWriteFast(STEP_PINS[stepping_motor], LOW);
    
    steps_remaining--;
    if (steps_remaining == 0) {
      Serial.printf(">>> J%d: 200 steps DONE\n", stepping_motor + 1);
      stepping_motor = -1;
    }
  }

  // ── Print encoder readings at 10 Hz ──
  static elapsedMillis print_timer;
  if (print_timer >= 100) {
    print_timer -= 100;

    for (int i = 0; i < NUM_MOTORS; i++) {
      if (!ENC_ACTIVE[i]) continue;

      uint32_t pw = pw_us[i];
      uint32_t count = isr_count[i];

      if (pw == 0) {
        Serial.printf("J%d    | NO SIGNAL      | %9lu |       ---       |  --- |      ---\n",
                       i + 1, (unsigned long)count);
        continue;
      }

      float clocks = (float)pw / (CLOCK_NS / 1000.0f);
      float counts = clocks - (float)START_CLOCKS;
      if (counts < 0.0f) counts = 0.0f;
      if (counts >= (float)RESOLUTION) counts = (float)(RESOLUTION - 1);

      float motor_deg = (counts / (float)RESOLUTION) * 360.0f;

      if (first_reading[i]) {
        last_motor_deg[i] = motor_deg;
        first_reading[i] = false;
      } else {
        float delta = motor_deg - last_motor_deg[i];
        if (delta > 180.0f)  revolutions[i]--;
        if (delta < -180.0f) revolutions[i]++;
        last_motor_deg[i] = motor_deg;
      }

      float total_motor_deg = motor_deg + revolutions[i] * 360.0f;
      float joint_deg = total_motor_deg / GEAR[i];

      Serial.printf("J%d    | %10lu us  | %9lu | %12.1f deg | %4d | %12.1f deg\n",
                     i + 1,
                     (unsigned long)pw,
                     (unsigned long)count,
                     motor_deg,
                     revolutions[i],
                     joint_deg);
    }

    // Show stepping status
    if (stepping_motor >= 0) {
      Serial.printf("       STEPPING J%d: %d steps remaining\n", stepping_motor + 1, steps_remaining);
    }
    Serial.println("------+----------------+-----------+-----------------+------+----------------");
  }
}
