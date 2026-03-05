# Legacy Code Migration Guide

> **Authoritative Source Priority Order:**
> 1. `legacy code open loop/PAROL6 control board main software/` — **DEFINITIVE** (real STM32 firmware shipped with the robot, actual gear ratios, actual homing logic)
> 2. `PAROL6/firmware/realtime_servo_teensy/` — Experimental Teensy bring-up (different gear ratios, different microstep — NOT authoritative for gear ratios)
> 3. `PAROL6/firmware/realtime_servo_control/` — ESP32 prototype (similar approach, ESP32-specific)

---

## 1. Definitive Hardware Constants

### 1.1 Pin Map (Teensy 4.1) — QuadTimer-Safe

> [!CAUTION]
> The encoder pins for the `QuadTimerEncoder` are **LOCKED** at Zone 1: `[10, 11, 12, 14, 15, 18]`.
> These pins are hardwired in `QuadTimerEncoder.h` via IOMUXC registers.
> Any STEP, DIR, or other peripheral pin assignment that overlaps these pins **will silently break encoder capture**.
>
> The `realtime_servo_teensy` encoder pins `[14-19]` overlap with our QuadTimer set at `14, 15, 18` — that config is **incompatible** with our architecture and cannot be used verbatim.

| Joint | Role | Pin | Zone | Notes |
| :--- | :--- | :--- | :--- | :--- |
| J1 | Encoder | **10** | Zone 1 LOCKED | QuadTimer1 Ch0, IOMUXC alt-1 |
| J2 | Encoder | **11** | Zone 1 LOCKED | QuadTimer1 Ch2 |
| J3 | Encoder | **12** | Zone 1 LOCKED | QuadTimer1 Ch1 |
| J4 | Encoder | **14** | Zone 1 LOCKED | QuadTimer3 Ch2 |
| J5 | Encoder | **15** | Zone 1 LOCKED | QuadTimer3 Ch3 |
| J6 | Encoder | **18** | Zone 1 LOCKED | QuadTimer3 Ch1 |
| J1 | STEP | **2** | Zone 2 | FlexPWM4.2A |
| J2 | STEP | **6** | Zone 2 | FlexPWM2.2A |
| J3 | STEP | **7** | Zone 2 | FlexPWM1.3B |
| J4 | STEP | **8** | Zone 2 | FlexPWM1.3A |
| J5 | STEP | **4** | Zone 2 | FlexPWM2.0A |
| J6 | STEP | **5** | Zone 2 | FlexPWM2.1A |
| J1 | DIR | **30** | Zone 3 | Pure GPIO |
| J2 | DIR | **31** | Zone 3 | Pure GPIO |
| J3 | DIR | **32** | Zone 3 | Pure GPIO |
| J4 | DIR | **33** | Zone 3 | Pure GPIO |
| J5 | DIR | **34** | Zone 3 | Pure GPIO |
| J6 | DIR | **35** | Zone 3 | Pure GPIO |

> **Why Zone 3 for DIR (not [24-29] from `realtime_servo_teensy`):**
> Pins 28 and 29 are listed as Zone 2 candidates (FlexPWM-capable). Using them for DIR creates an irresolvable conflict if we ever need those FlexPWM submodules for STEP generation. Zone 3 pins 30-39 have no special peripheral functions and are safe for pure digital output.

---

### 1.2 Gear Ratios — Authoritative (from `motor_init.cpp`, STM32 legacy)

> [!IMPORTANT]
> The `realtime_servo_teensy` had WRONG gear ratios: J1=20, J2=1. These are INCORRECT.
> The real PAROL6 robot uses the following ratios, proven by the mechanical design:

| Joint | Gear Ratio | Calculation | Microstep | Steps/Rad |
| :--- | :--- | :--- | :--- | :--- |
| J1 | **6.4** | 96 / 15 | 32 | `200×32×6.4 / (2π) ≈ 6514` |
| J2 | **20.0** | 20:1 gearbox | 32 | `200×32×20.0 / (2π) ≈ 20372` |
| J3 | **18.0952381** | 20 × (38/42) | 32 | `200×32×18.095 / (2π) ≈ 18415` |
| J4 | **4.0** | 4:1 gearbox | 32 | `200×32×4.0 / (2π) ≈ 4074` |
| J5 | **4.0** | 4:1 gearbox | 32 | `200×32×4.0 / (2π) ≈ 4074` |
| J6 | **10.0** | 10:1 gearbox | 32 | `200×32×10.0 / (2π) ≈ 10186` |

**Global microstep:** `32` (from `constants.h`: `#define MICROSTEP 32`)

**Direction inversion** (from `direction_reversed` field in `motor_init.cpp`):
- J1: **reversed** | J2: normal | J3: **reversed** | J4: normal | J5: normal | J6: **reversed**

**Direction sign in firmware:** use `-1` in `MOTOR_DIR_SIGN` for reversed joints.

---

### 1.3 Validated Velocity Limits (rad/s at joint output shaft)

From `realtime_servo_teensy/config.h` (experimentally validated):

| Joint | Max (rad/s) | Reason |
| :--- | :--- | :--- |
| J1 | 3.0 | High gear ratio, conservative |
| J2 | 3.0 | Large arm, conservative |
| J3 | 6.0 | Elbow |
| J4 | 6.0 | Wrist |
| J5 | 6.0 | Wrist |
| J6 | 6.0 | Wrist end |

---

## 2. Legacy Open-Loop Firmware Analysis (`motor_init.cpp`)

### 2.1 Homing: Two-Stage Approach (per joint)

The legacy code implements a 2-stage homing per joint (`homing_stage_1`, `homing_stage_2`), coordinated across groups. Each stage:
- **Stage 1:** Drive toward limit at `homing_speed` until switched triggered
- **Stage 2:** Back off by `homed_position` steps to reach zero (URDF home frame)

**Homing order is inter-dependent** — the legacy code homes two groups in sequence:
- **Group A** (J4, J5, J6): Home first (wrist joints, shorter reach, lower collision risk)
- **Group B** (J1, J2, J3): Home second, all three in parallel

> [!WARNING]
> J5 must partially move its arm out of the way before J6 can safely home. This mechanical coupling is baked into the legacy homing sequence as `J5_stage4` etc. — it MUST be replicated in the firmware homing FSM or a physical collision will occur.

### 2.2 Home Position Offsets (in steps at 32 microstep, from `motor_init.cpp`)

| Joint | `homed_position` (steps from limit) | `standby_position` (operational home) |
| :--- | :--- | :--- |
| J1 | 13500 | 10240 |
| J2 | 19588 | -32000 |
| J3 | 23020 | 57905 |
| J4 | -10200 | 0 |
| J5 | 8900 | 0 |
| J6 | 15900 | 32000 |

These step counts represent the offset from the limit switch trigger point to the URDF zero position.

### 2.3 Motor Driver (Legacy: TMC5160 SPI, New: MKS Servo42C STEP/DIR)

The original board used TMC5160 drivers over SPI. The new Teensy 4.1 architecture uses MKS Servo42C stepper servo drivers, which accept standard STEP/DIR signals. This simplifies the firmware significantly — no SPI management, no current tuning via firmware.

### 2.4 Limit Switch Polarity (from `hw_init.cpp`)

| Limit Switch | Joint | Trigger Edge |
| :--- | :--- | :--- |
| LIMIT1 | J6 | FALLING |
| LIMIT2 | J2 | RISING |
| LIMIT3 | J3 | RISING |
| LIMIT4 | J4 | RISING (but `limit_switch_trigger=0` in motor_init) |
| LIMIT5 | J5 | RISING |
| LIMIT6 | J1 | FALLING |

---

## 3. FlexPWM Step Generation Pattern

### 3.1 `analogWriteFrequency` Usage (Verified on Teensy 4.1)

```cpp
// Init (each STEP pin maps to different FlexPWM submodule — no timer sharing)
analogWriteResolution(8);                // 8-bit = 0..255
analogWriteFrequency(STEP_PINS[i], 1000); // Initial frequency
analogWrite(STEP_PINS[i], 0);           // Start STOPPED (0% duty)

// Runtime frequency update (only affects this pin's submodule)
if (freq_hz < 10.0f) {
    analogWrite(STEP_PINS[i], 0);       // Stop: duty=0
} else {
    analogWriteFrequency(STEP_PINS[i], freq_hz);
    analogWrite(STEP_PINS[i], 128);     // Run: 50% duty
}
```

### 3.2 Direction Change Ordering

Always set DIR pin **before** calling `analogWriteFrequency`. At 1 kHz ISR, 1 ms of DIR setup time is far more than the MKS Servo42C's 2 µs requirement. However, for correctness, still follow this order:

```cpp
digitalWriteFast(DIR_PINS[i], direction ? HIGH : LOW);
analogWriteFrequency(STEP_PINS[i], freq_hz);  // Never before DIR
```

### 3.3 Velocity Deadband

```cpp
// Suppress micro-stepping noise at near-standstill
if (fabsf(velocity_command_rad_s) < 0.02f) velocity_command_rad_s = 0.0f;
```

---

## 4. Anti-Glitch Filter Pattern (Missing from `AlphaBetaFilter`)

The working `realtime_servo_teensy/control.cpp` adds glitch rejection before the multi-turn counter update. This is **critical** — a single bad encoder read will increment the revolution counter and permanently offset the position by `2π/GEAR_RATIO`.

```cpp
// Compute max possible angular delta in one control tick
float max_possible_delta = MAX_JOINT_VEL[i] * GEAR_RATIOS[i]
                         * control_period_s * 5.0f;  // 5x safety margin
if (max_possible_delta < 0.3f) max_possible_delta = 0.3f;

float raw_delta = current_raw_angle - last_raw_angle;
// Unwrap to [-π, +π]
if (raw_delta >  PI) raw_delta -= 2.0f * PI;
if (raw_delta < -PI) raw_delta += 2.0f * PI;

if (fabsf(raw_delta) > max_possible_delta) {
    return last_valid_position;  // Reject glitch, hold estimate
}
// ... proceed with multi-turn tracking and AlphaBetaFilter update
```
