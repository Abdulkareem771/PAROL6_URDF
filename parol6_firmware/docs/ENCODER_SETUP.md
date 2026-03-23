# MT6816 Encoder Setup & QuadTimer Debug Guide

> **Status: Verified ✅** — All 6 channels confirmed working (2026-03-02)

---

## Hardware Overview

The PAROL6 uses **MT6816 absolute magnetic encoders** (one per joint). Each encoder outputs a **PWM signal** whose duty cycle encodes the absolute angle:

| Parameter | Value |
|:--|:--|
| Output type | PWM (not A+B quadrature) |
| Frame frequency | ~971 Hz (4119 × 250 ns periods) |
| Resolution | 12-bit (4096 steps) |
| Angle mapping | duty = angle / 360° (range: 1/4096 to 4095/4096) |
| Logic level | 3.3 V |

The Teensy 4.1 reads these signals using its hardware **QuadTimers (TMR1/TMR3)** in **Gated Count Mode**, measuring how many IP Bus clock cycles the pin is HIGH per ISR tick.

---

## Wiring

| Joint | MT6816 PWM pin → | Teensy 4.1 pin | QuadTimer channel |
|:--|:--|:--|:--|
| J1 | → | **Pin 10** | TMR1 CH0 |
| J2 | → | **Pin 11** | TMR1 CH2 |
| J3 | → | **Pin 12** | TMR1 CH1 |
| J4 | → | **Pin 14** | TMR3 CH2 |
| J5 | → | **Pin 15** | TMR3 CH3 |
| J6 | → | **Pin 18** | TMR3 CH1 |

> **Common GND is required** between encoders and Teensy.

---

## How the QuadTimer Measurement Works

```
For each 1 ms ISR tick:

  1. Read CNTR (16-bit hardware counter)
  2. delta_hi = CNTR - last_CNTR     (16-bit, handles overflow)
  3. expected  = CPU_cycles / 4 / 8  (= 18,750 at 1ms, 600MHz/150MHz/8)
  4. duty      = delta_hi / expected  (= 0.0–1.0)
  5. angle     = duty × 2π  (radians)
```

**Key constraint:** The counter runs at IP Bus / 8 = 18.75 MHz.
In 1 ms it counts at most 18,750 ticks — safely within the 16-bit range (max 65,535).
The ISR **must** call `read_angle()` at ≥ 1 kHz to prevent counter overflow.

---

## QuadTimer Register Configuration

```cpp
// Correct setup (in QuadTimerEncoder::init())
tmr_->CH[ch_].COMP1 = 0xFFFF;           // free-run counter (no early reset)
tmr_->CH[ch_].CTRL  = TMR_CTRL_CM(3)    // Gated Count: count primary while secondary HIGH
                    | TMR_CTRL_PCS(8+3)  // Primary = IP Bus / 8 = 18.75 MHz
                    | TMR_CTRL_SCS(ch_); // Secondary = this channel's external pin
```

### ⚠️ Bugs Found and Fixed (2026-03-02)

| Bug | Incorrect value | Correct value | Symptom |
|:--|:--|:--|:--|
| Count mode | `CM(6)` | **`CM(3)`** | Reads ~0 always |
| Auto-reload | `TMR_CTRL_LENGTH` present | **Removed** | Reads 0 (COMP1=0 on reset, counter reloads instantly) |
| Compare register | `COMP1` = 0 (reset default) | **`COMP1 = 0xFFFF`** | Counter stuck at 0 with LENGTH set |
| Sampling rate | 100 ms poll | **1 ms ISR** | 28× overflow → residual ≈ 0 |

> **CM(6) description** (NXP RM): *"Count rising edges of secondary input while primary is LOW"*  
> **CM(3) description** (NXP RM): *"Count rising edges of primary while secondary input is HIGH"* ← correct

---

## IOMUXC Daisy-Chain Selectors

Pins 14, 15, 18 (TMR3) require daisy-chain routing. Pins 10, 11, 12 (TMR1) are **hardwired** — no selector needed.

```cpp
// TMR1 pins (10, 11, 12): NO select register needed — hardwired in silicon
// TMR3 pins: require explicit routing
IOMUXC_QTIMER3_TIMER2_SELECT_INPUT = 1;  // pin 14
IOMUXC_QTIMER3_TIMER3_SELECT_INPUT = 1;  // pin 15
IOMUXC_QTIMER3_TIMER1_SELECT_INPUT = 0;  // pin 18
```

> `IOMUXC_QTIMER1_TIMERx_SELECT_INPUT` does **not exist** in Teensyduino — do not use it.

---

## Verification Procedure

### Step 1 — QuadTimer hardware self-test (no ESP32 needed)

```bash
cd parol6_firmware/diagnostic/quadtimer_diagnostic
./flash_diagnostic.sh     # press Teensy reset button when prompted
```

| Test | Expected output |
|:--|:--|
| 3.3V jumper on pin 10 | `J1=1.0000(6.28rad) — TIMER WORKS!` |
| GND jumper on pin 10 | `J1=0.0000(0.00rad)` |
| ESP32 PWM 50% on pin 10 | `J1=0.5000(3.14rad) — J1 PASS` |

### Step 2 — MT6816 simulator (all 6 joints, no real encoders)

Flash the ESP32 with the simulator that sweeps all 6 joints at different rates:

```bash
cd esp32_feedback_firmware
./flash_encoder_sim.sh /dev/ttyUSB0
```

**Wiring (ESP32 GPIO → Teensy):**

| ESP32 GPIO | → | Teensy Pin | Joint |
|:--|:--|:--|:--|
| 18 | → | 10 | J1 |
| 19 | → | 11 | J2 |
| 21 | → | 12 | J3 |
| 22 | → | 14 | J4 |
| 23 | → | 15 | J5 |
| 25 | → | 18 | J6 |
| GND | → | GND | — |

All 6 joints will sweep at different rates (10–25°/s), wrapping correctly at 0/2π.

### Step 3 — GUI live display

1. Open the firmware configurator: `python3 main.py`
2. Connect to Teensy (`/dev/ttyACM1`, 115200)
3. Open **📈 Oscilloscope** tab
4. Check **J1–J6** boxes in the **Joint Positions** row
5. You should see 6 independent sweeping traces

---

## config.h Requirements

For the main firmware to use QuadTimer encoders, `generated/config.h` must contain:

```c
#define ENCODER_MODE 0   // 0 = QuadTimer PWM mode, 1 = interrupt mode
```

Set this via the **⚙️ Features** tab → **Encoder Mode = QuadTimer (0)**.

---

## Troubleshooting

| Symptom | Cause | Fix |
|:--|:--|:--|
| All zeros, 3.3V on pin reads 0 | Wrong CM mode or LENGTH bug | Verify `CM(3)`, no LENGTH, `COMP1=0xFFFF` |
| One joint reads 0, others OK | Wired to wrong Teensy pin | Use exact pin table above |
| Duty drifts at whole-number boundaries | 100ms poll with overflow | Ensure `read_angle()` called at 1 kHz ISR rate |
| `IOMUXC_QTIMER1_TIMER0_SELECT_INPUT` compile error | Doesn't exist for TMR1 | Remove — TMR1 pins are hardwired, no selector needed |
| Slight jitter (±1–2%) | Normal for 1ms window at 18.75 MHz | Acceptable; real encoder has 12-bit EEPROM averaging |
