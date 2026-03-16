# PAROL6 Firmware Bring-Up Testing Protocol

Each phase maps to a **saved config preset** in the GUI. Load the preset, flash, test, save notes.  
Do **NOT** skip phases. Each one catches a class of bug before it becomes dangerous.

> [!NOTE]
> This protocol covers both **Teensy 4.1** and **STM32F411CE BlackPill** targets.  
> STM32-specific steps are marked with 🔵.

---

## Equipment Checklist (Before Powering On)

| Item | Purpose |
|:--|:--|
| Oscilloscope (2+ channel) | STEP/DIR signal integrity. Not optional. |
| Bench PSU with current limit | Set to 24 V, limit to 0.5 A initially |
| USB cable (board → PC) | Flashing and serial monitor |
| Multimeter | Continuity, motor coil resistance |
| Logic analyzer (optional) | Captures all 6 STEP lines simultaneously |
| 🔵 STM32 only: confirm PA10 disconnected | BlackPill DFU errata — PA10 must float |

**Power-on sequence:**
1. Flash firmware with current phase config
2. Turn on bench PSU — watch current meter (should be <50 mA idle)
3. Open serial monitor in GUI
4. Proceed to phase tests

---

## Phase 0 — Hardware Sanity (No Firmware)

**Config preset:** `phase0_hardware_check`  
**Feature flags:** ALL OFF — `ENCODER_TEST_MODE=1` only  
**Risk:** None

### Checklist
- [ ] Continuity between Teensy STEP pins and motor driver STEP inputs
- [ ] Continuity between Teensy DIR pins and motor driver DIR inputs  
- [ ] Continuity between encoder MT6816 PWM out and Teensy encoder pins
- [ ] No shorts between adjacent pins (use DMM diode mode between adjacent pads)
- [ ] Motor coil resistance: ~2–5 Ω per coil (high resistance = open winding)

### Pass Criteria
All continuity checks pass, no unexpected shorts, power supply idle current <50 mA.

---

## Phase 1 — Encoder Wiring (Software Interrupt, Hand-turn Motor)

**Config preset:** `phase1_encoder_interrupt`  
**Feature flags:** `ENCODER_MODE=INTERRUPT`, `CONTROL_LOOP=OFF`  
**Risk:** Zero (no motor power output)

### Test
1. Flash preset → open Jog tab → enable "Encoder Test Mode"
2. Turn motor shaft **by hand** slowly (CW then CCW)
3. Watch the raw angle readout in the Joints tab

### Pass Criteria
- Raw angle changes smoothly (no jumps > 0.3 rad)
- CW rotation → increasing angle OR correctly marked as inverted
- Stationary motor → angle stable ±0.005 rad

### Fail Indicators
| Symptom | Cause |
|:--|:--|
| Angle stuck at 0 | No encoder signal — check cable continuity |
| Wild jumping | Ground loop — shield encoder cable |
| Wrong direction | Set `DIR_INVERT=1` in Joints tab |

---

## Phase 2 — QuadTimer Encoder vs Interrupt Comparison

**Config preset:** `phase2_quadtimer`  
**Feature flags:** `ENCODER_MODE=QUADTIMER`, `CONTROL_LOOP=OFF`

### Test
1. Repeat Phase 1 hand-turn test with QuadTimer mode
2. Watch ISR profiler gauge in status bar

### Pass Criteria
- Same angle readings as Phase 1 (proves QuadTimer wiring is correct)
- ISR profiler time **lower** than Phase 1 (QuadTimer offloads CPU)
- Typical: Phase 1 ~8–12 µs ISR time, Phase 2 ~2–4 µs

### Notes
If QuadTimer reads wrong value: check `IOMUXC_SELECT_INPUT` register in `QuadTimerEncoder.h` matches your exact Teensy 4.1 revision.

---

## Phase 3 — STEP/DIR Signal Integrity (Oscilloscope Required)

**Config preset:** `phase3_step_dir_test`  
**Feature flags:** `CONTROL_LOOP=OFF`, `FIXED_STEP_FREQ=1000`

### Test
1. Flash preset  
2. Connect scope probe to STEP pin (J1 → Pin 2), ground clip to Teensy GND
3. Scope settings: 500 µs/div, 2 V/div, DC coupling

### What to Verify
- [ ] ✅ Clean square wave at exactly 1000 Hz
- [ ] ✅ 50% duty cycle (500 µs HIGH, 500 µs LOW)
- [ ] ✅ Amplitude 3.0–3.3 V (Teensy 3.3 V IO)
- [ ] ✅ Rise time < 10 ns (no slow edges = no cable capacitance problem)
- [ ] ✅ No ringing (>10% overshoot = add 100 Ω series resistor)
- [ ] ✅ DIR pin stable (no glitch within 2 µs of STEP edge)

Repeat for all 6 STEP pins.

### Pass Criteria
All 6 STEP pins produce clean 1 kHz square waves. DIR pins are stable.

---

## Phase 4 — Open-Loop Motor Rotation (No Feedback)

**Config preset:** `phase4_open_loop`  
**Feature flags:** `CONTROL_LOOP=ON`, `ENCODER_FEEDBACK=OFF`, `FILTER=OFF`, `WATCHDOG=OFF`

> [!CAUTION]
> First time motor actually rotates. Have a physical E-stop ready (pull PSU).
> Current limit PSU to 1.5 A max. Motor should draw <1 A at low speed.

### Test
1. Use Manual Jog tab: set J1 velocity to `0.5 rad/s`  
2. Motor should rotate smoothly for 3 seconds, then stop (jog timeout)

### Pass Criteria
- Motor rotates in correct direction
- Rotation is smooth (no stutter or chirping)
- Motor stops cleanly when jog command ends
- PSU current < 1 A

### Fail Indicators
| Symptom | Possible Cause |
|:--|:--|
| Motor stutters | Microstep mismatch — check driver DIP switches vs `MICROSTEPS=32` |
| Motor too hot immediately | Current too high — reduce `max_current_ma` in Joints tab |
| Wrong direction | Set `DIR_INVERT` in Joints tab, save as new preset |
| No movement | STEP signal not reaching driver — check wiring from Phase 3 |

---

## Phase 5 — Closed-Loop, Single Axis, No Filter (J1 Only)

**Config preset:** `phase5_closed_loop_j1`  
**Feature flags:** `ENCODER_FEEDBACK=ON`, `FILTER=OFF`, `WATCHDOG=OFF`, joints J2–J6 DISABLED

### Test
1. In Jog tab: send position command `+0.3 rad` to J1
2. Watch oscilloscope (Plot tab): `actual_pos` should converge to `desired_pos`
3. Note: will likely oscillate — expected without filter

### Measure
- **Overshoot**: how far past the target it goes (expect 0.05–0.2 rad)
- **Settling time**: how long before it stabilizes (expect 0.5–3 s)
- **ISR time**: should be < 15 µs

---

## Phase 6 — Add AlphaBeta Filter

**Toggle ON:** `FEATURE_ALPHABETA_FILTER`  
Compare Plot tab vs Phase 5 result.

### Pass Criteria
- Less overshoot than Phase 5
- Velocity estimate is smooth (no spikes in velocity channel)
- Settling time same or better

---

## Phase 7 — Add Velocity Feedforward

**Toggle ON:** `FEATURE_VELOCITY_FEEDFORWARD`  
Send a slow trajectory via Jog. Plot `desired_pos` vs `actual_pos`.

### Pass Criteria
- Tracking error (gap between desired and actual lines) < 15% of Phase 6

---

## Phase 8 — Watchdog Safety

**Toggle ON:** `FEATURE_WATCHDOG`

### Test
1. Start motor at `1.0 rad/s` via Jog
2. Pull USB cable while motor is running
3. Motor **must stop within 200 ms**
4. Plug cable back in — GUI should show supervisor state = `SOFT_ESTOP`
5. Motor must not restart without explicit reset

### Pass Criteria
Motor stops on cable pull. Does not restart on reconnect without command. E-stop light in GUI status bar turns red.

---

## Phase 9 — Multi-Axis

**Config preset:** `phase9_multiaxis`  
Enable J2, test alone. Then J1+J2 together. Repeat one joint at a time up to all 6.

---

## Phase 10 — Full Feature Stack + Communication Scaling

**Config preset:** `phase10_full_stack`  
All features ON, all joints enabled.

1. Start with `UART_115200`, ROS at 25 Hz — establish baseline
2. Switch to `USB_CDC_HS`, raise ROS to 100 Hz — compare tracking error (should improve)
3. Monitor SEQ gap counter in status bar — should read `0 dropped` at 100 Hz

### Final Acceptance Criteria

| Metric | Teensy 4.1 | BlackPill STM32 |
|:--|:--|:--|
| ISR time (max) | < 25 µs | N/A (no profiler) |
| Control jitter (std) | < 50 µs | — |
| ACK rate | ~1000 pkts/s | ~50 pkts/s |
| Tracking error at 25 Hz | < 20 mrad | < 30 mrad |
| Tracking error at 100 Hz | < 8 mrad | — |
| Dropped packets | 0 | 0 |
| Supervisor faults (idle) | 0 | N/A |
| DFU reboot round-trip | N/A | < 3 s |

---

## 🔵 STM32 BlackPill — Pre-Phase Checklist

Before running the standard phases on a BlackPill target, verify:

- [ ] `lsusb` shows `0483:df11` (DFU) or `0483:5740` (CDC) — board is enumerating
- [ ] **PA10 not connected** — BlackPill DFU errata (PA10 must float)
- [ ] Flash tab environment = `blackpill_f411ce`
- [ ] Initial flash via BOOT0+NRST successful (DFU download 100%)
- [ ] `INIT_OK` received in Serial tab after board resets
- [ ] ACK packets visible in Plot tab (interleaved format auto-detected, ~50 pkts/s)

### 🔵 Phase 0.5 — DFU Reboot Round-Trip (STM32 only)

Verify software DFU entry works before running any hardware tests:

1. Board connected via USB-CDC — Serial tab shows `INIT_OK`
2. Click **`<REBOOT_DFU>`** macro in Serial tab
3. `lsusb` changes to `0483:df11` within 1–2 s ✅
4. Flash tab status panel shows `⚡ DFU mode detected` ✅
5. Click **⏏ Detach DFU** — board reboots back into firmware ✅
6. `/dev/ttyACM0` reappears and `INIT_OK` is printed again ✅

**Pass criteria:** Full round-trip completes in < 3 seconds without pressing any physical button.  
Once this passes, every subsequent `🚀 Build & Upload` click is fully automatic.

### 🔵 Phase 1.5 — STM32 Encoder Capture (PWM Input Mode)

The `realtime_servo_blackpill` firmware uses hardware PWM Input mode — no interrupts, just register reads.

1. Connect board and open **Serial** tab
2. Manually rotate a motor shaft by hand (J5 or J6 — they are enabled in default config)
3. Watch **Plot** tab — `actual_pos` should change smoothly
4. ISR bar in Jog tab will read 0 (no profiler in blackpill firmware — this is expected)

**Pass criteria:**
- Position changes proportionally to shaft rotation
- No spikes or resets mid-rotation
- Stationary shaft → stable position ± 0.01 rad
