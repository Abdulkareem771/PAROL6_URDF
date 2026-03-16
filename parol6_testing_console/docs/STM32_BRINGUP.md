# STM32 BlackPill F411CE — Bring-Up Guide

## Board: WeAct Studio BlackPill V2.0 (STM32F411CEU6)

| Spec | Value |
|---|---|
| CPU | ARM Cortex-M4 @ 100 MHz |
| Flash | 512 KB |
| RAM | 128 KB |
| USB | Full Speed (12 Mbps) USB-CDC |
| Programmer (DFU) | `0483:df11` via BOOT0 pin |

---

## ⚡ Quick Reference: Master Pin Map

This consolidated table shows all critical physical connections for the robot. 
*Note: The teammate's firmware uses hardware PWM Input mode for encoders (zero CPU overhead).*

| Joint | STEP Pin | DIR Pin | PWM Encoder In | Limit Switch | Timer (Encoder) |
|-------|----------|---------|----------------|--------------|-----------------|
| **J1**| PA7      | PB10    | PA8            | PA1          | TIM1 (AF1)      |
| **J2**| PA9      | PB12    | PA15           | PA3          | TIM2 (AF1)      |
| **J3**| PB0      | PB13    | PA6            | PB4          | TIM3 (AF2)      |
| **J4**| PB1      | PB14    | PB6            | —            | TIM4 (AF2)      |
| **J5**| PB8      | PB15    | PA0            | —            | TIM5 (AF2)      |
| **J6**| PB9      | PA5     | PA2            | —            | TIM9 (AF3)      |

### Critical Hardware Notes
- **PA10 must NOT be connected** — BlackPill hardware errata. The pull-up on PA10 blocks DFU enumeration if connected.
- **PC13** = onboard LED (active LOW). Used as boot indicator in firmware.
- **BOOT0 pin** — needs to be held HIGH at power-on/reset to enter DFU bootloader.

---

## 🧪 Testing Verification Protocol

When validating a newly flashed board or checking hardware connections, follow these steps using the Console tabs:

1. **Verify Serial Comms (Serial Tab)**
   - Click `⚡ Connect`. You should immediately see `<ACK,...>` packets streaming.
   - If you see `ү>`, the flash failed or the float formatter is missing (recompile teammate firmware).
2. **Verify Encoders (Oscilloscope Tab)**
   - Manually move the physical robot joints by hand.
   - Watch the `Joint Positions` plot. The lines should move smoothly. If a line flatlines, check the **PWM Encoder In** pin connection for that joint.
3. **Verify Motors (Jog Tab)**
   - Click `ON` to enable the motors.
   - Set Jog Speed to 10 deg/s.
   - Click `+ ▶` for Joint 1. The motor should step smoothly. If it vibrates but doesn't rotate, check **DIR Pin**. If it does nothing, check **STEP Pin** or power.
4. **Serial Macros Available:**
   - `<ENABLE>`: Energizes the stepper coils.
   - `<DISABLE>`: Cuts power to steppers (free-drive).
   - `<HOME>`: Triggers the homing sequence (moves until limits hit).
   - `<RESET>`: Soft-resets the robot state.

---

## DFU Flash Workflow

### Manual Entry (first time)
1. Hold **BOOT0** button
2. Press and release **NRST**
3. Release **BOOT0**
4. `lsusb` → `0483:df11 STMicroelectronics DFU` ✅

### Using the Console (subsequent flashes)
1. Board must be running parol6_firmware with `<REBOOT_DFU>` support
2. In **Serial tab**: click **`<REBOOT_DFU>`** macro (or just click **🚀 Build & Upload** — it sends the command automatically)
3. Flash tab detects DFU mode, runs `pio upload -e blackpill_f411ce`, then auto-detaches

### Manual flash via CLI
```bash
docker exec -w /workspace/parol6_firmware parol6_dev \
  pio run -e blackpill_f411ce --target upload
```

### After flashing — board doesn't appear on `/dev/ttyACM*`?
The DFU bootloader waits for a detach before resetting. Options:
- Press **NRST** on board
- Click **⏏ Detach DFU** in the Flash tab
- The console sends `dfu-util -e` automatically after each successful upload

---

## Firmware Variants

### `parol6_firmware` (PlatformIO, `blackpill_f411ce` env)
- Full PAROL6 firmware cross-compiled for STM32
- Uses `HardwareTimer` (TIM3) for 1 kHz control loop
- Full ACK format: `<ACK,seq, p0..p5, v0..v5, lim_state, state_byte>`
- Supports `<REBOOT_DFU>` software DFU entry
- ISR profiling disabled on STM32 (Teensy-specific ARM_DWT)

### `realtime_servo_blackpill` (Arduino IDE / STM32Duino)
- Standalone sketch by teammate
- Hardware PWM Input mode encoders (TIM1-5, TIM9) — zero ISR load
- 500 Hz control loop (TIM11)
- 50 Hz feedback rate
- Interleaved ACK format: `<ACK,seq, p0,v0, p1,v1, ...>`
- No supervisor state machine, no DFU reboot command

---

## Clock Configuration

```
STM32F411CE @ 96 MHz (realtime_servo) / 100 MHz (parol6_firmware)
  HSE = 25 MHz crystal
  PLL: PLLM=25, PLLN=192, PLLP=2, PLLQ=4
  USB clock = 48 MHz (exact)
```

Encoder timer capture clock: `96 MHz / 8 = 12 MHz → 83.33 ns/tick`  
Digital filter: `ICxF=9 → 667 ns glitch rejection` (same as Teensy QTimer FILT)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `0483:5740` shown in `lsusb` | Board in normal CDC mode | Press BOOT0+NRST to enter DFU |
| `dfu-util -l` shows nothing | Board off USB | Press NRST, or re-plug |
| Board doesn't boot after flash | Auto-detach failed | Press NRST manually |
| `No such file /dev/ttyACM0` after flash | Board still in DFU mode | Click **⏏ Detach DFU** in Flash tab |
| ACK packets not parsed in Plot/Jog tab | Wrong firmware ACK format | Check which firmware variant is running — console auto-detects n=12 (interleaved) vs n≥13 (flat) |
| Encoder reads stuck at 0 | PA10 connected | Remove PA10 connection |
