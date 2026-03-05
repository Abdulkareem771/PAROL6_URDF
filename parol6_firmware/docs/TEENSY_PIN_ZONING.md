# PAROL6 Teensy 4.1 Hardware Pin Zoning Architecture

## Overview
The PAROL6 real-time controller utilizes the NXP i.MXRT1062 processor (Teensy 4.1). Because advanced hardware peripherals (like QuadTimers and FlexPWM) share the XBAR routing matrix and internal multiplexers, pins **must** be pre-allocated into strict functional "Zones" to prevent irresolvable hardware collisions as the project scales.

This document serves as the frozen reference for PCB design and firmware expansion. DO NOT violate these zones.

---

## 🛑 The Constraints & Rules
1. **QuadTimer Locks Muxes**: QuadTimer pins must be treated as permanently allocated after init. Dynamic remuxing is prohibited in real-time firmware.
2. **FlexPWM vs. QuadTimer XBAR Conflicts**: Conflicts are pad-group specific, not global. Zoning prevents accidental pad-level XBAR contention.
3. **Global Determinism Rule**: No Zone may introduce ISR load that interferes with the 1 kHz control loop. All high-frequency peripherals must be hardware-driven (PWM, DMA, timers).

---

## 🧭 The PAROL6 Pin Zones

### Zone 1 — Encoder Domain (LOCKED)
**Function**: 6x High-Speed Magnetic PWM Encoder Capture
**Peripheral**: QuadTimers (Gated Count Mode)
**Status**: 🔒 Frozen

These pins are definitively assigned to QuadTimers and cannot be reused for anything else.
*   **Joint 1**: Pin 10
*   **Joint 2**: Pin 11
*   **Joint 3**: Pin 12
*   **Joint 4**: Pin 14
*   **Joint 5**: Pin 15
*   **Joint 6**: Pin 18

### Zone 2 — Step Generation (FlexPWM Zone)
**Function**: 6x Motor Step Pulses
**Peripheral**: FlexPWM
**Requirement**: Step generation must use FlexPWM only. QuadTimer PWM mode (`analogWrite` on pins 10-15) is strictly prohibited as it will destroy encoder capture functionality.

**Recommended Pins**:
*   `2, 3, 4, 5, 6, 7, 8, 9`
*   `22, 23`
*   `28, 29`

**Confirmed STEP pin assignments (from `ActuatorModel.h`):**
| Joint | STEP Pin | FlexPWM Submodule |
| :--- | :--- | :--- |
| J1 | **Pin 2** | FlexPWM4.2A |
| J2 | **Pin 6** | FlexPWM2.2A |
| J3 | **Pin 7** | FlexPWM1.3B |
| J4 | **Pin 8** | FlexPWM1.3A |
| J5 | **Pin 4** | FlexPWM2.0A |
| J6 | **Pin 5** | FlexPWM2.1A |

*Implementation Note*: When writing the Step generation HAL, pick 6 sequential pins from this list to keep XBAR routing clean.

### Zone 3 — Direction Pins (Pure GPIO)
**Function**: 6x Motor Direction Signals
**Peripheral**: Standard Fast GPIO
**Requirement**: Direction signals update at most 1 kHz and don't require hardware timers.

**Recommended Pins**: 
*   `30, 31, 32, 33, 34, 35, 36, 37, 38, 39`

> [!WARNING]
> Do NOT use pins 24-29 for Direction signals. Pins 28, 29 are also listed as Zone 2 (FlexPWM-capable) candidates and are needed as STEP pin overflow if the primary Zone 2 pins get reassigned. Using them for DIR creates an irresolvable conflict.

**Confirmed DIR pin assignments (from `ActuatorModel.h`):**
| Joint | DIR Pin |
| :--- | :--- |
| J1 | **Pin 30** |
| J2 | **Pin 31** |
| J3 | **Pin 32** |
| J4 | **Pin 33** |
| J5 | **Pin 34** |
| J6 | **Pin 35** |

*Implementation Note*: Keeping them grouped mathematically simplifies bit-banging if optimization is needed later.

### Zone 4 — Safety Domain (Interrupt / Gating Bank)
**Function**: E-Stop, Limit Switches, Inductive Prox Sensors
**Peripheral**: EXTI (External Interrupts) + Hardware Gating
**Requirement**: All fast-acting safety sensors should live on the same GPIO port bank so they can be read atomically in a single 32-bit register read during the Supervisor ISR.

**Recommended Pins**:
*   `20, 21, 24, 25, 26, 27`

**Legacy Hardware Mapping Discovery:**
Analysis of the `PAROL6 control board main software` (the open-source baseline) reveals the following physical sensor realities for the 6 joints:
*   **Sensor Type**: They are initialized as pure `INPUT` (not `INPUT_PULLUP`), indicating active 12V/24V Inductive Proximity Sensors, not passive mechanical microswitches.
*   **Trigger Polarity**: The hardware relies on a mix of edge triggers.
    *   **J1**: LIMIT6 (Triggers FALLING / LOW)
    *   **J2**: LIMIT2 (Triggers RISING / HIGH)
    *   **J3**: LIMIT3 (Triggers RISING / HIGH)
    *   **J4**: LIMIT4 (Triggers RISING / HIGH)
    *   **J5**: LIMIT5 (Triggers RISING / HIGH)
    *   **J6**: LIMIT1 (Triggers FALLING / LOW)

**Safety Architecture Warning**:
*   **Primary vs Secondary**: The Emergency Stop MUST NOT rely solely on the Teensy. The E-Stop must physically cut power/enable logic to the stepper drivers (Primary Safety). The connection to the Teensy in Zone 4 is to trigger the software Supervisor to halt the trajectory (Secondary Safety).
*   **EMI Coupling Rule**: **Never mix safety inputs with high-speed PWM pads**. Fast-switching STEP/PWM signals placed physically adjacent to safety/encoder inputs on the PCB routing or wire harness will inject severe electro-magnetic interference (EMI), causing false limit-switch trips or encoder corruption.

### Zone 5 — Communications Domain (Reserved)
**Function**: ROS Transport, Debugging, Expansion Buses
**Peripheral**: USB HS, UART, CAN, SPI
**Requirement**: Do not assign safety or real-time peripherals to these trace-sensitive communications pins.

**Reserved Pins**:
*   **USB HS**: Native D+/D- interior pads (Fixed)
*   **UART1 (Serial1)**: `0, 1`
*   **UART2 (Serial2)**: `7, 8`
*   **CAN FD (Optional Future)**: `3, 4`

---

## Future Expansion
Inductive sensors operate at 12V-24V. As observed in the legacy firmware, they are active sensors. **Opto-isolation is strongly recommended over passive level shifting (voltage dividers) due to EMI and ground offset realities in industrial motor environments.** Direct 24V exposure will instantly destroy the 3.3V i.MXRT processor.
