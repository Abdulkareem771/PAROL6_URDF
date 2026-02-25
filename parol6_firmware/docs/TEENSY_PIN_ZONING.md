# PAROL6 Teensy 4.1 Hardware Pin Zoning Architecture

## Overview
The PAROL6 real-time controller utilizes the NXP i.MXRT1062 processor (Teensy 4.1). Because advanced hardware peripherals (like QuadTimers and FlexPWM) share the XBAR routing matrix and internal multiplexers, pins **must** be pre-allocated into strict functional "Zones" to prevent irresolvable hardware collisions as the project scales.

This document serves as the frozen reference for PCB design and firmware expansion. DO NOT violate these zones.

---

## 🛑 The Constraints
1. **QuadTimer Locks Muxes**: When a pin is initialized as a QuadTimer input, the GPIO function is permanently disabled for that pad. You cannot use `digitalRead` or `attachInterrupt` on it.
2. **FlexPWM vs. QuadTimer XBAR Conflicts**: Although independent peripherals, they share root clocks and XBAR (crossbar) pathways. Using a FlexPWM output and a QuadTimer input on the exact same pad group can saturate the routing matrix.
3. **Interrupt Contention**: High-frequency step generation must not block the 1 kHz deterministic control loop. Step generation must use hardware PWM (FlexPWM) or DMA, not busy loops or CPU ISRs.

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
**Requirement**: Must be routed to pins that native support FlexPWM without colliding with Zone 1 QuadTimers.

**Recommended Pins**:
*   `2, 3, 4, 5, 6, 7, 8, 9`
*   `22, 23`
*   `28, 29`

*Implementation Note*: When writing the Step generation HAL, pick 6 sequential pins from this list to keep XBAR routing clean.

### Zone 3 — Direction Pins (Pure GPIO)
**Function**: 6x Motor Direction Signals
**Peripheral**: Standard Fast GPIO
**Requirement**: Direction signals update at most 1 kHz and don't require hardware timers.

**Recommended Pins**: 
*   `30, 31, 32, 33, 34, 35, 36, 37, 38, 39`

*Implementation Note*: Keeping them grouped mathematically simplifies bit-banging if optimization is needed later.

### Zone 4 — Safety Domain (Interrupt / Gating Bank)
**Function**: E-Stop, Limit Switches, Inductive Prox Sensors
**Peripheral**: EXTI (External Interrupts) + Hardware Gating
**Requirement**: All fast-acting safety sensors should live on the same GPIO port bank so they can be read atomically in a single 32-bit register read during the Supervisor ISR.

**Recommended Pins**:
*   `20, 21, 24, 25, 26, 27`

**Safety Architecture Warning**:
The Emergency Stop MUST NOT rely solely on the Teensy. The E-Stop must physically cut power/enable logic to the stepper drivers (Primary Safety). The connection to the Teensy in Zone 4 is to trigger the software Supervisor to halt the trajectory (Secondary Safety).

---

## Future Expansion
Inductive sensors operate at 12V-24V. They will require optocouplers or logic-level shifters on the custom PCB before interfacing with Zone 4 pins to prevent destroying the 3.3V i.MXRT processor.
