# PAROL6 Legacy Hardware Mapping

This document captures the physical hardware reality of the PAROL6 robot based on an analysis of the original `PAROL6 control board main software` (the open-source, open-loop firmware baseline). 

As we migrate to a closed-loop Teensy 4.1 architecture (Phase 4), this map serves as the definitive reference for how the physical endstops and limit sensors are electrically configured.

---

## 🛑 Limit Switch & Inductive Sensor Mapping

The original firmware defines six distinct endstop inputs (`LIMIT1` through `LIMIT6`). Analysis of how these pins are initialized and evaluated reveals that the robot uses **Active Inductive Proximity Sensors** for its joints, not passive mechanical micro-switches.

### How do we know this?
1. **No Internal Pull-ups**: In `hw_init.cpp`, all six limit pins are initialized strictly as `pinMode(LIMITX, INPUT);`. If these were simple mechanical switches, reliable embedded design would mandate `INPUT_PULLUP` to prevent floating states. Active inductive sensors (which operate at 12V/24V and output a strong, driven logic signal—often via external optocouplers) require pure `INPUT`.
2. **Mixed Trigger Edges**: In `motor_init.cpp`, the `limit_switch_trigger` struct variables are explicitly assigned `1` (HIGH) or `0` (LOW) depending on the joint. Furthermore, the interrupt routines in `hw_init.cpp` are mapped to a mix of `RISING` and `FALLING` edges. This mixed-polarity design is highly characteristic of industrial inductive sensors (NPN normally-open vs. PNP normally-closed) being deployed across different axes.

### The Explicit Joint Mapping

| PAROL6 Joint | Assigned Limit Pin | Electrical Trigger Polarity | Expected Sensor Type |
| :--- | :--- | :--- | :--- |
| **Joint 1 (Base)** | `LIMIT6` | Active **LOW** / `FALLING` edge | Inductive Sensor (Likely NPN NO) |
| **Joint 2 (Shoulder)** | `LIMIT2` | Active **HIGH** / `RISING` edge | Inductive Sensor (Likely PNP NO) |
| **Joint 3 (Elbow)** | `LIMIT3` | Active **HIGH** / `RISING` edge | Inductive Sensor (Likely PNP NO) |
| **Joint 4 (Wrist 1)** | `LIMIT4` | Active **HIGH** / `RISING` edge | Inductive Sensor (Likely PNP NO) |
| **Joint 5 (Wrist 2)** | `LIMIT5` | Active **HIGH** / `RISING` edge | Inductive Sensor (Likely PNP NO) |
| **Joint 6 (Wrist 3)** | `LIMIT1` | Active **LOW** / `FALLING` edge | Inductive Sensor (Likely NPN NO) |

*⚠️ Important Academic Note on Polarity: Sensor polarity must be verified on hardware before final wiring, as inductive sensor types (NPN vs PNP) and Normally Open / Normally Closed states vary wildly across production batches of the same mechanical robot.*

---

## ⚙️ Safety & Architecture Implications for Teensy 4.1

1. **Voltage Warning**: The active inductive sensors on the PAROL6 operate at industrial voltages (typically 12V or 24V). The new Teensy 4.1 MCU is strictly **3.3V tolerant**. 
   * **Mandatory Requirement**: You must use opto-isolators or voltage dividers (level shifters) on the custom PCB between the inductive sensors and the Teensy GPIO pins (Zone 4) to avoid instantly destroying the i.MXRT processor.
2. **EMI Coupling Danger**: Because these sensors carry active voltage logic, their signal wires must be physically isolated from the high-speed FlexPWM `STEP` signals on the PCB and wire harness. Failing to do so will result in induced EMI causing false limit-trips and random robot halts.

---

## 💾 Legacy Software Logic & Control Flow

Understanding the baseline open-source software helps us quantify the improvements of the Phase 4 Real-Time architecture. The original `PAROL6 control board main software` operated under an entirely different paradigm.

### 1. The Core Execution Loop (Polling Architecture)
The legacy firmware did **not** use a deterministic hardware timer for step generation or control math. Instead, it relied on a standard Arduino `loop()` function.
*   The `loop()` continuously polled `Serial.available()` for incoming binary packets.
*   Once a packet was decoded, positional commands were passed to the `AccelStepper` library.
*   The loop then called `stepper[i].run()` or `stepper[i].runSpeed()` sequentially for all 6 joints.

**Why we migrated away from this:**
Software polling (`AccelStepper.run()`) means step pulse timing is entirely dependent on how fast the CPU can iterate the `loop()`. If serial parsing or safety checks briefly stall the CPU, step pulses are delayed, causing micro-vibrations and non-deterministic torch paths. Our Phase 4 architecture moves step generation to **Hardware FlexPWM**, completely decoupling it from CPU load.

### 2. Motor Drivers (SPI TMC5160)
The legacy board used Trinamic **TMC5160** stepper drivers.
*   They were configured via SPI (`stepper_driver.begin()`) during board initialization.
*   The firmware explicitly set parameters like `rms_current`, `microsteps`, and `hold_multiplier`.
*   Diagnostic data (over-temp warnings, short to ground) was read back via SPI.

**Teensy 4.1 Implication:** If the new architecture uses industrial MKS Servo42C drives (which accept standard STEP/DIR), the SPI initialization overhead is removed entirely, simplifying the firmware.

### 3. The Legacy Serial Protocol (Binary Payload)
Unlike our current ASCII `<SEQ,pos...>` format, the legacy firmware used a raw, un-framed binary stream.
*   **Packet Header**: Preceded by three `0xFF` bytes to signal a new message.
*   **Positions/Speeds**: Packed as raw 3-byte integers (`intTo3Bytes`) to save bandwidth.
*   **Termination**: A single CRC byte followed by `0x01` and `0x02`.
*   **Vulnerability**: Raw byte streaming is fragile. If a single byte is dropped on the USB line, the state machine (`data_counter == data_len - 1`) misaligns completely until the next `0xFF 0xFF 0xFF` header is coincidentally found.

**Phase 4 Upgrade:** This validates our decision to freeze the `<SEQ...>` ASCII protocol. While slightly less bandwidth-efficient, CSV ASCII strings terminated by `\n` are intrinsically self-aligning and human-readable for debugging.
