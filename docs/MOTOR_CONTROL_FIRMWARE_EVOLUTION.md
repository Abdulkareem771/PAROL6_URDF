# Motor Control Firmware Evolution & ROS 2 Communication

This document details the comprehensive evolution of the embedded systems, motor control firmwares, and ROS 2 communication architecture throughout the development phase of the PAROL6 robotic arm. It covers the chronological transition between different hardware microcontrollers, evaluates their advantages and limitations, and explains the critical shift in actuation strategy for the MKS SERVO42C closed-loop stepper drivers.

---

## 1. Introduction: The Need for Real-Time Determinism

The integration of the PAROL6 robotic arm with the ROS 2 `ros2_control` ecosystem required bridging the gap between high-level, asynchronous trajectory generation (MoveIt 2) and the stringent, hard real-time requirements of six independent stepper motors.

Industrial applications, such as welding, require extreme kinematic fidelity—meaning the end-effector must exactly follow a smooth, pre-calculated Cartesian path without unexpected stops, jitter, or vibration. Achieving this required evolving the embedded system from a basic open-loop listener to a deterministic, high-bandwidth servo controller capable of 1 kHz closed-loop execution.

---

## 2. Microcontroller Evolution & Comparison

Throughout the development cycle, three microcontrollers were evaluated and implemented as the bridge between the ROS 2 Host PC and the motor drivers. Based on the project's timeline, the development evolved chronologically: **ESP32 → Teensy 4.1 → STM32 BlackPill**.

### 2.1 Prototyping Phase: ESP32

The ESP32 was investigated initially to implement a Position Servo + Velocity Feedforward control layer, aiming to decouple ROS 2 commands from raw step generation.

*   **Role in Development:** Prototyping a 500 Hz control loop running as a FreeRTOS task.
*   **Advantages:**
    *   Dual-core architecture and built-in WiFi/Bluetooth offered wireless debugging possibilities.
    *   Inexpensive, highly documented, and ubiquitous.
*   **Disadvantages (Why it was deemed unsuitable):**
    *   **Execution Jitter:** The ESP32 suffered from significant context-switching overhead. The FreeRTOS control task competed continuously with the internal background processes (especially the WiFi/Bluetooth radio stacks and network interrupts).
    *   **Timing Instability:** As a consequence of the RTOS preemption, the 500 Hz control loop exhibited non-deterministic timing jitter. In a robotics context, varying loop periods distort velocity outputs, injecting mechanical vibration and causing MoveIt execution timeouts in precision domains.

### 2.2 High-Performance Phase: Teensy 4.1

To resolve the ESP32's jitter and FreeRTOS overhead, the architecture was migrated to the **Teensy 4.1** (NXP i.MX RT1062 ARM Cortex-M7 running at 600 MHz).

*   **Role in Development:** Designing and proving a strict 1 kHz, hard real-time servo controller driven by bare-metal interrupts.
*   **Advantages:**
    *   **Unmatched Bare-Metal Determinism:** By avoiding an RTOS entirely for the control loop in favor of a raw IntervalTimer interrupt, execution jitter was compressed aggressively (measured at `< 1 µs` deviation).
    *   **Hardware Offloading:** `FlexPWM` generated robust STEP pulses concurrently for all 6 axes without CPU bit-banging (`digitalWrite`). `QuadTimers` read encoder edges with zero CPU overhead.
*   **Disadvantages / Ultimate Fate:**
    *   3.3V logic interfacing with 5V industrial sensors required careful hardware zoning and level shifting.
    *   **Catastrophic Hardware Failure:** During hardware-in-the-loop validation, the Teensy 4.1 burned out. Its lower electrical tolerance for voltage spikes from inductive traces ultimately forced a migration to a more fault-tolerant controller.

### 2.3 Final Fault-Tolerant Phase: STM32 BlackPill

Following the burnout of the Teensy 4.1, the mature control architecture was rapidly ported to the **STM32 BlackPill**.

*   **Role in Development:** The current and final microcontroller architecture. It inherited the structural lessons (Alpha-Beta filtering, deterministic interpolation) from the Teensy phase while providing a more robust electrical baseline.
*   **Advantages:**
    *   **Industrial Resilience:** Proven reliability and electrical tolerances. Many STM32 pins are 5V tolerant, significantly reducing the burn-out risks from inductive sensor spikes or wiring faults that destroyed the Teensy.
    *   **Shared Heritage:** The original open-loop control board that shipped with the PAROL6 robot was also STM32-based. This allowed for the re-use of definitive hardware constants (e.g., precise mechanical gear ratios and limit switch polarity) with absolute confidence.
    *   Successfully executed the deterministic Hardware Timer ISR architecture developed on the Teensy without the associated fragility.
*   **Disadvantages:**
    *   Slightly lower clock speed than the Cortex-M7 Teensy 4.1, though still highly capable of executing the 1 kHz control loop and math logic without starvation.

---

## 3. The Actuation Shift: UART vs. Step/Dir + PWM Feedback

As the project transitioned to using the **MKS SERVO42C** closed-loop stepper drivers, the method of controlling them became a critical architectural decision.

### The Initial Attempt: UART Control
The MKS SERVO42C drivers possess a UART interface. Initially, it was hypothesized that sending direct target angles or velocities via UART packets to each driver would be the smartest, "most digital" approach.

**Why UART Failed:**
*   **Bus Saturation and Latency:** Communicating via UART to 6 different drivers within a 1 kHz control loop creates a massive transmission bottleneck. The baud rate becomes a physical limit, preventing synchronous, simultaneous triggering of all 6 axes.
*   **Non-Determinism:** Serial buffers and parsing on both the MCU and the Motor Driver introduce unpredictable microseconds of delay.
*   **Loss of Centralized Control:** Tossing trajectories over UART turns the motor drivers into black boxes, stripping the central MCU of the ability to instantly halt motion via E-Stops or to implement coordinated kinematic runaway protection.

### The Proven Solution: Robust Step/Dir + PWM Generation
The architecture shifted to treat the smart MKS SERVO42C drivers exactly like standard "dumb" stepper drivers using standard **Step/Direction** pins, while letting the MKS driver's internal Field Oriented Control (FOC) handle the actual closed-loop torque application.

**How It Works:**
1.  **Hardware Timers as Metronomes:** The microcontroller utilizes its hardware timers (like `FlexPWM` on Teensy or advanced timers on STM32) to generate the STEP pulses. 
    *   **Frequency = Velocity**
    *   **Edge Count = Position**
    *   **Duty Cycle = Irrelevant** (Fixed to a safe 2-5µs pulse width).
2.  **Absolute Decoupling:** The 1 kHz ISR calculates standard control effort (e.g., `cmd_vel_ff + (Kp * pos_error)`), converts that demanded velocity in radians/sec into a raw Hertz frequency based on the gear ratios, and updates the timer register. 
3.  **The Result:** The MCU spends zero cycles manually toggling pins. The step generation perfectly reflects the interpolated velocity intent without UART latency. The MKS driver’s internal FOC algorithm guarantees that every pulse received results in physical motion without stalling, rendering the system extremely robust computationally.

---

## 4. Communication with ROS 2

The pipeline linking the high-level ROS 2 intelligence (running on a host PC/Docker container) to the STM32 BlackPill was meticulously layered to prevent timing collisions. 

### The Protocol
Communication flows over a USB/Serial link at 115200 Baud. It utilizes an ASCII framed protocol.
*   **Command (PC to MCU, ~25 Hz):** `<SEQ, J1_pos, J1_vel, J2_pos, ..., J6_vel>\n`
*   **Feedback (MCU to PC, ~10+ Hz):** `<ACK, SEQ, J1_pos, ..., J6_vel>\n`

### The Asynchronous Decoupling Strategy
If the MCU's 1 kHz control loop had to wait for a 115200 baud serial string to parse, the system would immediately crash. To solve this, Data Ownership and Thread Safety rules were implemented:

1.  **Background Parsing (Transport Layer):** The primary `main()` loop continuously polls the serial port. It parses incoming ASCII bytes and constructs fully validated `RosCommand` structures.
2.  **Lock-Free Circular Queue:** The parsed commands are pushed into a lock-free queue. This acts as the sole boundary between the asynchronous ROS timing and the synchronous motor timing.
3.  **The 1 kHz Control ISR:** 
    *   Fires precisely every 1000 µs.
    *   Pops the latest command from the queue safely.
    *   Hands the distant (e.g., 40ms away) ROS waypoint to a **Linear Interpolator**.
    *   The Interpolator up-samples the trajectory, providing smooth 1ms delta targets (`cmd_pos` and `cmd_vel_ff`).
4.  **Signal Filtering:** To ensure sensor noise does not cause voltage spikes in the motor outputs, an **Alpha-Beta Observer** is used inside the ISR. It estimates true position and velocity from the raw encoders while handling multi-turn wrapping cleanly and deterministically.

### Conclusion of the ROS 2 Integration
By implementing a strict "Interpolate on MCU" architecture (Centralized Planning via MoveIt, Distributed Interpolation/Control via STM32 BlackPill), the system perfectly mimics industrial robotics paradigms. Safety is enforced by a Supervisor that clamps velocities to zero if a ROS packet is inexplicably delayed, ensuring fault-tolerant operation in a real-world environment.
