# PAROL6 Firmware Architecture Validation

The overall structure of the firmware is **highly robust, exceptionally well-segregated, and employs correct real-time (RTOS/bare-metal) design patterns.** 

Here is an architectural breakdown of why it is valid, and a few minor weaknesses to watch out for.

---

## 1. Timing Domains & Segregation (Exceptional)
The firmware perfectly isolates completely non-deterministic operations from strictly deterministic real-time operations.

### The 1 kHz Hardware Timer ISR ([run_control_loop_isr](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/main.cpp#110-220))
- **Strictly Deterministic:** This loop does **zero** waiting. It contains no `delay()`, no serial formatting, no unbounded loops. It runs mathematical calculations (P+FF control law, AlphaBeta filter, Interpolator tick) and writes directly to hardware registers.
- **Profiling Built-In:** The CPU cycle counter (`ARM_DWT_CYCCNT`) measures exactly how many microseconds the ISR takes, providing a verifiable proof of real-time execution limits.
- **Zero-Interrupt Sensor Capture:** The [QuadTimerEncoder](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/hal/QuadTimerEncoder.h#43-44) uses hardware gated-counters. Instead of the CPU being interrupted 10,000 times a second to measure PWM pulses, the hardware does it silently. The ISR simply reads a register (`tmr_->CH[ch_].CNTR`) that takes 1 CPU cycle. This is an advanced and highly robust design.

### The Background Loop ([loop()](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/main.cpp#269-330))
- Handles slow, unpredictable tasks: reading USB/UART bytes, decoding ROS packets, and `sprintf` generating telemetry output.
- **Lock-free/Minimal-lock Data Transfer:** It uses a `CircularBuffer` to pass parsed commands from the background loop to the ISR. 
- When updating telemetry, it briefly pauses interrupts (`noInterrupts()`) just long enough to copy the 6 floats, preventing "data tearing" (where half the floats are from tick A and half from tick B).

---

## 2. Signal & Control Flow (Valid)

The path from sensor to motor is structurally sound:

1. **Hardware Capture:** QuadTimer measures PWM duty cycle in hardware.
2. **Measurement:** AlphaBeta filter smooths the raw angle and derives velocity without phase-delaying the signal too much.
3. **Scaling:** Motor-space angles are divided by the hardware gear ratios to yield correct joint-space angles.
4. **Trajectory Following:** The [LinearInterpolator](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/control/Interpolator.h#5-61) receives sparse ROS commands (e.g. at 25Hz or 100Hz) and generates a smooth 1 kHz micro-setpoint path. This prevents the motors from taking "stair-step" violent jumps every time a new ROS packet arrives.
5. **Control Law:** A Proportional + Feedforward (P + FF) controller calculates the velocity error.
6. **Actuation:** [ActuatorModel](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/hal/ActuatorModel.h#79-139) converts radial velocity to stepper pulses, and [FlexPWMGenerator](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/hal/FlexPWMGenerator.h#23-25) outputs hardware PWM pulses precisely.

---

## 3. Potential Architectural Weaknesses (To Watch For)

While the structure is excellent, here are 3 things to keep an eye on during physical testing:

### A. Missing Integral (I) Term in Control Law
The control law (`velocity_command = Kp * pos_error + vel_ff`) is **P + FF only**.
* **Risk:** Stepper motors can drop steps if external forces are applied, or if friction is high. A purely Proportional controller cannot overcome steady-state error (e.g. if the arm sags under gravity slightly, P might generate a command too small to move the stepper).
* **Mitigation:** The Feedforward (`vel_ff`) helps tremendously during dynamic movement. If you notice the robot stopping slightly short of targets and staying there, you may need to add a small Integral sum to the control law in [main.cpp](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/main.cpp).

### B. Interpolator Dynamic Duration Logic
In [loop()](file:///home/kareem/Desktop/PAROL6_URDF/parol6_firmware/src/main.cpp#269-330), the code dynamically guesses the ROS packet frequency:
```cpp
delta_ms = current_tick - last_cmd_ts;
if (delta_ms > 100) delta_ms = 100;
interpolator[i].set_target(cmd.pos, cmd.vel, delta_ms);
```
* **Risk:** If the ROS PC drops a packet, `delta_ms` will instantly double. The interpolator will then stretch the *next* movement over 2x the time, causing the arm to noticeably slow down, then speed up when the next packet arrives on time.
* **Mitigation:** If motion is stuttery, it's better to lock `delta_ms` to a constant value that matches your ROS `ros2_control` update rate (e.g. exactly `10` for 100Hz) rather than measuring the network jitter.

### C. "Homing" / Absolute Initialization
The MT6816 encoder is absolute within **one motor revolution**, but because of the 6.4x to 20x gear ratios, the joint can be in several physical positions that map to the same motor-encoder value.
* **Risk:** Right now, the firmware powers on and assumes the *current physical position* is the joint-space position derived from the raw motor angle. It has no way to know which of the 20 possible motor revolutions the joint is actually in.
* **Mitigation:** You MUST mechanically home the robot (or have it in a known pose) before powering on the Teensy. A true multi-turn absolute homing routine using the limit switches (`LIMIT_PINS`) will eventually need to be written to solve this cleanly.
