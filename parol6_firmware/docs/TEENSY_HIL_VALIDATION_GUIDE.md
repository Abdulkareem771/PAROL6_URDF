# PAROL6 Teensy HIL Validation & Quick Start Guide

This document provides a bulletproof workflow for teammates to rapidly replicate the Hardware-in-the-Loop (HIL) validation environment for the Teensy 4.1 firmware migration.

## 1. Quick Start: Flashing the Teensy

The build environment is completely contained within the `parol6-ultimate` Docker image to ensure binary consistency across all developer laptops.

### Prerequisites
*   Teensy 4.1 connected via USB.
*   Docker installed and running.
*   The `parol6-ultimate:latest` image built.

### Flash Command
Navigate to the firmware directory and run the automated script:
```bash
cd ~/Desktop/PAROL6_URDF/parol6_firmware
./flash_teensy.sh
```
*Note: The first time you flash a brand new Teensy, you may need to physically press the tiny white button on the board to enter programmed mode.*

---

## 2. Phase 1.5 vs Phase 3 Testing Workflows

We use an **ESP32 PWM Simulator** to inject synthetic motor encoder signals into the Teensy. This validates the 1 kHz control loop and the Alpha-Beta filter under heavy interrupt load without needing physical motors.

### ESP32 Simulator Setup
Flash the ESP32 simulator firmware (located in `vision_work/tests/encoder_simulator/`) to your ESP32. This generates 6 continuous PWM signals representing the 6 joints.

### Phase 1.5 Validation (Software Interrupts)
This phase used `attachInterrupt` to read the PWM signals. It serves as our baseline.
1. Wire the 6 ESP32 PWM pins to Teensy pins: **2, 3, 4, 5, 6, 7**.
2. Flash the Phase 1.5 firmware (git checkout the appropriate tag/commit).
3. Connect an oscilloscope to **Teensy Pin 13** (`ISR_PROFILER_PIN`).
4. **Expected Result:** The oscilloscope will show a 1 kHz square wave. The HIGH duration represents the execution time of the `run_control_loop_isr()`. You should observe ~6µs nominal execution time, but with **erratic jitter spiking up to 10-15µs** as the 6 external software interrupts randomly collide with the control math.

### Phase 3 Validation (Zero-Interrupt QuadTimers)
This is the final, hard real-time implementation using the i.MXRT1062 hardware timers.
1. Wire the 6 ESP32 PWM pins to the strictly hardware-mapped QuadTimer pins: **10, 11, 12, 14, 15, 18**.
2. Flash the `main` firmware branch. 
3. Connect an oscilloscope to **Teensy Pin 13** (`ISR_PROFILER_PIN`).
4. **Expected Result:** The `ISR_PROFILER_PIN` HIGH duration will drop significantly (to a rigid 1-2µs) because the CPU is no longer being interrupted by the PWM edges. The pulse width will be **dead constant** resulting in absolute zero-jitter determinism.

---

## 3. Full System HIL Integration Test

Once the firmware is validated on the oscilloscope, you can run the full ROS 2 hardware interface against the simulated Teensy.

1. Ensure the ESP32 is injecting PWM into the Teensy (Phase 3 pins: 10, 11, 12, 14, 15, 18).
2. Launch the fully automated HIL test script:
   ```bash
   cd ~/Desktop/PAROL6_URDF
   ./start_hil_test.sh
   ```
3. This will launch `ros2_control`, MoveIt, and RViz.
4. In RViz, use the interactive markers to drag the robot arm to a new Cartesian pose.
5. Click **"Plan and Execute"**.
6. **Validation:** Ensure the virtual robot animates smoothly to the target pose. The MoveIt `FollowJointTrajectory` action should return `SUCCESS`. This proves the entire pipeline—from ROS trajectory generation, serialized UART transmission, Teensy interpolation, Alpha-Beta filtering, and strict 1 kHz execution—is functioning flawlessly.
