# Firmware & Hardware-in-the-Loop (HIL) Troubleshooting

This document is dedicated to solving deeply technical issues encountered when bridging the ROS 2 environment with the Teensy 4.1 hardware (or ESP32 simulation), specifically concerning the Hardware Abstraction Layer, Action Servers, and Real-Time control loop timing.

---

## 🛑 Controller / RViz Crashes

### Issue 1: Controller Manager Segfaults on Launch (`exit code -11`)

**Symptoms:**
When launching the hardware interface, the `ros2_control_node` crashes instantly, taking down the robot state publisher as well.
```
[ERROR] [ros2_control_node]: process has died [exit code -11]
```

**Root Cause:**
In ROS 2, `rclcpp::Time` instances must be initialized with the same clock base before performing subtraction. Subtracting a default-initialized `rclcpp::Time` (which acts as `RCL_ROS_TIME`) from a monotonic `time` parameter passed into a hardware interface `read()` loop (`RCL_STEADY_TIME`) will throw an unhandled C++ exception that silently aborts the node.

**Fix:**
Never directly subtract raw `rclcpp::Time` headers in hardware loops unless both are explicitly instantiated against `clock_.now()`. Instead, perform float arithmetic on their extracted seconds:

```cpp
// Bad: Throws fatal exception
double dt = (time - last_rx_time_).seconds();

// Good: Safe explicit float math
double dt = std::abs(time.seconds() - last_rx_time_.seconds());
```

---

### Issue 2: RViz Segfaults AFTER Successful Trajectory Execution

**Symptoms:**
The terminal log shows that MoveIt executed the trajectory beautifully:
```
[move_group]: Completed trajectory execution with status SUCCEEDED
```
But RViz immediately throws an error and crashes the entire GUI:
```
[rviz2]: unknown result response, ignoring...
[ERROR] [rviz2]: process has died [exit code -11]
```

**Root Cause:**
This is an Action Server namespace mismatch. In ROS 2 Humble, the Controller Manager automatically maps the result topic for the `joint_trajectory_controller`. If you explicitly declare `action_ns: follow_joint_trajectory` in your `moveit_controllers.yaml`, RViz gets double-nested (`/follow_joint_trajectory/follow_joint_trajectory`) and listens to a ghost pointer when trying to parse the success message.

**Fix:**
Remove the redundant `action_ns` from `parol6_moveit_config/config/moveit_controllers.yaml`:
```yaml
  parol6_arm_controller:
    type: FollowJointTrajectory
    # action_ns: follow_joint_trajectory  <-- DELETE THIS LINE
    default: true
```
Then re-build the config package:
```bash
colcon build --packages-select parol6_moveit_config
```

---

## 🕒 Timing & Trajectory Aborts

### Issue 3: MoveIt Reports "TIMED_OUT" Despite Robot Moving

**Symptoms:**
During Hardware-in-the-Loop (HIL) simulation or real execution with encoder noise, the RViz robot model animates perfectly, but the execution eventually aborts with:
```
[move_group]: Controller is taking too long to execute trajectory... TIMED_OUT
```

**Root Cause:**
MoveIt's execution monitor enforces mathematical state trajectory margins. If you comment out the `constraints` block in `ros2_controllers.yaml`, MoveIt defaults to its internal boundary (usually <0.01 rad). Even minuscule floating-point rounding errors caused by the UART feedback loop or serial parser will cause the monitor to instantly preempt the path for "violating constraints."

**Fix:**
Re-instate the explicit constraints in `config/ros2_controllers.yaml`, but aggressively relax them to `999.0` to disable the preemption monitor altogether during spoofing validations:

```yaml
    constraints:
      stopped_velocity_tolerance: 999.0
      goal_time: 0.0
      joint_L1: { trajectory: 999.0, goal: 999.0 }
      # ... repeat for all joints
```

---

### Issue 4: HIL Spoofing Initialization Instability (`NaN` Jumps)

**Symptoms:**
When running the robot offline (spoofing commands directly to state feedback to test planners), the arm jumps erratically or aborts instantly at the `start()` request. 

**Root Cause:**
MoveIt controller arrays initialize to `0.0`. If you aggressively spoof `hw_state_positions_[i] = hw_command_positions_[i]` continuously from boot, you will overwrite the actual starting URDF position of the arm with zeros, creating a massive start-point deviation that permanently condemns the trajectory tracking monitor.

**Fix:**
Always explicitly filter out `NaN` and `0.0` initialization states before reflecting commands to state:

```cpp
  for (size_t i = 0; i < 6; ++i) {
    if (!std::isnan(hw_command_positions_[i])) {
      hw_state_positions_[i] = hw_command_positions_[i];
    }
  }
```

---

### Issue 5: RViz Throws "unknown goal response" and Segfaults Despite Robot Moving

**Symptoms:**
You click "Plan and Execute" in RViz. The robot physically moves (or the simulation visualizer shows movement), but the MoveIt planner instantly says `MoveGroupInterface::move() failed or timeout reached` and you see:
```
[rviz2]: unknown goal response, ignoring...
[ERROR] [rviz2]: process has died [exit code -11]
```

**Root Cause:**
Orphaned background processes! If you restart the HIL test or the Docker container without cleanly killing previous ROS 2 nodes, multiple `move_group` instances will run concurrently in the background. When RViz sends an Action Server request to execute a trajectory, *all of the orphaned `move_group` nodes* try to reply at the exact same millisecond. RViz receives multiple conflicting Action Server signatures, panics, immediately aborts the GUI request, and often segfaults.

**Fix:**
Ensure your launch scripts forcefully purge all zombie `move_group`, `rviz2`, and `ros2_control_node` processes before spinning up a new instance:

```bash
# Add this to the top of your start_hil_test.sh script
echo "Cleaning up any orphaned background nodes..."
docker exec parol6_dev pkill -9 -f "move_group|rviz2|ros2_control_node|robot_state_publisher" || true
```

---

### Issue 6: Robot Violently Jumps on MoveIt Execution Start

**Symptoms:**
You send an trajectory from MoveIt. As soon as the trajectory begins, one or more joints violently jump to a random position and MoveIt instantly throws `TIMED_OUT` or `Goal Tolerance Violated`.

**Root Cause:**
A serialization format mismatch between the ROS 2 Hardware Interface (`parol6_system.cpp`) and the Teensy Firmware parser (`SerialTransport.h`). If ROS packs the positions and velocities *interleaved* (`<seq, p1, v1, p2, v2...>`), but the Teensy parses them *grouped* (`<seq, p1, p2, p3... v1, v2...>`), the Teensy will interpret Joint 1's velocity command as Joint 2's position target. Since velocities are small numbers (e.g. `0.2 rad/s`), Joint 2 attempts to instantly travel to position `0.2 rad`, causing a violent mechanical jump and tracking abortion.

**Fix:**
Ensure `parol6_system.cpp` packs its arrays in the exact grouped format the firmware expects:
```cpp
// Correct format: Grouped
snprintf(buffer, sizeof(buffer), "<%u,%.3f,%.3f...%.3f,%.3f...>", 
         seq, pos[0], pos[1], /*...*/ vel[0], vel[1] /*...*/);
```

---

### Issue 7: Robot Spins Uncontrollably After `<HOME>` Command

**Symptoms:**
When pressing the `HOME ALL` button in the UI, a joint that has a homing offset greater than radians `> 2π` (e.g. 1.2 revolutions) instantly begins spinning out of control.

**Root Cause:**
If the `AlphaBetaFilter` is initialized with a raw float `> 2π`, it assumes the raw hardware magnetic sensor suddenly jumped from `0` to `> 2π`. The MT6816 encoder *only* outputs `[0, 2π)`. If the filter doesn't strip the modulo `2π` component from the initialization vector and store it as a structural multi-turn offset, the filter tracking math will explode trying to "catch up" to an impossible continuous rotation error.

**Fix:**
The observer initialization must explicitly decouple the hardware modulo frame from the multi-turn integer offset:
```cpp
void set_initial_position(float initial_rad) {
    last_raw_angle_ = fmodf(initial_rad, 2.0f * M_PI);
    turn_offset_ = initial_rad - last_raw_angle_;
}
```

---

## ⏱️ Hard Real-Time & Hardware Diagnostics

With the transition to the hard real-time Teensy 4.1 architecture (Phase 4), new physical and timing failure modes exist outside of the ROS domain. We have implemented several diagnostic testing modes inside `main.cpp` (controllable via the Configurator GUI) to isolate mechanical issues from PID issues.

### Diagnostic 1: Open-Loop Mode Bypass

**Use Case:** A motor is making terrible grinding noises or missing steps, and you don't know if the PID loop is unstable or if the mechanical belt is too tight.
**How it works:** Check `Open-Loop Mode` in the GUI and set a fixed `Open-Loop Hz`. The firmware completely bypasses the MT6816 observer and the PID control law, issuing a hardcoded step frequency directly to the stepper driver. If the motor *still* binds in Open-Loop mode, the issue is entirely mechanical or electrical (current limit on the driver).

### Diagnostic 2: Internal Sine Sweep Test

**Use Case:** The robot runs perfectly using fake hardware, but micro-stutters wildly when commanded by MoveIt in real life.
**How it works:** Check `Sine Test Mode` in the GUI. The firmware ignores all USB trajectories and generates a perfect `0.5 Hz` sine wave internally. If the robot moves perfectly smooth during the Sine Test, but stutters during MoveIt execution, the issue is USB packet jitter (or the ROS `ControllerManager` loop rate constraint), *not* the PID tuning. 

### Diagnostic 3: Interpolator Duration Lock

**Use Case:** USB latency causes consecutive packets to arrive at `18ms`, `22ms`, `15ms`, `25ms` intervals. The dynamic interpolator calculates violent velocity spikes to bridge the erratic gaps.
**How it works:** Check `Lock Duration to ROS Rate` in the GUI. The firmware ignores actual packet arrival timestamps and forces the interpolator to assume a perfect gap (e.g. exactly `20ms` for `50 Hz`). This acts as a powerful low-pass jitter filter at the expense of slight lagging tracking error.

---

### Issue 8: Control Jitter Measuring > 1 µs

**Symptoms:**
When profiling the 1 kHz `run_control_loop_isr` with an oscilloscope on the `ISR_PROFILER_PIN` or reading the DWT Cycle Counter telemetry, you notice the jitter (deviation from the strict 1000 µs interval) occasionally spikes to 15 µs or more.

**Root Cause:**
If QuadTimers are used, jitter should be `< 1 µs`. A spike indicates an architectural violation. Common culprits:
1. **Rogue Software Interrupts**: A background library (e.g., USB Serial or a standard Arduino library) activated a higher-priority interrupt that preempted the `IntervalTimer`.
2. **Cache Misses**: You didn't place the critical control arrays (e.g., `AlphaBetaFilter` states) in Tightly Coupled Memory (TCM), causing the Cortex-M7 D-cache to stall the CPU while fetching data from slower RAM.
3. **ISR Printf**: Someone accidentally left a `Serial.print()` inside the control loop.

**Fix:**
Ensure all arrays used inside the ISR are decorated with `EXTMEM` or explicitly routed to `DMAMEM`/TCM depending on the platform config. audit the `setup()` function to ensure no competing `IntervalTimers` or software interrupts are active.

---

### Issue 7: SafetySupervisor Triggers "ISR Overrun" Fault

**Symptoms:**
The robot abruptly halts, and the telemetry stream outputs a `FAULT_ISR_OVERRUN` error code. The system refuses to resume motion.

**Root Cause:**
The `run_control_loop_isr()` execution time exceeded the 1000 µs tick window (less a safety margin). Based on our profiling, the math + safety checks should complete in `< 25 µs`. An overrun means a fatal mathematical hang or an infinite loop occurred inside `ControlLaw` or `AlphaBetaFilter`. 
A common trigger is accidentally introduced floating-point division by zero (e.g., dividing by `delta_t` when `delta_t` evaluates to `0.0f`), which can cause the FPU to stall or throw an exception that delays execution.

**Fix:**
1. **Remove Divisions**: Audit the filter and interpolator math. Replace `A / B` with `A * (1.0f / B)` pre-computed in constructors where possible.
2. **Check Unwrapping Logic**: Ensure the 360-degree `M_PI` encoder unwrap bounds cannot get stuck in an infinite `while()` loop if the sensor completely disconnects and sends garbage floating-point noise. Always use bounded `if()` statements for angle wrapping.

---

## 🏗️ Core Architecture & Timing Domains

For new developers joining the team, it is critical to understand the real-time segregation within `main.cpp`.

### 1. The 1 kHz Hardware Timer ISR (`run_control_loop_isr`)
- **Strictly Deterministic:** This loop does **zero** waiting. It contains no `delay()`, no serial formatting, no unbounded loops. It runs mathematical calculations (P+FF control law, AlphaBeta filter, Interpolator tick) and writes directly to hardware registers.
- **Profiling Built-In:** The CPU cycle counter (`ARM_DWT_CYCCNT`) measures exactly how many microseconds the ISR takes, providing a verifiable proof of real-time execution limits.
- **Zero-Interrupt Sensor Capture:** The `QuadTimerEncoder` uses hardware gated-counters. Instead of the CPU being interrupted 10,000 times a second to measure PWM pulses, the hardware does it silently. 

### 2. The Background Loop (`loop()`)
- Handles slow, unpredictable tasks: reading USB/UART bytes, decoding ROS packets, and `sprintf` generating telemetry output.
- **Lock-free Data Transfer:** It uses a `CircularBuffer` to pass parsed commands from the background loop to the ISR. 
- When updating telemetry, it briefly pauses interrupts (`noInterrupts()`) just long enough to copy the floats, preventing "data tearing".

### 3. Absolute Initialization & Homing
The MT6816 encoder is absolute within **one motor revolution**, but because of the gear ratios (e.g. 20:1), the joint can be in up to 20 physical positions that map to the exact same motor-encoder value.
* **Important:** The firmware currently assumes the *current physical position on boot* is exactly the joint-space position derived from the raw motor angle. 
* **Action Required:** You MUST manually align/home the robot to its known 0-pose before powering on the Teensy, until the limit-switch absolute homing sequence is implemented in firmware.

---

## 🧲 Encoder Hardware: QuadTimers vs Interrupts

The MT6816 magnetic encoder outputs absolute angle data encoded in the duty cycle of a 971 Hz PWM signal. 

**DO NOT use Interrupt Mode to read these encoders.**

### Why `PwmCaptureEncoder` (Interrupt Mode) Fails
The naive interrupt stub has severe flaws:
1. **Hardcoded Assumed Period:** Assumes 1000 µs frame; the MT6816 is actually 1029.75 µs. This 3% error skews all angles.
2. **Naive Math:** Maps 0-100% duty to 0-360°. The MT6816 actually maps data into a 4096-count window nested inside a 4119-count total frame.
3. **Low Resolution:** Arduino `micros()` provides 1 µs resolution. The MT6816 PWM clock is 250 ns. `micros()` destroys 75% of the sensor's physical precision.
4. **CPU Starvation:** 6 joints × 2 edges × 1000 Hz = 12,000 interrupts per second, destroying the PID control loop timing.

### Why `QuadTimerEncoder` is Superior
The QuadTimer implementation uses the NXP i.MXRT hardware timers in "Gated Count" mode.
* It measures the HIGH time at **53.3 nanosecond** resolution (IP-BUS ÷ 8).
* It accumulates ticks over 10 full MT6816 frames (~10 ms) before calculating the angle, completely eliminating mid-frame sampling artifacts and multiplying the precision.
* It implements the precise `(duty * 4119 - 1) / 4096` math from the MT6816 datasheet.
* **It uses exactly 0% CPU overhead.** The 1 kHz ISR simply reads a register (`CNTR`).

**Conclusion:** Always use the natively supported QuadTimer pins (10, 11, 12, 14, 15, 18). Interrupt mode is deprecated and removed.
