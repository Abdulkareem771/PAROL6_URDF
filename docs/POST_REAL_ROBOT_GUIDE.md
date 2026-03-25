# PAROL6 Post-Real-Robot Guide

Congratulations on successfully getting the real PAROL6 hardware to move! The first successful motion is a major milestone, but a robotic arm requires significantly more tuning before it is ready for real workloads like trajectory planning, welding, or perception tasks.

This document covers what you should do **after** you have verified basic, low-speed joint jogging and ROS 2 communication.

---

## 1. Safety Systems Check

Before running any high-speed motions, manually verify every hardware safety loop:
* **E-Stops:** Hit every Emergency Stop button while jogging. Ensure the system halts immediately and cannot be re-enabled until physically un-latched.
* **Limit Switches:** Carefully jog each joint into its limit switches (if equipped). Verify the `FAULT` state engages and MoveIt is notified of the failure.
* **Watchdog:** While jogging from the GUI, forcibly kill the serial connection. The driver should time out (`SOFT_ESTOP`) and halt motors within 200ms.

## 2. Dynamic PID Tuning (Per-Joint)

Initial motion proves your directions and connections are correct, but the default PID gains (`Kp`, `Ki`) in `config.h` are extremely conservative to prevent runaway oscillations. 

1. **Start with Base (J1):** Tune one joint at a time, moving from base to end-effector.
2. **Increase Kp (Proportional):** Command rapid `std_msgs/Float64` position steps. Increase Kp until the joint reaches the target snappily. If it rings or oscillates, back off ~10%.
3. **Add Ki (Integral):** If the joint stops *just short* of the target due to friction or gravity, slowly increase Ki. *Note: Watch out for integral windup; ensure the anti-windup resets are working.*
4. **Iterate under load:** Re-tune the upper joints (J4, J5, J6) carrying your actual intended payload (e.g., a welding torch dummy weight).

## 3. Alpha-Beta Filter Tuning (Encoder Noise)

If your real encoders have jitter (e.g., magnetic encoders flickering ±1 count while stationary), the velocity derived from them will be noisy. This noise will feed directly into your D-term (if added) or simply sound terrible.

* **Enable `FEATURE_ALPHABETA_FILTER`** in the configurator.
* Start with `ALPHA = 0.85` (trusts new position highly) and `BETA = 0.05` (smooths velocity).
* If velocity readouts still spike, lower `ALPHA` slightly and increase `BETA`.

## 4. Homing Calibration

Until you calibrate the `HOME_OFFSETS_RAD`, the robot will assume its "zero" position is literally where the physical limit switches are. This is almost never what MoveIt expects.

1. Home the robot using the GUI or `ros2 control` interface.
2. Read the `joint_states` topic — it should be exactly `[0,0,0,0,0,0]`.
3. Jog the robot manually (using a level or digital angle finder) until it visually perfectly matches the `[0,0,0,0,0,0]` pose defined in your RViz/URDF.
4. Record the *current* `joint_states` values. These values, negated, are your `HOME_OFFSETS_RAD`.
5. Enter these offsets into the Configurator's Joints tab and re-flash.

## 5. Integrating with MoveIt and Vision

Once tuned and homed, you can integrate with your perception stack.

* **Velocity/Accel Limits:** Ensure the real limits you determined during tuning are set in the Configurator. The Configurator will automatically generate `parol6_moveit_config/config/joint_limits.yaml` to ensure MoveIt never plans a path the hardware can't follow.
* **Trajectory Tolerance:** In your MoveIt config, you may need to increase the `path_tolerance` and `goal_tolerance` slightly. Simulated robots hit targets perfectly; real robots have gearbox backlash and friction.
* **Camera Calibration:** Re-run hand-eye calibration with the real robot. The kinematics of the printed hardware will differ slightly from the ideal URDF.

## 6. Long-Term Reliability

* **Thermal Management:** Monitor the stepper drivers (TMC2160/2209) and the Teensy. If they run hot during extended operations, add active cooling.
* **Belt Tension:** 3D printed components creep. Re-tension belts after the first 10 hours of operation.
* **Lubrication:** If using printed gearboxes or cycloidal drives, check for excessive wear particles and re-lubricate as necessary.
