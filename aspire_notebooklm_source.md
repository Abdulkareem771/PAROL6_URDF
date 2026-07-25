# ASPIRE: Autonomous System for Path Identification and Robotic Execution
## Technical Overview and Design Documentation

### Overview
ASPIRE (Autonomous System for Path Identification and Robotic Execution) is an open-source, vision-guided robotics system built to automate the path-planning and welding-seam tracing process. Built on the PAROL6 6-DOF (Degrees of Freedom) robotic arm platform, the system replaces traditional, expensive, and inflexible industrial teaching methods with autonomous, closed-loop visual perception and trajectory execution. 

Industrial robots traditionally operate "blindly," relying on human programmers to manually teach every single 3D waypoint using a teach pendant. If the physical workpiece shifts by even a single millimeter, the entire sequence must be manually reprogrammed. These rigid setups cost upwards of $100,000, are highly inflexible, and require significant downtime for reconfiguration. 

ASPIRE changes this paradigm by integrating:
1. **Multi-Modal Visual Perception:** Using a Microsoft Kinect v2 RGB-D camera to capture color and 3D spatial depth.
2. **AI & Computer Vision Processing:** Utilizing YOLOv8 object detection, classic color filtering, or manual operator input to locate path seams.
3. **Dynamic Motion Planning:** Resolving 6-axis inverse kinematics and collision checking in real time using MoveIt 2.
4. **Decentralized Embedded Control:** Executing deterministic motion via a custom STM32 microcontroller pipeline and Field-Oriented Control (FOC) motor drivers.

ASPIRE does not perform active arc welding itself; instead, it is a high-precision path tracing and execution system. For demonstrations and testing, it uses a brass pointer tool to trace complex seams on test workpieces, eliminating dangerous arc flames, smoke, and sparks from the research environment while maintaining sub-millimeter tracking accuracy.

---

## 1. System Architecture

The ASPIRE system is organized into a modular 5-layer software and hardware stack. This pipeline takes a raw camera frame and converts it step-by-step into high-precision, collision-free physical motion.

```
+-------------------------------------------------------------+
|                Layer 1: PySide6 Operator GUI                |
|  Central command interface, system status, and live feeds.  |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|                 Layer 2: Vision Pipeline                    |
| - RGB-D Spatial Sensing (Microsoft Kinect v2)               |
| - Seam Detection (YOLOv8 AI / OpenCV Color / Manual Draw)   |
| - depth_matcher (2D-to-3D Back-Projection)                 |
| - path_generator (B-Spline Curve Interpolation)             |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|              Layer 3: MoveIt 2 Motion Planning              |
|  Inverse Kinematics (IK), collision check, resolution gate. |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|            Layer 4: C++ ros2_control Hardware Bridge         |
|  25 Hz deterministic ASCII Serial packets over USB connection|
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|                Layer 5: STM32 Blackpill MCU                 |
|  500 Hz parallel joint loops, MT6816 absolute encoder state |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|             Layer 6: MKS SERVO42C Motor Drivers             |
|  Embedded Field-Oriented Control (FOC) for low-speed smooth |
+-------------------------------------------------------------+
```

### Layer 1: PySide6 Operator GUI
The central cockpit of the system is a custom PySide6 graphical user interface. Rather than typing raw commands in a terminal, the operator manages the launch sequence, monitors camera feeds, chooses detection modes, and triggers path execution from a unified console. Status lights keep the operator informed of individual ROS node health, processing rates, and connection states.

### Layer 2: The Vision Pipeline (Perception)
* **RGB-D Spatial Sensing:** The Kinect v2 camera captures synchronized 1920x1080 color frames and 512x424 depth maps. Depth map calculations rely on Time-of-Flight (ToF) technology to construct a 3D points representation of the environment.
* **Three Modes of Path Identification:**
  1. **AI Mode (YOLOv8):** A deep convolution neural network (YOLOv8 segment) is trained to automatically locate complex, non-linear weld seams. This allows for automated target identification under varying shop lighting conditions.
  2. **Color Mode (OpenCV Computer Vision):** For environments with clearly marked pathways, classic computer vision is used. The pipeline applies color segmentation (in HSV space) and morphological filtering to isolate painted guide lines.
  3. **Manual Mode:** A fallback system where the operator draws the path directly onto the GUI canvas. The pixels are collected into an ordered list.
* **2D-to-3D Projection (depth_matcher):** The isolated 2D pixels representing the weld seam are mapped to the depth buffer. Using the camera's intrinsic calibration matrix, the system mathematically back-projects each pixel into a physical 3D point relative to the camera frame.
* **Curve Fitting (path_generator):** Because depth data is inherently noisy and subject to outlier reflections, the raw points cannot be fed directly to the robot. The system orders the points using Principal Component Analysis (PCA) and fits a continuous B-Spline curve. This filters out spatial jitter, generating a mathematically smooth 3D trajectory.

### Layer 3: MoveIt 2 Motion Planning (Cognition)
The generated B-Spline trajectory is published as a `nav_msgs/Path` topic. The robot's motion planner uses MoveIt 2 to translate this 3D path into a sequence of joint actions.
* **Inverse Kinematics (IK):** Resolves the mathematical angles required for each of the 6 physical joints.
* **Collision Checking:** Validates the planned trajectory against a virtual "digital twin" of the work cell, ensuring the arm does not collide with the workpiece, camera brackets, or the workbench.
* **Adaptive Stepping:** During tight curves, the resolution of waypoints is dynamically scaled up to maintain tight geometric tolerances.

### Layer 4: C++ ros2_control Hardware Bridge
A custom C++ `ros2_control` system hardware interface acts as the link between the high-level planning software and the physical microcontroller. 
* Operates a deterministic write-read cycle at 25 Hz.
* Packages calculated joint positions and target velocities into a lightweight, ASCII-delimited serial packet: `<SEQ,J0_pos,J0_vel,...,J5_pos,J5_vel>\n`.
* Decouples the variable rates of the vision pipeline (1–5 Hz) from the strict, real-time timing required by the motor controller.

### Layer 5: STM32 Blackpill Microcontroller (Coordination)
Deep inside the electronic cabinet is the main STM32 Blackpill microcontroller. Working as the local hardware coordinator, the STM32 performs several real-time tasks at 500 Hz:
* Parses incoming serial packets and validates the data integrity.
* Runs 6 independent, parallel joint control loops in hardware.
* Decodes absolute magnetic rotation angles from MT6816 encoders via dedicated micro-timer registers.
* Applies a hybrid noise-rejection filter (Median-of-3 combined with an Exponential Moving Average) to prevent sensor jitter from destabilizing the control loop.
* Computes target velocities using a combined Feedforward and Proportional error-correction loop.
* Outputs precise Step and Direction hardware pulses.

### Layer 6: MKS SERVO42C Motor Drivers (Execution)
The step and direction signals from the STM32 drive 6 closed-loop MKS SERVO42C stepper motors. Unlike standard hobbyist stepper motors that move roughly between discrete phases, these motors represent high-efficiency actuators running embedded Field-Oriented Control (FOC).
* FOC measures the rotor's exact magnetic position using a built-in encoder.
* It dynamically shapes the sinusoidal current flowing into the motor windings, keeping the stator's magnetic field at a perfect 90-degree angle to the rotor.
* **Benefits:** This closed-loop control completely eliminates lost steps, dramatically reduces motor heat, lowers power consumption, and provides exceptionally smooth, high-torque rotations even at very low speeds—a necessity for uniform weld tracing.

---

## 2. Engineering Challenges and Solutions

Building the ASPIRE system required overcoming several complex hardware-software integration hurdles:

### The Locale Parsing Bug (Data Corruption)
* **Challenge:** During initial deployments in European countries, the C++ `ros2_control` serial parser would randomly fail, causing the robot to freeze or behave erratically.
* **Root Cause:** In standard C/C++, floating-point parsing (such as `std::sprintf`) respects the host operating system's geographic setting (locale). On European systems, numbers are formatted with commas (e.g., `1,57` instead of `1.57`). When the C++ hardware interface formatted the joint angles, it embedded commas in the packet, which broke the microcontroller's comma-separated serial parser.
* **Solution:** Explicitly overrode the locale settings within the C++ hardware initialization sequence using `std::setlocale(LC_NUMERIC, "C")`. This forces the compiler to format floating-point data with period decimals regardless of where the robot is deployed globally.

### Violent Arm Startup (Hardware Protection)
* **Challenge:** On power-up, the robot arm would occasionally jerk violently, triggering motor safety shutoffs.
* **Root Cause:** On startup, the MoveIt 2/ros2_control controllers would initialize and read a joint state of `0` before the physical absolute encoders had fully initialized and reported their coordinates. The controller would immediately command an instantaneous, maximum-speed move from `0` to the actual position, causing a mechanical shock.
* **Solution:** Implemented a startup handshake. The `ros2_control` interface now blocks trajectory execution until the STM32 micro returns a series of valid, non-zero encoder feedback packets over a 2-second stabilization window. Additionally, a 1-second "grace period" on the Teensy/STM32 firmware ignores high-velocity spikes during state synchronization.

### Sensor Depth Noise (Path Optimization)
* **Challenge:** Point clouds generated by the camera's depth sensor suffer from high noise (jitter and missing pixels) near edges. Feeding these raw coordinates directly to the motion planner caused the robot to move with a jagged, vibrating motion, resulting in jerky tracing.
* **Solution:** Introduced PCA-based ordering and a mathematical B-Spline interpolation pass in the `path_generator` node. By fitting a continuous parametric curve through the raw point clouds, the system calculates a smooth, continuous path. This filters out spatial noise and guarantees the robot moves smoothly.

---

## 3. Key Design Philosophy
ASPIRE is designed to be accessible and cost-effective. By using standard consumer cameras (Kinect v2), open-source middleware (ROS 2 Humble and MoveIt 2), affordable microcontrollers (STM32 Blackpill), and off-the-shelf closed-loop FOC motors (SERVO42C), ASPIRE provides a low-cost, high-performing alternative to commercial robotic systems. It proves that combining intelligent software filtering with modern closed-loop control algorithms can match or exceed the performance of rigid industrial architectures.
