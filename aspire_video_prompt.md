# ASPIRE Promo Video — Unified Structure & Prompt

> **Project:** ASPIRE — Autonomous System for Path Identification and Robotic Execution  
> **Platform:** PAROL6 6-DOF Robot Arm  
> **Duration:** ~2.5–3 minutes | **Style:** Cinematic engineering product reveal

---

## What Files to Share with Gemini
> [!IMPORTANT]
> Share these **exact files** from the `images/` directory in your workspace to give Gemini maximum visual context:

### ✅ Must Share
| File from `images/` | Why |
|---|---|
| `the robot Executing the identified path and calculated trajectory.png` | Shows the physical robot executing the task |
| `GUI Operator's point of view interaction.png` | The GUI as the central product interface. Essential for Scene 2. |
| `GUI Live camera view.png` | Shows the live feed prior to detection. |
| `identified path with point cloud and generated path in rviz.png` | MoveIt motion planning layer visualization. Essential for Scene 4. |
| `masked image after image processing applied and red line identifies the desired path.png` | Shows the vision pipeline output. Essential for Scene 3. |
| `Auto path identification with AI Model (YOLO).png` | Better view of the AI model's output in the GUI. |
| `camera calibrated frame, ArUco frame, and camera point cloud.png` | Shows 3D spatial awareness of the system in RViz. |

> [!NOTE]
> Do NOT share the old `system_architecture.png` in the root folder — it shows the ESP32 era and Ignition Gazebo simulation, which is legacy and misleading. Use the new architecture diagrams from your presentations instead if needed.

---

## Unified Video Structure (8 Scenes, ~2:40 total)

```
SCENE 1   [0:00–0:20]   THE PROBLEM — Why does this exist?
SCENE 2   [0:20–0:40]   THE SOLUTION — What is ASPIRE?
SCENE 3   [0:40–1:10]   THE PIPELINE — How it perceives the world (Modes Explained)
SCENE 4   [1:10–1:35]   MOTION PLANNING — How it plans collision-free paths
SCENE 5   [1:35–2:05]   THE HARDWARE BRIDGE — How commands reach the motors
SCENE 6   [2:05–2:20]   DECENTRALIZED CONTROL + FOC — What happens inside each joint
SCENE 7   [2:20–2:35]   PROBLEMS SOLVED — Real engineering challenges overcome
SCENE 8   [2:35–2:50]   END CARD — The complete picture
```

---

## Unified Final Prompt (Script Format)

---

> **Title:** ASPIRE — Autonomous Welding Path Identification and Robotic Execution  
> **Format:** 2:40 cinematic video. Dark background. Electric blue, orange, and white accents. Clean sans-serif typography (Inter or Roboto). Animated diagram transitions between scenes. Background score: calm but tense instrumental — builds in intensity as the pipeline unfolds.

---

> ### SCENE 1 — THE PROBLEM (0:00–0:20)
>
> **Visuals:** Factory floor. A traditional industrial robot arm repeating the same path, frozen in routine. A technician manually programming waypoints one by one at a teach pendant.
>
> **On-screen text (fade in, line by line):**
> - "Traditional welding robots are blind."
> - "They must be manually taught — every single weld path."
> - "Move the workpiece 1 millimeter? Start over."
> - "Industrial systems cost $100,000+. Inflexible. Closed. Fragile."
>
> **End beat:** Black screen. One line: *"We built something different."*

---

> ### SCENE 2 — THE SOLUTION: ASPIRE (0:20–0:40)
>
> **Visuals:** ASPIRE logo animates in. Cut to the PySide6 GUI opening on screen — clean, organized, with node status lights. An operator clicks one button. The camera feed activates.
>
> **Narration / on-screen text:**  
> *"ASPIRE — Autonomous System for Path Identification and Robotic Execution."*
>
> **Three pillars appear on screen as animated icons:**
> - 👁️ Perceive — Kinect v2 RGB-D camera sees the workpiece
> - 🧠 Plan — AI identifies the weld seam. MoveIt 2 plans the trajectory
> - ⚙️ Execute — PAROL6 6-DOF robot arm moves with precision
>
> **Key clarification text (important to prevent misunderstanding):**  
> *"ASPIRE does not perform actual welding. It autonomously identifies the welding path visually, plans the robot trajectory, and physically traces it — end to end, without human teaching."*

---

> ### SCENE 3 — THE VISION PIPELINE: Three Modes of Perception (0:40–1:10)
>
> **Visuals:** Animated pipeline diagram — each node lights up sequentially as data flows downward.
>
> **Step 1 — RGB-D Spatial Awareness**  
> Show: RGB image of workpiece side-by-side with a depth map.  
> Text: *"Microsoft Kinect v2 captures a synchronized color and depth frame. True 3D spatial awareness at 30 frames per second."*
>
> **Step 2 — Multi-Modal Path Identification**  
> Show: 3-way split-screen showing the three different detection modes working on the workpiece.  
> Text: *"Engineered for flexibility, ASPIRE uses three interchangeable detection modes to identify the weld seam:"*
>   - **1. AI Mode (YOLOv8):** *"A trained neural network recognizes complex weld seams automatically."*
>   - **2. Color Mode (Computer Vision):** *"Detects specific painted guide lines using color thresholding and filtering."*
>   - **3. Manual Mode:** *"The operator draws a custom path directly onto the live camera feed."*
>
> **Step 3 — 2D to 3D Projection (depth_matcher)**  
> Show: 2D mask pixel → mathematical back-projection → 3D point in space.  
> Text: *"Whichever mode is used, every 2D pixel on the path is mathematically back-projected into 3D using the camera's depth sensor and intrinsic calibration."*
>
> **Step 4 — Path Generation (path_generator)**  
> Show: noisy point cloud → PCA principal axis → smooth B-Spline curve.  
> Text: *"A B-Spline curve is fitted through the raw 3D points — transforming noisy sensor data into a continuous, mathematically smooth robotic trajectory."*

---

> ### SCENE 4 — MOTION PLANNING: Collision-Free Execution (1:10–1:35)
>
> **Visuals:** RViz visualization — robot arm shown in environment with the planned green path overlaid. 
>
> **Step 6 — MoveIt 2 Trajectory Planning**  
> Text: *"The 3D trajectory is handed to MoveIt 2, the industry standard for motion planning."*
>
> **Three sub-steps (fast flash):**
> - 🔢 *Inverse Kinematics — calculating the 6 joint angles needed for every micro-movement*
> - 🚫 *Collision checking — verifying the movement against a digital twin of the environment*
> - 📐 *Adaptive Execution — dynamically adjusting the stepping resolution for complex shapes*
>
> **Text:** *"The output is a time-parameterized joint trajectory — mathematically guaranteed to be collision-free before the physical robot even twitches."*  

---

> ### SCENE 5 — THE HARDWARE BRIDGE: From Software to Steel (1:35–2:05)
>
> **Visuals:** Split screen — left: C++ code (`parol6_system.cpp`). Right: physical PAROL6 arm. Between them: a data stream animation.
>
> **Text:** *"A custom C++ ros2_control Hardware Interface Plugin bridges the high-level software with the physical machine."*
>
> **What it does (animated list):**
> - *Every 40ms:* Packages joint speeds and positions into an ASCII serial packet
> - *Deterministic Timing:* Enforces a strict 25Hz loop, completely decoupling the slow AI processing from the rigid timing the physical motors demand
> - *Safety First:* Parses acknowledgment feedback, drops stale packets, and monitors timeout latency

---

> ### SCENE 6 — DECENTRALIZED CONTROL + FOC: Inside Each Joint (2:05–2:20)
>
> **Visuals:** One UART packet arrives at the STM32 chip. It fans out into 6 simultaneous parallel streams — one per joint.
>
> **Text:** *"The main STM32 Blackpill microcontroller receives the command packet and distributes it to 6 independent, parallel joint control loops."*
>
> **Per-joint loop (animate one joint, then zoom out to show 6 parallel loops):**
> 1. 📡 Reads absolute magnetic encoder positions via independent hardware timers 
> 2. 🔍 Applies statistical filtering (Median-of-3 + Exponential Moving Average) to eliminate sensor noise
> 3. 📈 Calculates required motor velocity (Feedforward + Proportional Error Correction)
> 4. ⚙️ Outputs precise Step/Direction signals to the motors
>
> **Transition to MKS SERVO42C Motors:**
>
> **Text:** *"Commands stream into the PAROL6's MKS SERVO42C motors. These are not basic steppers—they contain their own embedded microcontrollers running Field-Oriented Control (FOC)."*
>
> **What FOC means (simple, animated text):**
> *"FOC dynamically shapes the magnetic torque inside the specific motor windings based on the rotor's exact position. This eliminates lost steps, minimizes heat, and ensures ultra-smooth, industrial-grade motion at low speeds."*
>
> **Final line:** *"6 independent control loops on the STM32. 6 FOC drivers on the motors. All running simultaneously at 500 Hz."*

---

> ### SCENE 7 — REAL ENGINEERING CHALLENGES SOLVED (2:20–2:35)
>
> **Style:** Fast-cut montage. Each challenge appears as a headline followed by the solution. 3 seconds per item.
>
> | Challenge | How We Solved It |
> |---|---|
> | 🌍 **Locale Bug** — Non-US systems corrupted motor packets by using commas instead of decimal points | Fixed with `std::setlocale` override in `on_init()` to strictly enforce standard formatting |
> | 🦾 **Violent Arm Startup** — Attempting motion before the absolute encoders initialized | Set the controller to strictly block all commands until physical encoder feedback is fully validated |
> | 📉 **Noisy Depth Data** — Point cloud anomalies created a jagged, erratic path | Mathematical B-Spline curve fitting transformed the raw noise into a polished, traceable line |

---

> ### SCENE 8 — END CARD (2:35–2:50)
>
> **Visuals:** The complete system diagram (5-layer stack) fades in, fully animated.
>
> **Text reads out the full stack from top to bottom:**
> ```
> Vision GUI (PySide6)
>   ↓  trigger
> 3-Mode Vision Pipeline (YOLO / Color / Manual) + Kinect v2 
>   ↓  nav_msgs/Path
> MoveIt 2 (IK + collision planning)
>   ↓  FollowJointTrajectory
> C++ ros2_control Hardware Interface (25 Hz UART)
>   ↓  ASCII serial
> STM32 Blackpill (500 Hz, 6 parallel control loops)
>   ↓  Step/Dir
> MKS SERVO42C × 6 (Field-Oriented Control)
>   ↓
> PAROL6 6-DOF Robot Autonomous Execution
> ```
>
> **Final text:**  
> *"ASPIRE — From camera pixel to robot motion. Fully autonomous. Open source."*
>
> Team name | University | Year  
> Stack logos: ROS 2 Humble | MoveIt 2 | YOLOv8 | Docker | STM32 | PySide6

---

## 🔤 RAW PROMPT FOR GEMINI NANO BANANA
**(Copy & Paste this cohesive block directly into your AI Video/Storyboard tool)**

Generate a 2.5-minute cinematic, fast-paced tech startup product reveal video for an engineering project called ASPIRE (Autonomous System for Path Identification and Robotic Execution). The project is an open-source, vision-guided 6-DOF robotic arm (PAROL6) that uses a Kinect v2 RGB-D camera and ROS 2 middleware to autonomously identify welding seams and execute collision-free robotic paths, serving as an alternative to inflexible $100,000+ industrial systems.

The video should have a dark, sleek aesthetic with electric blue and orange accents. It should tell a complete story from high-level software perception down to physical steel execution. 

Start by showing the problem: traditional robots are rigid, require expensive fixed jigs, and must be manually taught every single waypoint by a human with a pendant. Then reveal the ASPIRE solution: a PySide6 GUI interface that controls everything with a single click. 

Crucially, demonstrate the flexibility of the Vision Pipeline using a 3-way split-screen to explain its three distinct detection modes: 1) AI Mode (using a YOLOv8 neural network to identify complex weld geometries), 2) Color Mode (traditional computer vision filtering to detect specific painted reference lines), and 3) Manual Mode (an operator drawing a custom path directly onto the live GUI camera canvas). Show how every 2D pixel from these masks is mathematically back-projected into a 3D physical coordinate using the camera's Time-of-Flight depth sensor. Then, show a mathematical B-Spline curve smoothing out the noisy 3D point cloud into a clean, traceable trajectory.

Next, show the MoveIt 2 motion planning software calculating the complex 6-axis inverse kinematics and ensuring the movement is collision-free within a digital twin of the environment. 

Transition to a split-screen showing C++ code executing a 25Hz deterministic UART hardware bridge next to the physical robot. Emphasize the decentralized control architecture: an STM32 Blackpill central microcontroller receives the high-level commands and runs 6 independent, parallel control loops at 500 Hz. These loops read absolute encoder feedback, apply statistical filtering (EMA/Median), and output calculated speeds to 6 MKS SERVO42C motors. Use sleek technical text to explain how these specific motors run embedded Field-Oriented Control (FOC)—dynamically managing magnetic torque at the winding level to guarantee ultra-smooth, industrial-grade motion without dropped steps. 

Conclude with a fast-paced montage of three engineering solutions (fixing a locale parsing bug that corrupted data packets, blocking violent startup motion by waiting for firm encoder feedback, and smoothing sensor noise). End on an animated diagram of the entire 5-layer software/hardware stack and a bold title card: "ASPIRE. From pixel to motion. Fully autonomous." 

*(Note to AI System: Do NOT generate visuals of active arc welding, fire, or sparks. The system demonstrates autonomous visual seam identification and robotic physical trajectory execution by tracing the path with a metal pointer tool over red and green blocks, not an active welder.)*
