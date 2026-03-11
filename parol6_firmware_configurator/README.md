# PAROL6 Firmware Configurator & Launcher

A unified desktop GUI to configure the PAROL6 robot hardware, tune control loops, flash the Teensy 4.1, and launch the full ROS 2 MoveIt stack.

## Overview

This tool eliminates the need to manually edit C++ headers (`config.h`), memorize bash arguments, or juggle multiple terminal windows.

### Features

| Tab | Description |
|-----|-------------|
| **📖 Docs** | In-app teammate guide — Quick Start, per-tab walkthroughs, limit switch wiring, homing, MoveIt, and troubleshooting. **Start here.** |
| **🔬 Protocol** | Phase-by-phase testing presets (phase0 → phase10). Load a preset, flash, and validate before advancing. |
| **⚙️ Features** | Compile-time feature flags — AlphaBeta filter, feedforward, safety supervisor, hardware PWM/Encoder, debug modes. |
| **🔩 Joints** | Per-joint pin assignments, gear ratios, PID gains, velocity limits, and limit switch config (type/pin/polarity/pull). |
| **📡 Comms** | Transport mode (USB/UART/Ethernet), ROS command rate, feedback rate, control loop rate. |
| **🕹 Jog** | Real-time slider-based single-joint jogging over serial. |
| **💬 Serial** | Raw serial terminal — see ACK frames and send `<HOME>` / `<ENABLE>` commands directly. |
| **📈 Oscilloscope** | Live position + velocity scope (PyQtGraph). Verify tracking without oscillation before enabling full stack. |
| **⚡ Flash** | Generate `config.h` → Compile with PlatformIO inside Docker → Flash to Teensy 4.1. |
| **🚀 ROS2 Launch** | Start Fake / Gazebo / Real Hardware MoveIt stacks. Logs stream into terminal panel below. |
| **⚠️ Faults** | Safety supervisor ESTOP log with timestamps and fault reasons. Export to CSV. |

## Setup

Runs on the **host machine** (not in Docker). Python 3.10+ and PyQt6 required.

```bash
pip install -r requirements.txt
python3 main.py
```

## Developer Guide

### 1. Config → Firmware Pipeline

When you change any setting in the GUI:

1. Open the **Flash** tab.
2. Click **Generate config.h** → writes to `parol6_firmware/generated/config.h`.
3. Click **Compile & Flash** → PlatformIO builds and uploads to the Teensy.

The firmware's `main.cpp` picks up all changes via `#include "../generated/config.h"`.

> [!IMPORTANT]
> The `Dir Inv` checkbox controls the **physical motor direction at the Teensy level** via `DIR_INVERT[]`.  
> The ROS hardware interface (`parol6_system.cpp`) has a separate `dir_signs[]` array for the **kinematic sign correction** seen by MoveIt. Both must remain consistent. The expected inversion set is: **J1 ✓ J2 – J3 ✓ J4 – J5 – J6 ✓**

### 2. Testing Motors (Jog & Oscilloscope)

Always validate limits on the bench before connecting to ROS:

1. Connect to the Teensy in the **Serial** tab (`/dev/ttyACM0`, 115200 baud).
2. Go to **Jog** — move sliders to command individual joints.
3. Go to **Oscilloscope** — verify actual encoder position tracks commanded position. If you see oscillation, lower `Kp` and re-flash.

### 3. Launching the ROS 2 Stack from the GUI

The **ROS2 Launch** tab calls the `.sh` launcher scripts located in `scripts/launchers/`.

| Mode | Script Called | Description |
|------|--------------|-------------|
| Fake Feedback | `launch_fake_feedback.sh` | Publishes sinusoidal joint states to animate RViz independently |
| Fake | `launch_moveit_fake.sh` | RViz + fake controllers (no hardware) |
| Simulation | `launch_gazebo_only.sh` + `launch_moveit_with_gazebo.sh` | Gazebo + MoveIt |
| Real Hardware | `launch_moveit_real_hw.sh` | Teensy-in-the-loop via `parol6_hardware` |

The scripts are **Docker-aware**: when called from inside the container (as the GUI does), they execute `ros2 launch` natively. When called from the host, they use `docker exec`.

See [`scripts/launchers/LAUNCH_METHODS.md`](../../scripts/launchers/LAUNCH_METHODS.md) for the full protocol.

### 4. Saved Profiles

Pre-configured `.json` profiles in `saved_configs/` map to hardware bring-up phases:

| File | Phase |
|------|-------|
| `phase0_hardware_check.json` | Encoder sanity, ISR timing |
| `phase1_encoder_interrupt.json` | Interrupt-based position read |
| `phase2_quadtimer.json` | QuadTimer zero-CPU encoder capture |
| `phase3_step_dir_test.json` | Open-loop STEP/DIR smoke test |
| `phase4_open_loop.json` | Full 6-axis open-loop |
| `phase5_closed_loop_j1.json` | Closed-loop J1 only |
| `phase6_with_filter.json` | Alpha-Beta observer enabled |
| `phase7_with_feedforward.json` | Velocity feedforward added |
| `phase8_with_watchdog.json` | Safety supervisor enabled |
| `phase9_multiaxis.json` | All 6 joints closed-loop |
| `phase10_full_stack.json` | Full stack + Ethernet ready |

### 5. Fault Logging

If the safety supervisor triggers an E-Stop (e.g. encoder glitch, velocity limit exceeded), the event is immediately logged in the **Faults** tab with a timestamp and fault code. Export to CSV for post-mortem debugging.

## Architecture

```
GUI (PyQt6, host)
    │
    ├── tabs/joints_tab.py  ── JointConfig model ──► code_generator.py ──► config.h
    ├── tabs/comms_tab.py   ── CommsConfig model ──►                    ──► config.h
    ├── tabs/features_tab.py── FeatureFlags model ──►                   ──► config.h
    │
    ├── tabs/flash_tab.py   ── PlatformIO CLI ────────────────────────► Teensy 4.1
    │
    ├── tabs/serial_tab.py  ── Serial port (115200 baud) ──────────────► Teensy (Jog/Scope)
    │
    └── tabs/launch_tab.py  ── subprocess.Popen ─► scripts/launchers/*.sh ─► Docker / ROS 2
```
