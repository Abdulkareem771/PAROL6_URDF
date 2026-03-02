# PAROL6 Firmware Configurator

> A desktop GUI for testing and configuring PAROL6 Teensy 4.1 firmware — step by step, one phase at a time.

---

## Quick Start

```bash
# Clone and enter the project
cd PAROL6_URDF/parol6_firmware_configurator

# Install dependencies (once)
pip install -r requirements.txt

# Launch
python3 main.py
```

**Requirements:** Python ≥ 3.10, `PyQt6`, `pyqtgraph`, `pyserial`, `numpy`

> If running inside Docker, ensure X11 forwarding is enabled (`-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` in your `docker run` command).

---

## What This Tool Does

The configurator lets you:

1. **Toggle firmware features** (AlphaBeta filter, watchdog, encoder mode, etc.) via a GUI, without editing any C code.
2. **Generate `generated/config.h`** — a C header the firmware reads via `#include`. Click **Generate & Flash** in the Flash tab.
3. **Flash to Teensy** via PlatformIO (`pio run --upload`) with a live build log.
4. **Monitor serial output** with filter + timestamp support.
5. **Visualize live telemetry** (joint angles, velocities, ISR timing) on a real-time oscilloscope.
6. **Manually jog joints** one at a time for safe isolated testing.
7. **Log faults** automatically from serial output, exportable as CSV.

---

## App Layout

```
┌─ Toolbar: New | Open… | Save | Save As… | [Profile name] ──────────────────────┐
├─ Tabs ───────────────────────────────────────────────────────────────────────────┤
│  🔬 Protocol │ ⚙️ Features │ 🔩 Joints │ 📡 Comms │ 🕹 Jog │ 💬 Serial │ ...   │
├─ Status bar: [🔴 Disconnected] [0 pkt/s] [State: —]  Port: [▼] Baud: [▼] [⚡]──┤
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tab Reference

### 🔬 Protocol — Start Here
Phase cards **P0 → P10**, ordered by risk. Click **Load Preset** on a phase card to load the matching config, then switch to the **Flash** tab.

| Phase | What you test |
|:--|:--|
| P0 | Continuity checks (no motor power) |
| P1 | Encoder wiring — interrupt mode, hand-turn motor |
| P2 | Encoder — switch to QuadTimer, compare ISR time |
| P3 | STEP/DIR signal integrity (oscilloscope required) |
| P4 | Open-loop rotation (first motor movement) |
| P5 | Closed-loop J1 only, no filter |
| P6 | Add AlphaBeta filter |
| P7 | Add velocity feedforward |
| P8 | Add watchdog — cable-pull test |
| P9 | Multi-axis bring-up |
| P10 | Full feature stack at 100 Hz |

Full procedure: [`docs/TESTING_PROTOCOL.md`](docs/TESTING_PROTOCOL.md)

---

### ⚙️ Features — Toggle Firmware Capabilities

Each checkbox maps to a `#define` in the generated `config.h`. Disable features to isolate problems.

| Feature | What disabling it does |
|:--|:--|
| AlphaBeta Filter | Raw encoder readings only — useful to see noise floor |
| Velocity Feedforward | Pure P-control — larger tracking error but simpler |
| Watchdog | Motor won't stop on cable pull — **bench only** |
| Safety Supervisor | No velocity limit enforcement — **bench only** |
| Anti-Glitch Filter | Accepts all encoder deltas, including impossible jumps |
| Encoder Test Mode | Disables control loop entirely — just reads encoders |
| Fixed STEP Freq | All STEP pins output a constant Hz — for scope testing |

---

### 🔩 Joints — Hardware Parameters

One row per joint. Pre-populated with authoritative values from the legacy STM32 firmware.

| Column | Notes |
|:--|:--|
| STEP / DIR / Enc Pin | Teensy 4.1 GPIO. Don't change unless you rewired the board. |
| Gear Ratio | From legacy `motor_init.cpp`. Must match physical hardware. |
| Dir Inv | Check this if the joint moves backward. |
| Limit Type | NONE = limit switch ignored. Set to NPN/PNP/MECHANICAL to enable. |
| Kp | Start low (0.5–2.0). Increase after filter is verified. |

---

### 📡 Comms — Transport & Timing

| Setting | Recommended |
|:--|:--|
| Transport | **USB_CDC_HS** for anything ≥ 50 Hz. UART for basic testing. |
| ROS Command Rate | Start at 25 Hz; raise to 100 Hz once stable. |
| Control Loop Rate | 1000 Hz (1 ms ISR). Lower only if running out of ISR budget. |
| Command Timeout | 200 ms — motor stops if no waypoint received within this window. |

---

### 🕹 Jog — Manual Movement

- **Hold** `+ ▶` or `◀ −` to move a joint at the set velocity.
- **Release** → automatically sends zero velocity.
- **ON/OFF toggle** per joint (disables that motor instantly).
- **🛑 STOP ALL** — sends zero velocity to all joints immediately.
- **ISR bar** at top: green = under budget, red = over 25 µs (investigate).

> ⚠️ Only use Jog with control loop enabled (Phase 5+). Do NOT jog in encoder-test-only modes (Phase 0–2).

---

### ⚡ Flash — Build & Flash

| Button | What it does |
|:--|:--|
| **⚙️ Generate config.h** | Writes `generated/config.h` from GUI settings (preview shown below) |
| **🔨 Build Only** | `pio run` — compile check without flashing |
| **⚡ Generate & Flash** | Generate config.h → `pio run --upload` |
| **⚡ Flash Only** | `pio run --upload` **without** regenerating config.h — use for diagnostic sketches or when config is already correct |
| **✖ Abort** | Kill the running build/flash process |

The build log highlights errors in red and success in green.

---

### 💬 Serial — Monitor

- **Port / Baud**: selected from **status bar** (always visible, any tab).
- **Filter box**: type `ACK` to show only telemetry, or `FAULT` to watch for errors.
- **Timestamps**: elapsed seconds from connection start.
- Any line containing `FAULT` or `ERR` is highlighted yellow automatically.

---

### 📈 Oscilloscope — Live Telemetry

Four real-time panels updated at 20 Hz:

| Panel | Channels | Notes |
|:--|:--|:--|
| **Joint Positions (rad)** | J1–J6 | Encoder-derived angle = closed-loop feedback |
| **Joint Velocities (rad/s)** | J1–J6 | AlphaBeta-filtered rate |
| **PWM Output (%)** | J1–J6 | Motor command effort — tick `J1 pwm` checkbox |
| **ISR Time (µs)** | — | Red dashed line = 25 µs budget |

Enable/disable any channel with the checkboxes in the toolbar row above the plots.

**ACK packet formats supported (firmware side):**

```
12 fields: <ACK,seq, p0..p5, v0..v5>
13 fields: <ACK,seq, p0..p5, v0..v5, isr_us>
18 fields: <ACK,seq, p0..p5, v0..v5, pwm0..pwm5>        ← enables PWM panel
19 fields: <ACK,seq, p0..p5, v0..v5, pwm0..pwm5, isr_us>
```

---

### ⚠️ Faults — History

Automatically captures every `FAULT`/`SOFT_ESTOP` line from serial. Each row records:
- Time of fault
- Supervisor state
- Joint velocities at the moment of fault
- ISR execution time

Export as CSV for thesis documentation.

---

## Saving & Loading Configs

- **Save** (`Ctrl+S` style or toolbar): saves to `saved_configs/<name>.json`.
- **Open**: load any `.json` from `saved_configs/`.
- **Protocol tab presets**: 11 pre-built configs for phases P0–P10 are ready to use.
- **Autosave**: the last session is restored automatically on next launch.

Config files are plain JSON — you can diff them with `git diff` to track what changed between sessions.

---

## Connecting to the Teensy

The **second toolbar row** (below File/Save buttons) contains the serial picker:

```
┌─ Toolbar 1: New | Open | Save | Save As | [Profile name] ──────────────────────┐
├─ Toolbar 2: 🔌 Port: [/dev/ttyACM1      ] [� Scan]  Baud: [115200]  [⚡ Connect]─┤
```

1. Plug in Teensy via USB → Port field auto-fills with the first detected port.
2. If port is wrong, click **🔍 Scan** to pick from a list, or just **type** the path directly (e.g. `/dev/ttyACM1`).
3. Click **⚡ Connect** — status bar turns **🟢 Connected**.

---

## File Structure

```
parol6_firmware_configurator/
├── main.py                  ← Run this
├── requirements.txt
├── core/
│   ├── config_model.py      ← All settings as Python dataclasses (JSON)
│   ├── code_generator.py    ← Writes generated/config.h
│   ├── flash_manager.py     ← PlatformIO subprocess wrapper
│   └── serial_monitor.py    ← Serial port reader thread
├── tabs/                    ← One file per GUI tab
├── docs/
│   └── TESTING_PROTOCOL.md  ← Detailed 10-phase hardware bring-up guide
├── saved_configs/           ← JSON profiles (phase0–phase10 pre-loaded)
└── generated/
    └── config.h             ← Auto-generated — #include in firmware main.cpp
```

---

## Firmware Integration (for the developer)

The firmware needs to `#include "generated/config.h"` and wrap features:

```c
// In firmware main.cpp / constants.h:
#include "generated/config.h"

#if FEATURE_ALPHABETA_FILTER
  // AlphaBeta filter code
#endif

#if FEATURE_WATCHDOG
  // Watchdog init
#endif
```

The GUI already generates the correct header format. Firmware integration is the next step.

---

## QuadTimer Diagnostic Sketch

Before connecting the real robot, verify QuadTimer PWM capture hardware works using the standalone diagnostic sketch.

**How `QuadTimerEncoder` works:** It uses Teensy 4.1 hardware QuadTimers in "Gated Count Mode" to measure PWM duty cycle (not A+B quadrature). It counts IP Bus clock pulses ONLY while the encoder pin is HIGH, then divides by expected total pulses → duty ∈ [0, 1] → angle ∈ [0, 2π rad].

**Encoder pin map** — ESP32 must connect to exactly these pins:

| Joint | Teensy Pin | QuadTimer |
|:--|:--|:--|
| J1 | **Pin 10** | TMR1 CH0 |
| J2 | **Pin 11** | TMR1 CH2 |
| J3 | **Pin 12** | TMR1 CH1 |
| J4 | **Pin 14** | TMR3 CH2 |
| J5 | **Pin 15** | TMR3 CH3 |
| J6 | **Pin 18** | TMR3 CH1 |

> Any other pin: `init()` silently returns, counter stays at zero.

**Recommended ESP32 PWM:** 1000 Hz, 10–90% duty cycle (matches AS5048A encoder spec).

**Flash the diagnostic:**

```bash
# Option A: script
cd parol6_firmware/diagnostic/quadtimer_diagnostic
./flash_diagnostic.sh

# Option B: GUI Flash tab
# Firmware dir → .../diagnostic/quadtimer_diagnostic/
# Click ⚡ Flash Only
```

**Monitor output:**
```bash
# Inside docker container:
python3 -m serial.tools.miniterm /dev/ttyACM1 115200
```

Expected output with ESP32 PWM ~50% on pin 10:
```
[ 1.0s]  J1=0.5001(3.14rad)  J2=0.0000  ...
         *** J1 PASS  duty=0.5001  angle=3.142 rad ***
```

---

## Troubleshooting

| Problem | Fix |
|:--|:--|
| `ImportError: attempted relative import` | Run from `parol6_firmware_configurator/` dir: `python3 main.py` |
| Port field shows `ttyS31` only | Inside Docker, Teensy USB not mapped. Check `ls /dev/ttyACM*` in container. |
| Port dropdown doesn't open | Known X11/Docker issue — fixed: port is now a text field. Type path directly. |
| `pio not found` | Install PlatformIO: `pip install platformio` |
| Black window / no display | X11 not forwarded to Docker. Add `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` |
| QuadTimer reads all zeros | Wrong Teensy pin, wrong ENCODER_MODE, or PWM freq too low/high. Run diagnostic sketch. |
| ISR bar always red | Control loop rate too high for your tasks. Drop to 500 Hz in Comms tab. |
