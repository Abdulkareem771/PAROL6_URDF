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

1. Set **Firmware dir** to `parol6_firmware/` (pre-filled).
2. Click **⚙️ Generate config.h** to preview the header.
3. Click **🔨 Build Only** to check for compile errors without connecting Teensy.
4. Connect Teensy via USB, then click **⚡ Generate & Flash**.

The build log highlights errors in red and success in green.

---

### 💬 Serial — Monitor

- **Port / Baud**: selected from **status bar** (always visible, any tab).
- **Filter box**: type `ACK` to show only telemetry, or `FAULT` to watch for errors.
- **Timestamps**: elapsed seconds from connection start.
- Any line containing `FAULT` or `ERR` is highlighted yellow automatically.

---

### 📈 Oscilloscope — Live Telemetry

- Requires `<ACK,seq,p0,p1,...,p5,v0,...,v5[,isr_us]>` packets from firmware.
- Show/hide individual channels with checkboxes at the top.
- **Window**: scroll back up to 60 s of data.
- **ISR panel**: red dashed line at 25 µs budget — stay below it.

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

The **status bar** at the bottom of every tab shows:

```
Port: [ /dev/ttyACM0 ▼ ] [🔄]  Baud: [ 115200 ▼ ]  [ ⚡ Connect ]
```

1. Plug in the Teensy via USB.
2. Click **🔄** to refresh the port list (or it auto-fills on startup).
3. Select `/dev/ttyACM0` (or `/dev/ttyUSB0` for UART adapter).
4. Click **⚡ Connect** — the indicator turns **🟢 Connected**.

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

## Troubleshooting

| Problem | Fix |
|:--|:--|
| `ImportError: attempted relative import` | Run from `parol6_firmware_configurator/` dir: `python3 main.py` |
| `(no ports found)` in dropdown | Teensy not plugged in, or missing USB permissions. Try `sudo usermod -aG dialout $USER` |
| `pio not found` | Install PlatformIO: `pip install platformio` |
| Black window / no display | X11 not forwarded to Docker. Add `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` |
| ISR bar always red | Control loop rate too high for your tasks. Drop to 500 Hz in Comms tab. |
