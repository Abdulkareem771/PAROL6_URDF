# PAROL6 Testing Console

A hardware-agnostic runtime testing console for the PAROL6 robot arm.  
Supports **Teensy 4.1** (production) and **STM32F411CE BlackPill** (development/STM32 bringup).  
Connects via USB-CDC serial, not Docker — runs directly on the host.

---

## Quick Start

```bash
cd parol6_testing_console
pip install -r requirements.txt
python3 main.py
```

Select your project from the top-left dropdown, set the serial port, and click **⚡ Connect**.

---

## Tab Reference

| Tab | Purpose |
|-----|---------|
| **💬 Serial** | Raw terminal. Firmware state badge (IDLE/RUNNING/FAULT), coloured messages, command history (↑↓), macro buttons, ACK spam suppressed by default |
| **📈 Plot** | Live position + velocity oscilloscope via pyqtgraph |
| **🕹 Jog** | Per-joint velocity jog (hold button), degree slider, absolute Go, per-joint home |
| **⚠️ Faults** | Safety supervisor event log — timestamps, joint velocities at fault, CSV export |
| **⚡ Flash** | PlatformIO Build & Upload. USB/DFU status panel (auto-polls every 2 s). Auto-detach after successful DFU flash |
| **🚀 Launch** | ROS 2 action orchestration (Fake / Gazebo / Real Hardware) |
| **📖 Docs** | In-app documentation |

---

## Projects (`project_registry.json`)

Each project entry declares capabilities that drive which features appear:

| Capability | Effect |
|---|---|
| `supports_dfu_reboot` | Adds `<REBOOT_DFU>` macro + Flash tab auto-sends it before uploading |
| `supports_dtr_reboot` | Adds **HW Reboot (DTR)** button to Serial tab |
| `supports_jog` | Shows Jog tab |
| `supports_fault_log` | Shows Fault Log tab |
| `supports_ack_plot` | Shows Plot tab |

### Supported Projects

| ID | Board | Macros |
|---|---|---|
| `stm32_blackpill` | BlackPill F411CE | `<ENABLE>` `<DISABLE>` `<HOME>` `<RESET>` `<REBOOT_DFU>` |
| `teensy41_parol6` | Teensy 4.1 | `<ENABLE>` `<DISABLE>` `<HOME>` |

---

## STM32 / BlackPill Flash Workflow

### First flash (from DFU bootloader)
1. Hold **BOOT0**, press+release **NRST**, release **BOOT0**
2. `lsusb` shows `0483:df11` — DFU mode confirmed
3. Flash tab: **⚡ Flash via DFU** (or `🚀 Build & Upload`)
4. Console sends `dfu-util -e` automatically after upload — board reboots into firmware

### Subsequent flashes (software DFU reboot)
1. Click **`<REBOOT_DFU>`** macro in Serial tab **or** just click **🚀 Build & Upload**
2. Flash tab detects `0483:df11`, uploads, auto-detaches

See [`docs/STM32_BRINGUP.md`](docs/STM32_BRINGUP.md) for full hardware details.

---

## Serial Protocol

Two ACK formats are supported automatically:

| Firmware | Format | Fields |
|---|---|---|
| `parol6_firmware` (Teensy/STM32) | Flat | `<ACK,seq, p0..p5, v0..v5, lim_state, state_byte>` |
| `realtime_servo_blackpill` | Interleaved | `<ACK,seq, p0,v0, p1,v1, ...>` |

See [`docs/SERIAL_PROTOCOL.md`](docs/SERIAL_PROTOCOL.md) for complete protocol reference.

---

## Architecture

```
main.py  (MainWindow)
  │
  ├─ project_registry.json    ← capability declarations per hardware target
  ├─ core/serial_monitor.py   ← QThread: reads serial, emits telemetry dict
  ├─ core/process_workers.py  ← QThread: runs pio / ros2 launch subprocesses
  ├─ core/diagnostics.py      ← checks pio, dfu-util, openocd on PATH
  │
  ├─ tabs/serial_tab.py       ← firmware state badge, command history, macros
  ├─ tabs/plot_tab.py         ← pyqtgraph live oscilloscope
  ├─ tabs/jog_tab.py          ← per-joint velocity jog + encoder readout
  ├─ tabs/flash_tab.py        ← PlatformIO orchestration + DFU probe panel
  ├─ tabs/fault_log_tab.py    ← supervisor ESTOP event table
  └─ tabs/launch_tab.py       ← ROS 2 action launcher
```
