# PAROL6 Serial Communication Protocol

## Transport

| Board | Interface | Speed |
|-------|-----------|-------|
| Teensy 4.1 | USB CDC (native) | 480 Mbps |
| STM32 BlackPill | USB CDC (full-speed) | 12 Mbps |
| Both (fallback) | UART via `TRANSPORT_MODE=0` | 115200 baud |

Baud rate is ignored for USB-CDC. The `115200` setting in the console port picker is a no-op for BlackPill/Teensy but must be set for bare UART connections.

---

## Host → Firmware (Commands)

All commands are ASCII, wrapped in `< >`, terminated with `\n`.

### Position + Velocity Command (normal operation)
```
<SEQ,p0,p1,p2,p3,p4,p5,v0,v1,v2,v3,v4,v5>
```
- `SEQ` — monotonically increasing uint32 sequence number
- `p0..p5` — joint positions in radians (joint-space, after gear ratio)
- `v0..v5` — joint velocity feedforward in rad/s

### Special Commands

| Command | Effect |
|---------|--------|
| `<HOME>` | Start full 6-joint homing sequence |
| `<HOME1>` .. `<HOME6>` | Home individual joint |
| `<ENABLE>` | Clear SOFT_ESTOP, re-arm safety supervisor |
| `<DISABLE>` | Disable all motor outputs |
| `<RESET>` | Soft reset (firmware-defined) |
| `<REBOOT_DFU>` | Write DFU magic to RTC backup register, call `NVIC_SystemReset()` → board enters DFU bootloader (**STM32 only**) |

---

## Firmware → Host (Feedback / ACK)

### FORMAT A — `parol6_firmware` (Teensy 4.1 / blackpill_f411ce env)

```
<ACK,seq, p0,p1,p2,p3,p4,p5, v0,v1,v2,v3,v4,v5, lim_state, state_byte>
```

| Field | Type | Meaning |
|-------|------|---------|
| `seq` | uint32 | Echo of the last received command seq |
| `p0..p5` | float (4 dp) | Actual joint positions in radians |
| `v0..v5` | float (4 dp) | Actual joint velocities in rad/s |
| `lim_state` | uint8 bitmask | Bit N set = joint N+1 limit switch triggered |
| `state_byte` | uint8 | Supervisor state (see table below) |

**`state_byte` values:**

| Value | Label | Meaning |
|-------|-------|---------|
| 0 | IDLE | No active command |
| 1 | RUNNING | Executing trajectory |
| 2 | HOMING | Homing sequence active |
| 3 | FAULT | Latched fault — requires reflash or restart |
| 4 | ESTOP | Soft ESTOP — send `<ENABLE>` to recover |
| 5 | DISABLED | Outputs disabled |

### FORMAT B — `realtime_servo_blackpill` (Arduino sketch)

```
<ACK,seq, p0,v0, p1,v1, p2,v2, p3,v3, p4,v4, p5,v5>
```

- Exactly 12 data fields after `seq` — interleaved position and velocity per joint
- No `lim_state`, no `state_byte`, no ISR timing field
- Sent at **50 Hz** (not every control tick)

### Auto-Detection in Console

The `serial_monitor.py` parser detects format automatically:
- **12 fields** → interleaved (FORMAT B)
- **≥13 fields** → flat (FORMAT A)

---

## Unsolicited Messages (Firmware → Host)

These are plain text strings (not `<ACK>` frames) logged to the Serial tab:

| Message | Meaning |
|---------|---------|
| `INIT_OK\n` | Firmware started successfully |
| `HOMING_DONE\n` | Homing sequence completed |
| `HOMING_FAULT\n` | Homing failed (sensor not triggered within travel limit) |
| `REBOOTING_TO_DFU\n` | Received `<REBOOT_DFU>`, entering bootloader |
| `STALE_COMMAND\n` | Last command arrived >20 ms late — supervisor warning |

---

## Timing

| Parameter | parol6_firmware | realtime_servo_blackpill |
|-----------|----------------|--------------------------|
| Control loop | 1 kHz (1000 µs) | 500 Hz (2000 µs) |
| ACK rate | Every control tick | 50 Hz (20 ms) |
| Host command rate | Up to 100 Hz (ROS) | Any rate |
| Max step frequency | 20 kHz | 20 kHz |

---

## Console Packet Rate Display

The status bar shows `PKT/s` and `KB/s`:
- parol6_firmware at 1 kHz → ~1000 pkts/s, ~60–80 KB/s
- realtime_servo_blackpill at 50 Hz → ~50 pkts/s, ~3 KB/s
