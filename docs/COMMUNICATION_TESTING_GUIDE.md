# Serial Communication Testing Guide

**Purpose**: Verify Teensy 4.1 ↔ host serial link **before** connecting motors, using the Firmware Configurator GUI and raw serial commands.

---

## What we're testing

| Test | Pass criterion |
|------|---------------|
| Link up | Feedback packets appear at ≥ 20 Hz after `<ENABLE>` |
| Packet loss | 0 lost in 1000 packets at 25 Hz |
| Latency | < 10 ms round-trip (command sent → ACK observed) |
| Sequence integrity | `STALE_CMD` never appears during normal operation |
| Limit state | `lim_state` bit changes correctly when switch is triggered |
| Jitter | Packet interval std-dev < 2 ms |

---

## Hardware setup

- Teensy 4.1 plugged into host USB (`/dev/ttyACM0` — check with `ls /dev/ttyACM*`)
- Firmware flashed with `FEEDBACK_RATE_HZ = 25` and `ROS_COMMAND_RATE_HZ = 25`
- **No motors need to be powered** for this test

---

## Step 1 — Open the serial monitor

Launch the Firmware Configurator:
```bash
./scripts/launchers/launch_configurator.sh
```

Go to **🔌 Serial** tab → select `/dev/ttyACM0` → **Connect**.

---

## Step 2 — Trigger feedback

Send in the command box:
```
<ENABLE>
```

Feedback packets should appear within 1 second:
```
<ACK,0,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0>
```

Format: `<ACK,seq,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6,lim_state>`

**Status bar** (bottom of serial tab) shows **Pkt/s** — should be ~25.

---

## Step 3 — Packet loss test (CLI)

Run this from inside the Docker container to send 1000 `<ENABLE>` pulses at 25 Hz and count ACK responses:

```bash
python3 - <<'PY'
import serial, time, re

PORT   = "/dev/ttyACM0"
BAUD   = 115200
N      = 1000
RATE   = 25   # Hz
PERIOD = 1.0 / RATE

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(0.5)

sent = 0
received = 0
latencies = []
ack_re = re.compile(r"<ACK,(\d+),")

for _ in range(N):
    t0 = time.monotonic()
    ser.write(b"<ENABLE>\n")
    sent += 1
    deadline = t0 + PERIOD
    while time.monotonic() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if ack_re.match(line):
            latencies.append((time.monotonic() - t0) * 1000)
            received += 1
            break

import statistics
loss = (sent - received) / sent * 100
print(f"\nSent:     {sent}")
print(f"Received: {received}")
print(f"Loss:     {loss:.2f}%")
if latencies:
    print(f"Avg lat:  {statistics.mean(latencies):.1f} ms")
    print(f"Std dev:  {statistics.stdev(latencies):.1f} ms")
    print(f"Max lat:  {max(latencies):.1f} ms")
ser.close()
PY
```

### Expected results

| Metric | Target | Acceptable | Problem |
|--------|--------|-----------|---------|
| Packet loss | 0% | < 0.1% | > 1% |
| Avg latency | < 5 ms | < 15 ms | > 50 ms |
| Std dev (jitter) | < 2 ms | < 5 ms | > 10 ms |
| Max latency | < 10 ms | < 30 ms | > 100 ms |

---

## Step 4 — Stale command test

Send a deliberately out-of-order position command with a regressed sequence number:

```bash
# Normal forward command (seq=100)
echo -ne "<100,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0>" > /dev/ttyACM0

# Stale command (seq=5, older than 100)
echo -ne "<5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0>" > /dev/ttyACM0
```

Watch the serial tab — the second command should produce:
```
STALE_CMD
```

✅ Firmware correctly rejects out-of-order commands.

---

## Step 5 — Limit state test (if switches wired)

Manually press/trigger a limit switch while watching the serial tab.  
The `lim_state` field in the feedback packet should change:

- J1 trigger: last field becomes `1` (binary `000001`)
- J2 trigger: last field becomes `2` (binary `000010`)
- J1+J2: last field becomes `3` (binary `000011`)
- etc.

Release switch → field returns to `0`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No feedback packets | Re-send `<ENABLE>`. Check baud (115200). Check port. |
| Loss > 1% | Replace USB cable. Connect directly (no hub). Close other serial monitors. |
| Latency > 50 ms | Close other apps on host. Check Docker CPU allocation (`docker stats parol6_dev`). |
| `STALE_CMD` during normal ROS operation | ROS command sequence reset — send `<ENABLE>` to resync. |
| `lim_state` stuck at non-zero | Switch wired active-high but config says active-low — correct `LIMIT_ACTIVE_HIGH` and reflash. |

---

**Last Updated**: 2026-03-11  
**Maintained by**: PAROL6 Team
