# PAROL6 Experimental Tool

Minimal PyQt tool for the experimental Teensy firmware.

It reuses:

- `parol6_firmware_configurator/core/serial_monitor.py`
- `parol6_firmware_configurator/core/flash_manager.py`

Features:

- connect/disconnect to Teensy serial
- raw serial monitor
- send `ENABLE`, `DISABLE`, `ZERO`, and raw protocol frames
- view parsed `<ACK,...>` telemetry
- build or flash a PlatformIO project
- run an arbitrary script

Run:

```bash
python3 parol6_experimental_tool/main.py
```
