# Experimental PAROL6 Firmware

This is a stripped-down Teensy 4.1 PlatformIO project intended to debug the ROS-to-firmware path separately from the main configurator firmware.

Protocol:

- Commands: `<SEQ,p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6>`
- Enable: `<ENABLE>`
- Disable: `<DISABLE>` or `<STOP>`
- Zero current pose: `<HOME>` or `<ZERO>`
- Feedback: `<ACK,seq,p1..p6,v1..v6,lim_state,state>`

Design notes:

- It speaks the packet format expected by `parol6_system.cpp`.
- It uses a normal-loop proportional controller, not the current ISR-heavy stack.
- It unwraps the MT6816 encoder angle into a continuous position estimate.
- It is intended for debugging communication and basic motion first, not as the final firmware.

Flash:

```bash
cd parol6_experimental_firmware
pio run --environment teensy41 --target upload
```
