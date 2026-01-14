---
name: 📡 Day 2 - Serial TX
about: Implement serial transmission to ESP32
title: '[DAY-2] Serial TX Implementation'
labels: 'phase-2, serial, esp32'
assignees: ''
---

## 🎯 Objective
Implement non-blocking serial transmission to ESP32 at 25Hz.

## ✅ Tasks
- [ ] Add serial port opening in `on_configure()`
- [ ] Implement `write()` method with command formatting
- [ ] Add sequence number tracking
- [ ] Implement timing guards (< 5ms)
- [ ] Test with ESP32 echo firmware
- [ ] Validate non-blocking behavior

## 📊 Success Criteria
- [ ] Serial port opens without crashing
- [ ] Commands sent at exactly 25Hz
- [ ] Jitter remains < 5ms
- [ ] ESP32 receives valid formatted data
- [ ] 15-minute stable operation
- [ ] Clean shutdown closes serial port

## 🔧 Implementation Files
- `include/parol6_hardware/parol6_system.hpp` - Add serial member
- `src/parol6_system.cpp` - Implement on_configure() and write()

## 📚 Documentation
- [Day 2 Plan](../parol6_hardware/DAY2_SERIAL_TX_PLAN.md)
- [Hardware Interface Guide](../parol6_hardware/HARDWARE_INTERFACE_GUIDE.md)

## ⚠️ Blockers
None - Day 1 validation complete ✅
