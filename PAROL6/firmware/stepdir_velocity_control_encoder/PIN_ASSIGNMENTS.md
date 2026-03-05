# ESP32 Pin Assignments for PAROL6 Encoder Feedback

## Quick Reference

### J1 (Motor 1)
- **Step**: GPIO 5
- **Dir**: GPIO 2
- **Encoder PWM**: GPIO 27 ✅ (already connected)

### J2 (Motor 2) - 20:1 Gearbox
- **Step**: GPIO 25
- **Dir**: GPIO 26
- **Encoder PWM**: GPIO 33 🔧 (to wire)

### J3 (Motor 3)
- **Step**: GPIO 14
- **Dir**: GPIO 27
- **Encoder PWM**: GPIO 25 🔧 (to wire)

### J4 (Motor 4)
- **Step**: GPIO 12
- **Dir**: GPIO 4
- **Encoder PWM**: GPIO 26 🔧 (to wire)

### J5 (Motor 5)
- **Step**: GPIO 13
- **Dir**: GPIO 16
- **Encoder PWM**: GPIO 34 🔧 (to wire, input-only)

### J6 (Motor 6)
- **Step**: GPIO 15
- **Dir**: GPIO 17
- **Encoder PWM**: GPIO 35 🔧 (to wire, input-only)

---

## Wiring Instructions

### For Each Motor (J2-J6):

1. **Locate PIN 3 on MKS SERVO42C board** (PWM output from MT6816 encoder)
2. **Connect to corresponding ESP32 GPIO**:
   - Use jumper wire: MKS PIN 3 → ESP32 GPIO
   - Ground reference: MKS GND → ESP32 GND (should already be common)
3. **Test connection**:
   - Upload firmware
   - Check Serial Monitor for encoder readings
   - Manually rotate motor shaft
   - Verify position changes smoothly

### Notes

- **GPIO 34 & 35**: Input-only pins (no pull-up resistors), perfect for reading PWM
- **Common ground**: Ensure ESP32 and all MKS boards share common ground
- **Signal quality**: If noisy, add 100Ω series resistor near ESP32 input

---

## Code Arrays (Reference)

```cpp
// These are configured in the firmware
const int STEP_PINS[6] = {5, 25, 14, 12, 13, 15};
const int DIR_PINS[6] = {2, 26, 27, 4, 16, 17};
const int ENCODER_PINS[6] = {27, 33, 25, 26, 34, 35};
```

---

## Testing Order

1. ✅ **J1 first** (already wired) - verify encoder reading works
2. 🔧 **J2 next** (gearbox) - test multi-turn tracking
3. 🔧 **J3-J6** (direct drive) - complete all motors
