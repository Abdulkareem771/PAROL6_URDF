# Motor Not Moving - Troubleshooting Guide

## Issue
Motor receives step commands from firmware but doesn't physically move.

## Root Cause Analysis

From your logs, we know:
1. ✅ ROS planning works
2. ✅ Trajectory execution starts  
3. ✅ Velocity commands sent (-1.5 rad/s)
4. ✅ Encoder position reads correctly (~1.35 rad)
5. ❌ **Motor doesn't move** - position stays constant
6. ❌ Trajectory aborts due to path tolerance violation

This means firmware is likely generating step pulses, but the motor isn't responding.

---

## Possible Causes

### 1. **MKS ENABLE Signal Not Active**

The MKS SERVO42C boards have an **ENABLE** input that must be pulled LOW to allow motor movement.

**Check**:
- Is there an ENABLE pin connected on your MKS boards?
- ESP32 needs to drive ENABLE LOW before sending steps

**Fix**: Add enable pin control to firmware

### 2. **Step/Dir Pins Swapped or Wrong**

We set:
- J1 Step = GPIO 5
- J1 Dir = GPIO 2

**Verify**:
- Are these the correct pins physically?
- Use multimeter or oscilloscope to check if GPIO 5 pulses during motion

### 3. **Motor in Open-Loop Mode (Not Step/Dir)**

MKS boards can be in different modes:
- **CR_OPEN** mode: Step/Dir control ✅ (what we need)
- **CR_CLOSE** / **CR_vFOC** mode: Internal control ❌

**Check**: MKS board DIP switches or configuration

### 4. **Power Issue**

- Motor power supply connected?
- Sufficient current rating?
- Check LED indicators on MKS board

### 5. **Step Pulse Width Too Short**

Current code uses 5μs pulses. Some drivers need longer.

**Test**: Increase pulse width to 10μs

---

## Quick Tests

### Test 1: Does Old Firmware Work?

Go back to the **original velocity control firmware** (without encoder):
```
/PAROL6/firmware/stepdir_velocity_control/stepdir_velocity_control.ino
```

Upload it and test if motor moves. If YES, the problem is in the encoder firmware. If NO, it's a hardware issue.

### Test 2: Check GPIO Output

Add an LED to GPIO 5 (J1 step pin). It should blink rapidly during motion.

### Test 3: Manual Step Test

Create simple Arduino sketch that just pulses GPIO 5:
```cpp
void setup() {
  pinMode(5, OUTPUT);
  pinMode(2, OUTPUT);
  digitalWrite(2, HIGH);  // Direction
}

void loop() {
  digitalWrite(5, HIGH);
  delayMicroseconds(10);
  digitalWrite(5, LOW);
  delay(10);  // Slow enough to see
}
```

Upload this - motor should slowly rotate. If not, hardware issue.

---

## Most Likely Fix: Add ENABLE Pin Control

MKS SERVO42C typically has an ENABLE pin (active LOW).

**Firmware modification needed**:

```cpp
// Add enable pins
const int ENABLE_PINS[NUM_MOTORS] = {4, 18, 23, 5, 27, 12};  // Choose unused GPIOs

void setup() {
  // ...existing code...
  
  // Configure enable pins
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(ENABLE_PINS[i], OUTPUT);
    digitalWrite(ENABLE_PINS[i], LOW);  // Enable motors (active LOW)
  }
}
```

---

## Recommended Action Plan

1. **Test with old firmware first**  
   - See if motor moves with `stepdir_velocity_control.ino`
   - This isolates encoder vs hardware issue

2. **Check physical connections**
   - Verify GPIO 5 → MKS Step pin
   - Verify GPIO 2 → MKS Dir pin
   - **Check if ENABLE pin exists and how it's connected**

3. **Try manual step test**
   - Upload simple blink sketch to GPIO 5
   - Confirms ESP32 → motor wiring works

4. **Add ENABLE pin support**
   - If MKS boards need ENABLE, add to firmware

5. **Increase pulse width**
   - Change 5μs to 10μs or 20μs
   - Some drivers are slower

---

## Questions to Answer

1. **Does the old velocity control firmware work?** (without encoder)
2. **Do MKS boards have ENABLE pins?** Check the manual or board
3. **Are motors powered?** LED indicators on MKS boards?
4. **Can you measure GPIO 5 with multimeter/scope during motion?**

Let me know the answers and we'll fix this! The encoder integration is correct, we just need to get the motor physically moving.
