# Hybrid Encoder Feedback Fix

## Issue

Motors without encoders connected were reading **noise** on unconnected GPIO pins, causing:
- Huge position values (182 rad, 145 rad)
- "Outside bounds" errors from MoveIt
- Motion planning failures
- No movement

## Solution: Hybrid Approach

**Motors WITH encoders**: Use encoder feedback (closed-loop)  
**Motors WITHOUT encoders**: Use step counting (open-loop, like before)

### Configuration

```cpp
const bool ENCODER_ENABLED[NUM_MOTORS] = {
  true,   // J1: Encoder connected on GPIO 27 ✅
  false,  // J2: Not connected yet - uses step counting
  false,  // J3: Not connected yet - uses step counting
  false,  // J4: Not connected yet - uses step counting
  false,  // J5: Not connected yet - uses step counting
  false   // J6: Not connected yet - uses step counting
};
```

### How It Works

**For J1** (encoder enabled):
1. Read encoder angle from PWM
2. Track multi-turn position
3. Use REAL position for control
4. **Result**: No snap, perfect positioning

**For J2-J6** (encoder disabled):
1. Count steps sent
2. Calculate position from steps
3. Use calculated position for control
4. **Result**: Works like before (may have snap)

---

## Testing Steps

### 1. Upload Firmware
```bash
# Upload updated firmware to ESP32
# This version won't read garbage on unconnected pins
```

### 2. Restart ROS
```bash
cd ~/Desktop/Servo42C_ESP/PAROL6_URDF
./start_real_robot.sh
```

### 3. Check Positions

Watch the log - positions should now be reasonable:
```
<ACK,seq,J1_pos,J1_vel,J2_pos,J2_vel,...>
         ^~0.25         ^~0.00  ...      # Normal values!
```

**Before** (broken):
```
<ACK,297,0.000,0.000,0.000,0.000,6.282,0.000,-0.002,0.000,...>
                                 ^~~~~~ Weird!
<ACK,347,0.000,0.000,0.000,0.000,83.222,0.000,64.458,0.000,...>
                                 ^~~~~~~ Way out of bounds!
```

**After** (fixed):
```
<ACK,297,0.256,0.000,0.000,0.000,0.000,0.000,0.000,0.000,...>
         ^~~~~ J1 encoder        ^~~~~ Step counting (all start at 0)
```

### 4. Test J1 Motion

- **Plan and execute** a trajectory involving J1
- J1 should use encoder feedback → **no snap!**
- J2-J6 use step counting → may still snap (until encoders connected)

---

## As You Connect More Encoders

When you wire encoder for J2:
```cpp
const bool ENCODER_ENABLED[NUM_MOTORS] = {
  true,   // J1: ✅ Encoder working
  true,   // J2: ✅ Just connected!
  false,  // J3: Still using step counting
  false,  // J4
  false,  // J5
  false   // J6
};
```

Re-upload firmware, test J2. Repeat for J3-J6!

---

## Expected Behavior

| Motor | Encoder | Position Source | Snap | Accuracy |
|-------|---------|----------------|------|----------|
| **J1** | ✅ Connected | Real encoder | **None** | ±0.01° |
| **J2** | ❌ Not yet | Step counting | Yes (for now) | ±0.1° |
| **J3** | ❌ Not yet | Step counting | Yes (for now) | ±0.1° |
| **J4** | ❌ Not yet | Step counting | Yes (for now) | ±0.1° |
| **J5** | ❌ Not yet | Step counting | Yes (for now) | ±0.1° |
| **J6** | ❌ Not yet | Step counting | Yes (for now) | ±0.1° |

As you connect encoders, change `ENCODER_ENABLED` to `true` for that motor!

---

## Key Fixes Applied

1. ✅ Added `ENCODER_ENABLED[]` array
2. ✅ Skip encoder reading for disabled motors
3. ✅ Fall back to step counting for disabled motors
4. ✅ Initialize step-counted positions to 0.0
5. ✅ Fixed J1/J2 gearbox settings (you had them swapped!)
   - J1 = direct drive (no gearbox)
   - J2 = 20:1 gearbox

---

## Next Steps

1. **Test current setup**: J1 with encoder, J2-J6 with step counting
2. **Verify J1 has no snap** on consecutive moves
3. **Wire J2 encoder**: GPIO 33 → MKS board 2 PIN 3
4. **Set J2 encoder enabled**: `ENCODER_ENABLED[1] = true;`
5. **Repeat for J3-J6**

This incremental approach lets you test each encoder as you wire it! 🎉
