# Shaky Movement - "Fighting the MKS" Issue

## Problem
Motor moves but with noticeable shaking/jerky motion. This happens because:

**ESP32 sends step/dir commands → MKS board tries to follow → MKS also runs its own closed-loop control**

Result: Two control loops fighting each other!

---

## Why This Happens

The MKS SERVO42C runs in **CR_OPEN mode** (step/dir input), but internally it still:
1. Counts steps from ESP32
2. Reads its encoder
3. Applies **its own** position correction
4. Drives the motor

When ESP32 also reads the encoder and adjusts velocity, both systems fight for control.

---

## Solutions

### Option 1: Disable MKS Internal Closed-Loop (Recommended)

Configure MKS boards to **pure step/dir mode** without internal correction:

**Check MKS settings**:
- Look for **"Open Loop" vs "Closed Loop"** setting
- Set to **Open Loop** if available
- This makes MKS a "dumb" stepper driver that just follows step/dir

**How**: Usually via USB configuration tool or DIP switches (see MKS manual)

### Option 2: Let MKS Handle Closed-Loop (Easier)

**Go back to old firmware** (`stepdir_velocity_control.ino` without encoder):
- ESP32 sends step/dir based on ROS commands
- MKS handles all closed-loop control internally
- Encoder used only by MKS, not ESP32

**Trade-off**: No encoder feedback to ROS, may still have snap issue

### Option 3: Disable MKS Step Correction

Some MKS boards have a parameter to **reduce or disable step error correction**:
- Keeps encoder for MKS torque control
- Reduces position correction aggressiveness
- ESP32 velocity control becomes primary

**Check MKS manual** for parameters like:
- `Step Error Threshold`
- `Position Correction Gain`
- `Step Tracking Mode`

---

## Testing Current Setup

With the corrected gearbox settings, try this motion:

1. **Upload fixed firmware** (J1=20:1, microsteps=4)
2. **Restart ROS** and test trajectory
3. **Check**:
   - Does motion complete without timeout?
   - Is shaking reduced?
   - Does encoder feedback match MoveIt simulation?

The gearbox fix should help significantly!

---

## Expected Behavior After Fix

**Before** (wrong gearbox):
- Motor spins 20× faster than expected
- Position feedback 20× off
- Timeout due to slow perceived motion

**After** (correct gearbox):
- Motor speed matches simulation  
- Position feedback accurate
- ✅ Should complete trajectories

The shaking may persist if MKS closed-loop is too aggressive, but motion should work!

---

## Next Steps

1. **Test with corrected firmware** - see if gearbox fix solves timeout
2. **If still shaky**: Check MKS settings for open-loop mode
3. **If can't disable MKS loop**: Consider using old firmware without ESP32 encoder reading

The encoder feedback approach is theoretically better, but only if MKS cooperates!
