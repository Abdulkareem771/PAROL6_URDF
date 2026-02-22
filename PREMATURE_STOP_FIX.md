# Critical Fix: Motor Stops Prematurely

## Problem
Motor moved briefly then stopped at -0.403 rad, even though trajectory should continue to goal.

**Logs show**:
```
Pos: -0.017 → -0.171 → -0.349 → -0.403 (STUCK)  
Vel: -0.782 → -1.782 → -0.940 → 0.000 (STOPPED)
```

## Root Cause
**ROS was only sending POSITION commands, not VELOCITY!**

`ros2_controllers.yaml` had:
```yaml
command_interfaces:
  - position  # ❌ Only position!
```

**Impact**:
1. ros2_control sends: position targets only
2. Firmware receives: target position (no velocity info)
3. Firmware calculates: error = target - current
4. When error becomes small → firmware stops
5. But trajectory should keep moving toward final goal!

## Fix Applied
Added velocity to command interface:
```yaml
command_interfaces:
  - position
  - velocity  # ✅ Now sends trajectory velocity!
```

## Why This Fixes It
- ros2_control now sends **both** position and velocity from trajectory
- Firmware uses velocity to keep moving toward goal
- Motor doesn't stop prematurely when approaching waypoint
- Follows full trajectory timing

## Next Steps
1. **Restart ROS** to load new config
2. **Re-upload firmware** with alpha=0.8 change
3. **Test trajectory** - should complete smoothly!

The timeout parameters and alpha=0.8 will also help, but THIS was the critical missing piece.
