# J1 Gearbox Velocity Limit Fix

## Problem
MoveIt planned trajectories assuming J1 is direct drive, but J1 has a 20:1 gearbox. This caused:
- Motor needs to spin 20× more to achieve same joint movement
- Trajectories timeout because they take 20× longer than planned
- Multiple executions needed to reach target

## Root Cause
`joint_limits.yaml` defined J1 with velocity/acceleration limits for direct drive motors, not accounting for the gearbox reduction.

## Solution
Updated `/parol6_moveit_config/config/joint_limits.yaml`:

**Before**:
```yaml
joint_L1:
  max_velocity: 1.5 rad/s
  max_acceleration: 3.0 rad/s²
```

**After**:
```yaml
joint_L1:
  max_velocity: 0.075 rad/s  # 1.5 / 20
  max_acceleration: 0.15 rad/s²  # 3.0 / 20
```

## Why This Works
- **Gearbox**: Motor spins 20× for each joint radian
- **Motor limit**: ~1.5 rad/s at motor shaft
- **Joint limit**: 1.5 / 20 = **0.075 rad/s** at joint
- MoveIt now plans achievable trajectories!

## Testing
1. **Restart ROS** to load new limits
2. **Plan trajectory** - should show slower, smoother paths for J1
3. **Execute** - should complete without timeout!

---

## How to Adjust for Other Motors

If J2-J6 also have gearboxes:
1. Measure motor shaft max velocity (rad/s)
2. Divide by gear ratio to get joint velocity
3. Update `joint_limits.yaml` accordingly

For direct drive motors (J2-J6 currently), use motor limits directly!
