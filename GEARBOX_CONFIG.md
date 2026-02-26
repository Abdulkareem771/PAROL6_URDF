# Gearbox Configuration Guide

## Current Setup
- **J1 (joint_L1)**: 20:1 gearbox
- **J2 (joint_L2)**: 20:1 gearbox
- **J3-J6**: Direct drive (1:1)

## Changes Made

### 1. Firmware (ESP32)
**File**: `stepdir_velocity_control.ino`
```cpp
const float GEAR_RATIOS[NUM_MOTORS] = {
  20.0,  // J1: 20:1 gearbox
  20.0,  // J2: 20:1 gearbox
  1.0,   // J3-J6: Direct drive
  ...
};
```

### 2. Joint Velocity Limits (MoveIt)
**File**: `joint_limits.yaml`
```yaml
joint_L1:
  max_velocity: 1.5  # Reduced from 6.0 for 20:1 gearbox
  max_acceleration: 2.0  # Reduced from 4.0
joint_L2:
  max_velocity: 1.5  # Reduced from 6.0 for 20:1 gearbox
  max_acceleration: 2.0  # Reduced from 4.0
```

### 3. Controller Timeout
**File**: `parol6_controllers.yaml`
```yaml
goal_time: 60.0  # Increased from 20s for geared joints
```

## Why These Changes?

### Step Rate Limit
ESP32 can generate max **20,000 steps/second** (20kHz).

With 20:1 gearbox:
- Steps per joint revolution: `3200 × 20 = 64,000 steps`
- Max safe joint velocity: `20,000 / (64,000 / 2π) ≈ 1.96 rad/s`
- **Set to 1.5 rad/s** for safety margin

### Calculation for Other Gearbox Ratios

Formula:
```
max_velocity = (20,000 × 2π) / (3200 × gear_ratio)
max_velocity ≈ 39.27 / gear_ratio
```

Examples:
- 10:1 gearbox → max 3.9 rad/s
- 20:1 gearbox → max 1.96 rad/s (use 1.5)
- 50:1 gearbox → max 0.79 rad/s (use 0.7)
- 100:1 gearbox → max 0.39 rad/s (use 0.3)

## Testing

After rebuilding (`./start_real_robot.sh`), you should see:
- ✅ No timeouts
- ✅ Smooth motion (no vibration)
- ✅ No packet loss
- ✅ Reaches target position accurately

## Troubleshooting

### Still vibrating?
- Reduce `max_velocity` further (try 1.0 rad/s)
- Check MKS driver acceleration settings

### Still timing out?
- Increase `goal_time` in `parol6_controllers.yaml`
- Check if motor is actually moving

### Packet loss?
- Step rate still too high
- Reduce `max_velocity` more
