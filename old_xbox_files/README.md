# Old Xbox Controller Files - Archive

## ⚠️ DO NOT USE THESE FILES

These are archived versions that didn't work correctly. They are kept for reference only.

## File Descriptions

### `xbox_trajectory_controller.py`
**Problem:** Published to `/parol6_arm_controller/joint_trajectory` topic  
**Why it failed:** Robot controller uses action interface, not topic subscription  
**Result:** Robot didn't move at all

### `xbox_controller_node.py`  
**Problem:** Very early test version, incomplete implementation  
**Why it failed:** Missing proper initialization, no action client  
**Result:** Unstable, unreliable

### `start_xbox_control.sh`
**Problem:** Launched the old broken controller  
**Replaced by:** `../start_xbox_action.sh`

### `test_movement.py`
**Purpose:** Simple test script to verify robot could move  
**Status:** Worked for testing, but not for real-time control

## What to Use Instead

✅ **Current working version:** `../xbox_action_controller.py`  
✅ **Launch script:** `../start_xbox_action.sh`  
✅ **Documentation:** `../XBOX_SOLUTION.md`

## Why Keep These?

1. **Learning**: Shows evolution of the solution
2. **Reference**: Might contain useful code snippets
3. **History**: Documents what didn't work (important!)

## Migration Notes

If you need to recover anything from these files:
- Check git history: `git log --all -- xbox_*.py`
- Compare versions: `git diff old_xbox_files/xbox_trajectory_controller.py xbox_action_controller.py`

---

**Last archived:** 2025-11-30  
**Reason:** Switched to action-based interface for proper robot control
