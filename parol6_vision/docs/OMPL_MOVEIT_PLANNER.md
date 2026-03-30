# OMPL / MoveIt2 Motion Planning — PAROL6 Reference

This document covers how OMPL (Open Motion Planning Library) is configured and used in the PAROL6 welding pipeline via MoveIt2, including key parameters, common failure modes, and tuning guidance.

---

## 1. Overview

```
moveit_controller.py
        │
        ├─ plan_to_home()            → MoveGroup action (joint-space)
        ├─ plan_approach_with_fallback()
        │      └─ plan_pose()        → MoveGroup action (approach retries)
        └─ plan_cartesian_with_fallback()
                ├─ compute_cartesian_path  → Cartesian arc (inline waypoints)
                └─ fallback: plan each waypoint as joint-space goal
```

All planning uses the **`parol6_arm`** planning group and the **`RRTConnect`** algorithm (OMPL default). Plans are pre-computed for all phases before any execution begins — this eliminates the mid-motion pause that occurred when the weld Cartesian path was planned after the approach move had already finished.

---

## 2. Key Parameters (`moveit_controller` node)

| Parameter | Default | Description |
|---|---|---|
| `planning_group` | `parol6_arm` | MoveIt planning group name (must match SRDF) |
| `base_frame` | `base_link` | Reference frame for all target poses |
| `end_effector_link` | `tcp_link` | EE link for IK and Cartesian planning |
| `approach_distance` | `0.15` m | Height above first weld waypoint for approach pose. **Critical: must be high enough to keep approach z inside workspace bounds** |
| `weld_velocity` | `0.01` m/s | Cartesian velocity scaling for weld phase |
| `cartesian_step_sizes` | `[0.005, 0.01, 0.02]` | Eef step sizes tried in sequence for Cartesian path planning (5mm → 1cm → 2cm) |
| `min_success_rates` | `[0.95, 0.95, 0.90]` | Minimum fraction of waypoints the Cartesian planner must reach per step size |
| `move_group_wait_timeout_sec` | `30.0` s | How long to wait for MoveGroup action server |
| `execute_wait_timeout_sec` | `20.0` s | How long to wait for trajectory execution to complete |
| `enable_joint_waypoint_fallback` | `True` | Fall back to joint-space waypoint execution if Cartesian planning fails |
| `joint_waypoint_fallback_count` | `8` | Number of evenly-spaced waypoints to visit in fallback mode |
| `enforce_reachable_test_path` | `False` | Clamp path waypoints into `test_workspace_*` bounds before planning |
| `test_workspace_min` | `[0.20, -0.35, 0.10]` | Conservative workspace lower bound (x,y,z) in metres |
| `test_workspace_max` | `[0.65, 0.35, 0.55]` | Conservative workspace upper bound |
| `test_min_radius_xy` | `0.20` m | Minimum XY distance from base axis |
| `test_max_radius_xy` | `0.70` m | Maximum XY distance from base axis |

---

## 3. Three-Phase Execution Sequence

```
Phase 1 — Home
  plan_to_home() → joint-space move to [0,0,0,0,0,0]
  MoveGroup: RRTConnect, max 10s

Phase 2 — Approach
  plan_approach_with_fallback(path.poses[0], start_state=home_end_state)
    attempt A: z += approach_distance, strict orientation
    attempt B: same lift, relaxed orientation
    attempt C: same lift, position-only
    attempt D/E/F: shorter lift distances with same retry ladder
  MoveGroup: RRTConnect, max 10s per attempt

Phase 3 — Cartesian Weld
  for step_size in [0.005, 0.01, 0.02]:
      compute_cartesian_path(waypoints, eef_step=step_size)
      if fraction ≥ min_success_rate: break
  if all fail and enable_joint_waypoint_fallback:
      execute N evenly-spaced waypoints in joint-space

All phases planned BEFORE any execution starts.
Execution: Home → Approach → Weld (back-to-back, no gaps).
```

### 3.1 Approach Fallback Ladder

The approach phase is now intentionally more forgiving than the weld phase.

- The weld path keeps the path generator's primary downward welding orientation.
- The approach planner first tries that same orientation with the configured lift.
- If OMPL cannot solve it, the controller relaxes orientation tolerance.
- If that still fails, it retries as a position-only goal.
- If the lifted pose itself is the issue, it repeats the same ladder with shorter lifts, down to the first weld pose.

This matches the current physical task well: a largely planar workspace with one main welding attitude, where pre-approach does not need a richer orientation family to be useful.

---

## 4. Why OMPL Fails — Root Causes

### 4.1 `Unable to sample any valid states for goal tree`

This means the **goal pose itself is unreachable** — not that the path to it is blocked.

| Root Cause | Symptom | Fix |
|---|---|---|
| Approach z below workspace_min | Approach z = weld_z + approach_dist < 0.10 m | Increase `approach_distance` (default is now 0.15 m) |
| Camera TF wrong → waypoints outside workspace | All approach attempts fail | Fix physical camera TF in `demo.launch.py` |
| Missing `tcp_link` collision geometry | URDF warning: `no collision geometry` | Add collision box to `tcp_link` in the URDF |
| IK singularity near joint limits | Planning fails for specific orientations | Check arm is not at full extension; rotate base joint |
| Wrong planning group or EE link | Planner uses wrong chain | Verify `parol6_arm` group and `tcp_link` are correct in SRDF |

### 4.2 `Catastrophic failure` (error code 99999)

MoveGroup returned error code **99999** — this is a generic "planning failed" code from MoveIt's action server. Always appears alongside one of the OMPL-specific error messages above.

In the current controller, this is most commonly seen in the **approach phase**, not the Cartesian weld phase. The most useful next diagnostic is to inspect the logged approach attempts:

- `lift=...`
- `mode=strict`
- `mode=relaxed`
- `mode=position_only`

If only `strict` fails but `position_only` succeeds, the issue is orientation constraint tightness rather than raw reachability.

### 4.3 Cartesian path fraction too low

`compute_cartesian_path` returns a fraction < `min_success_rate`.

| Cause | Fix |
|---|---|
| Waypoints too close together (< 1mm) | Increase `waypoint_spacing` in `path_generator` |
| Waypoints enter singularity mid-path | Reduce weld seam length; offset away from robot axis |
| EE orientation forces wrist-flip at some point | Switch from fixed downward orientation to path-tangent orientation |
| `tcp_link` has no collision geometry | MoveIt can't check self-collision → may over-reject | Add URDF collision geometry |

---

## 5. Workspace Bounds for PAROL6

The PAROL6 is a 6-DOF desktop arm. Empirically measured reachable workspace:

```
          z (up)
          │  0.55 m ─────────── upper bound
          │
          │  0.10 m ─────────── lower bound
          │
base ─────┼──────────────────────── x (forward)
          0.20 m        0.65 m

XY ring: 0.20 m ≤ r ≤ 0.70 m from base axis
```

> **Key constraint**: The robot cannot reach below z=0.10 m while maintaining a vertical downward tool orientation. The weld seam is typically at z ≈ 0.045 m (object on table) — this means the robot works just **above** the surface limit. The approach must come from ≥ 0.10 m, which requires `approach_distance ≥ 0.055 m`. The default of 0.15 m provides comfortable margin.

---

## 6. OMPL Configuration in MoveIt

The planner is configured in:
- `parol6_moveit_config/config/ompl_planning.yaml`
- `parol6_moveit_config/config/kinematics.yaml` (IK solver settings)

Key OMPL settings used by MoveIt2 for this robot:

```yaml
parol6_arm:
  default_planner_config: RRTConnect
  planner_configs:
    - RRTConnect
    - RRT
    - EST
  projection_evaluator: joints(joint_1, joint_2)
  longest_valid_segment_fraction: 0.005
```

The `RRTConnect` planner is bidirectional — it builds a tree from both the start and goal simultaneously, which is efficient for arm planning. It is sensitive to:
1. **Goal feasibility** — if the goal IK is infeasible, planning always fails
2. **Collision objects** — the planning scene must be accurate for self-collision checking

---

## 7. Quick Diagnosis Flow

```
OMPL fails
    │
    ├─ Approach attempt logs show all retries failed
    │       │
    │       ├─ Check: ros2 topic echo /vision/welding_path --once
    │       │   → inspect first pose z value
    │       │
    │       ├─ If z < 0.10 m:
    │       │   → Increase approach_distance OR fix camera TF
    │       │
    │       ├─ If strict fails but position-only succeeds:
    │       │   → Keep fixed weld orientation, but treat approach as looser
    │       │     pre-positioning rather than a full weld-pose lock
    │       │
    │       └─ If z OK:
    │           → Check tcp_link collision geometry in URDF
    │           → Check robot not at singularity (run to home first)
    │
    └─ Cartesian fraction low
            │
            ├─ Increase waypoint_spacing in path_generator (e.g. 0.01 m)
            ├─ Reduce cartesian_step_sizes[0] to 0.001 m for denser arc
            └─ Enable joint_waypoint_fallback as safety net
```

---

## 8. Tuning for Real Hardware

When transitioning from fake hardware to real hardware, tune these parameters:

| Parameter | Fake HW recommendation | Real HW recommendation |
|---|---|---|
| `approach_distance` | 0.15 m | 0.15–0.20 m (more margin for real kinematic errors) |
| `weld_velocity` | 0.01 m/s | 0.005–0.01 m/s (slower for safety) |
| `cartesian_step_sizes` | `[0.005, 0.01, 0.02]` | `[0.002, 0.005, 0.01]` (finer steps for accuracy) |
| `min_success_rates` | `[0.95, 0.95, 0.90]` | `[0.98, 0.95, 0.90]` (higher threshold for real weld quality) |
| `execute_wait_timeout_sec` | 20.0 s | 60.0 s (real trajectory takes longer) |
| `enforce_reachable_test_path` | `False` | `False` (use real camera path; clamp only for debugging) |
