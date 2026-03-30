# MoveIt Controller Guide

This document describes the `moveit_controller` node used by the PAROL6 vision pipeline to execute a generated weld path through MoveIt2.

It is the execution-side companion to:

- `PATH_GENERATOR_GUIDE.md`
- `OMPL_MOVEIT_PLANNER.md`
- `MOVEIT_CONTROLLER_DEBUG_CONTEXT.md`

---

## 1. Purpose

`moveit_controller` consumes the final latched path from `/vision/welding_path`, plans a safe sequence through MoveIt, and executes the result in three stages:

1. home
2. approach
3. weld

The node is designed for a planar welding setup where the torch usually uses one primary downward orientation. Because of that, the controller now treats the **approach** as a flexible positioning problem, while keeping the **weld** orientation stable.

---

## 2. Interfaces

### Topics

| Direction | Topic | Type | Notes |
|---|---|---|---|
| Subscribe | `/vision/welding_path` | `nav_msgs/Path` | `TRANSIENT_LOCAL`, receives latched path from `path_holder` |

### Services

| Service | Type | Purpose |
|---|---|---|
| `/moveit_controller/execute_welding_path` | `std_srvs/srv/Trigger` | Start a welding execution run |
| `/moveit_controller/is_execution_idle` | `std_srvs/srv/Trigger` | Report whether execution is idle and a path is ready |

### MoveIt Interfaces

| Interface | Type | Purpose |
|---|---|---|
| `/move_action` | `moveit_msgs/action/MoveGroup` | Home and approach planning |
| `/compute_cartesian_path` | `moveit_msgs/srv/GetCartesianPath` | Weld path planning |
| `/execute_trajectory` | `moveit_msgs/action/ExecuteTrajectory` | Execute planned trajectories |

---

## 3. Execution Model

The controller uses a "plan everything first" workflow:

```text
service trigger
  -> deep-copy latest path snapshot
  -> worker thread starts
     -> plan home
     -> plan approach
     -> plan weld
     -> execute home
     -> execute approach
     -> execute weld
```

Two details matter here:

- The path is **snapshotted** at trigger time, so late republish events do not change the current run.
- The worker thread prevents the service callback from blocking the ROS executor while planning is in progress.

---

## 4. Approach Planning Strategy

The approach phase is where most real failures happen.

Older behavior:

- take the first weld pose
- add `approach_distance` to `z`
- plan once with an orientation constraint
- abort immediately if MoveIt fails

Current behavior:

- take the first weld pose
- try a small deterministic fallback ladder

### Fallback Ladder

For each lift distance:

1. strict orientation constraint
2. relaxed orientation constraint
3. position-only goal

Lift distances are tried in descending conservatism:

- configured `approach_distance`
- `0.10 m` if smaller than current
- `0.05 m` if smaller than current
- `0.0 m`

This means the controller now answers:

- "Can I reach the lifted pose exactly?"
- "Can I reach it if I relax orientation?"
- "Can I at least pre-position above the seam?"
- "If the lift is the problem, can I reach a smaller-lift variant?"

---

## 5. Orientation Philosophy

For the current workspace, one or two stable tool orientations are more useful than highly varying orientation profiles.

That matches both:

- the current physical setup: flat work area, downward tool
- the current path generator: fixed downward quaternion for welding

So the controller does **not** introduce new fancy orientation generation. Instead, it preserves the weld orientation and only relaxes orientation constraints during the approach phase when OMPL needs help.

This keeps behavior predictable:

- weld motion keeps the intended torch pose
- approach motion is allowed to be more forgiving

---

## 6. Common Failure Modes

### 6.1 Approach fails with `MoveItErrorCodes(val=99999)`

This is a generic MoveIt planning failure.

Most useful questions:

- Did `strict` fail but `position_only` succeed?
  Then orientation constraint is the issue.
- Did all lift values fail?
  Then the point itself is likely out of reach or near a singular zone.
- Did the first weld point arrive near the edge of the workspace?
  Then TF or projection is still suspect.

### 6.2 `Received path with ... points` during execution

That can happen because `path_holder` republishes or the GUI refreshes the cached path.

This is no longer supposed to perturb the active run because execution uses a copied snapshot.

### 6.3 Cartesian planning fails after approach succeeds

Then the problem has moved from "goal reachability" to "path-following feasibility". In that case:

- inspect Cartesian success fractions
- reduce waypoint density
- rely on joint-waypoint fallback if needed

---

## 7. Useful Logs

The controller now emits approach diagnostics like:

```text
Approach attempt 1: lift=0.150m mode=strict quat=(...)
Approach attempt 2: lift=0.150m mode=relaxed quat=(...)
Approach attempt 3: lift=0.150m mode=position_only quat=(...)
```

These lines tell you whether the failure is:

- lift-related
- orientation-related
- fully reachability-related

---

## 8. Recommended Tuning Order

When Method 3 fails:

1. verify there is only one `moveit_controller`
2. verify the first weld pose is reasonable in `base_link`
3. inspect the approach attempt ladder
4. only after that adjust `approach_distance`
5. only after that revisit weld path density

This prevents mixing path-generation issues with approach-planning issues.

---

## 9. Related Files

- `parol6_vision/parol6_vision/moveit_controller.py`
- `parol6_vision/test/test_moveit_controller.py`
- `parol6_vision/docs/OMPL_MOVEIT_PLANNER.md`
- `parol6_vision/docs/MOVEIT_CONTROLLER_DEBUG_CONTEXT.md`
- `parol6_vision/docs/STABILIZING_PAROL6_VISION_PIPELINE.md`
