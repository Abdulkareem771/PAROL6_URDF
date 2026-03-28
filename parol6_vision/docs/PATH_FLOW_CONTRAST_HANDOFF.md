# Path Flow Contrast Handoff

Use this as a strict handoff prompt for a new AI coding conversation.

```text
You are taking over debugging and stabilization work for the PAROL6 vision pipeline in:

- Repo: /home/kareem/Desktop/PAROL6_URDF
- Main area: parol6_vision
- Focus: contrast between the working injected-path flow and the live path-generator flow

Primary question
- Why does the injected test path work reliably with MoveIt, while the live generated path or Stage 4 send flow may fail or behave differently?

Important conclusion up front
- The GUI inject path, the live path_generator path, and the Stage 4 "Send Path -> MoveIt" button are related, but they are not the same thing.
- Inject path and path_generator both produce a `nav_msgs/Path` on `/vision/welding_path`.
- `moveit_controller` does not care where the path came from. It only executes the latest path it has buffered from `/vision/welding_path`.
- Therefore, if inject works but live path execution does not, the likely issue is path content, path availability, or upstream pipeline state, not the final MoveIt execute service itself.

Files to inspect first
- `parol6_vision/scripts/vision_pipeline_gui.py`
- `parol6_vision/parol6_vision/path_generator.py`
- `parol6_vision/parol6_vision/moveit_controller.py`
- `parol6_vision/parol6_vision/depth_matcher.py`
- Optional context:
  - `scripts/launchers/inject_reachable_weld_path.sh`
  - `scripts/inject_path.sh`

1. GUI Inject Path flow

Location
- `parol6_vision/scripts/vision_pipeline_gui.py`
- function: `_inject_test_path()`

What it does
- Publishes a hard-coded, conservative, reachable `nav_msgs/Path` directly to `/vision/welding_path`.
- It does not depend on the live vision pipeline.

Path definition
- `frame_id = base_link`
- 5 fixed waypoints
- positions:
  - x = 0.20
  - y = -0.08, -0.04, 0.00, 0.04, 0.08
  - z = 0.33
- fixed orientation:
  - x = 0.707
  - y = 0.0
  - z = -0.707
  - w = 0.0

How it is published
- `ros2 topic pub --once`
- topic: `/vision/welding_path`
- type: `nav_msgs/msg/Path`
- QoS:
  - reliability = reliable
  - durability = transient_local

Why it works well
- The geometry is hand-authored and known conservative.
- The path is already expressed in `base_link`.
- The orientation is known-reachable for the robot.

2. Live path_generator flow

Location
- `parol6_vision/parol6_vision/path_generator.py`

Input
- `/vision/weld_lines_3d`

Output
- `/vision/welding_path`

What it does
- Builds a path from live 3D weld-line points.
- It is not hard-coded.

Path generation steps
- Select highest-confidence weld line.
- Convert points to numpy.
- Order points with PCA.
- Remove duplicates.
- Fit a B-spline.
- Resample at uniform spacing.
- Create `PoseStamped` waypoints.

Path definition characteristics
- positions come from real vision output, not fixed constants
- path header is copied from incoming `WeldLine3DArray`
- that usually means `base_link` if `depth_matcher` is configured normally

Important orientation detail
- `compute_orientation()` currently returns a fixed quaternion:
  - x = 0.7071068
  - y = 0.0
  - z = -0.7071068
  - w = 0.0
- So despite the name, the generator is not currently producing a truly varying orientation along the curve.

QoS
- publisher on `/vision/welding_path` uses:
  - reliability = reliable
  - durability = transient_local

Where the frame normally comes from
- `depth_matcher` publishes `/vision/weld_lines_3d` in `target_frame`
- default `target_frame` is `base_link`
- so the generated path should normally also be in `base_link`

Key difference from inject
- Inject path is fixed and known-good.
- Generated path depends on:
  - upstream weld-line availability
  - transform correctness
  - point quality
  - reachability of generated positions

3. Stage 4 "Send Path -> MoveIt" button

Location
- `parol6_vision/scripts/vision_pipeline_gui.py`
- function: `_send_path_to_moveit()`

What it is
- It is not a path-definition function.
- It is an orchestration/execution function.

Current behavior
- The GUI subscribes to `/vision/welding_path` and caches the latest received path.
- When the Stage 4 button is pressed:
  1. if a cached path exists, the GUI republishes that cached path to `/vision/welding_path`
  2. then it calls `/moveit_controller/execute_welding_path`
  3. if no cached path exists, it calls `/path_generator/trigger_path_generation`
  4. then it tries to republish the regenerated path
  5. then it calls `/moveit_controller/execute_welding_path`

Important design intent
- This button now tries to mimic the working inject-path pattern:
  - publish a path to `/vision/welding_path`
  - then trigger MoveIt execution

What it does NOT do
- It does not invent a new path geometry.
- It depends on a valid path already existing or being regeneratable.

4. moveit_controller behavior

Location
- `parol6_vision/parol6_vision/moveit_controller.py`

Input
- subscribes to `/vision/welding_path`

Execution service
- `/moveit_controller/execute_welding_path`

What it does
- stores the latest received path in `latest_path`
- when execution service is called, it executes `latest_path`

Critical fact
- It does not distinguish whether the path came from:
  - GUI inject path
  - path_generator
  - GUI republish of cached path

That means
- If injected path works, then the topic-to-controller-to-MoveIt chain is at least partially proven.
- If live path fails, suspicion should move toward:
  - missing path publication
  - malformed generated geometry
  - unreachable generated waypoints
  - stale or absent cached path in the GUI

Bottom-line comparison

Inject path
- hard-coded
- conservative
- known reachable
- directly published
- independent of live vision

Generated path
- computed from live weld-line data
- depends on upstream nodes and transforms
- same topic, same message type, similar orientation convention
- may fail because its geometry is bad or absent

Stage 4 send button
- not a path generator
- republish-and-execute orchestrator
- intended to behave like the inject flow, but only if a good path already exists

Most useful debugging lens
- Treat the system as 3 layers:

Layer 1: Path creation
- inject path
- path_generator path

Layer 2: Path transport
- `/vision/welding_path`

Layer 3: Path execution
- `moveit_controller`

Inference
- Because inject works, Layer 2 and Layer 3 are at least partly validated.
- That points attention back to Layer 1 for live vision failures.

High-value checks to run next
- `ros2 topic echo --once /vision/welding_path`
- `ros2 topic echo --once /vision/weld_lines_3d`
- `ros2 topic echo --once /vision/weld_lines_2d`
- `ros2 service call /path_generator/trigger_path_generation std_srvs/srv/Trigger "{}"`
- `ros2 service call /moveit_controller/execute_welding_path std_srvs/srv/Trigger "{}"`

Main question for the next debugging step
- When the user presses Stage 4, is there a valid live `/vision/welding_path` available, and if so, how do its waypoint positions compare numerically to the known-good injected path?
```

