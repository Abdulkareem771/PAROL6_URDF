# Stabilizing the PAROL6 Vision Pipeline

This document records all architectural changes made to stabilize the PAROL6 welding-path vision pipeline, including the motivation for each change, the new pipeline structure, QoS contracts, and the correct operating workflow.

---

## 1. Background — Why the Pipeline Was Unstable

The original pipeline had three core failure modes:

| Failure | Root Cause |
|---|---|
| `depth_matcher` never fired | `ApproximateTimeSynchronizer` waited for RGB + Depth + 2D-lines to share the same timestamp. Manual lines are drawn long after capture — timestamps never match |
| Markers stacked in RViz | No `DELETEALL` marker was published before new markers, so every new detection added to the existing set |
| MoveIt "No path received yet" | `path_holder` was never launched from the GUI; `path_generator`'s output went nowhere. Separately, `moveit_controller` subscribed with `TRANSIENT_LOCAL` but `path_holder` published `VOLATILE` |
| Camera TF totally wrong | Both `live_pipeline.launch.py` and `parol6_moveit_config/demo.launch.py` encoded the old camera position (x=1.2, z=0.65) instead of the real measured position |
| Must capture twice | `depth_matcher` subscribed to `captured_image_depth` with `VOLATILE` QoS, while `capture_images_node` published with `TRANSIENT_LOCAL`. DDS silently drops TRANSIENT_LOCAL messages to VOLATILE late-joining subscribers |
| Double-shutdown crash | All nodes called `rclpy.shutdown()` unconditionally on exit, causing `RCLError` when shutdown was already in progress |
| **Message storm** (10+ identical paths/s) | `path_generator` had a 0.5 s gate but `depth_matcher._on_lines` had no rate limit, so it still fired on every `weld_lines_2d` at camera rate, generating fresh `weld_lines_3d` that bypassed the generator gate — storm persisted |
| **RViz TF warning**: `cannot transform L6 → base_link` | `kinect2_bridge` published its own `kinect2_link` TF concurrently with our static TF, creating a duplicate-parent conflict. Fixed by targeting `kinect2` (bridge root) instead of `kinect2_link` in our static TF |
| **OMPL approach fails** — `Unable to sample any valid states for goal tree` | Weld surface projected at z ≈ 0.045 m. With `approach_distance=0.05` the approach pose was at z=0.095 m — just **below** the robot workspace minimum z=0.10 m. All goal samples were invalid |

---

## 2. New Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Stage 1 — Capture                         │
│                                                                 │
│  kinect2_bridge  ──►  capture_images_node                       │
│  (live RGB+Depth)          │                                    │
│                            ├──► /vision/captured_image_raw      │
│                            ├──► /vision/captured_image_depth  ◄─┤ TRANSIENT_LOCAL
│                            └──► /vision/captured_camera_info  ◄─┘ TRANSIENT_LOCAL
│                                                                 │
│                     crop_image_node                             │
│  /vision/captured_image_raw ──► /vision/captured_image_color   │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Stage 2 — Line Detection                      │
│                                                                 │
│  manual_line_node or yolo_segment                               │
│  /vision/captured_image_color ──► /vision/processing_mode/      │
│                                   annotated_image               │
│                                                                 │
│  path_optimizer                                                 │
│  annotated_image ──► /vision/weld_lines_2d                     │
│                      (header.frame_id always set to            │
│                       kinect2_rgb_optical_frame)                │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 3 — 3D Projection                      │
│                                                                 │
│  depth_matcher (cache-based, no timestamp sync)                 │
│  ┌──────────────────────────────────┐                          │
│  │ Cached on arrival (TRANSIENT_LOCAL subscribers):           │
│  │   /vision/captured_image_depth   ──► _cached_depth         │
│  │   /vision/captured_camera_info   ──► _cached_info          │
│  │                                                             │
│  │ Triggered by:                                              │
│  │   /vision/weld_lines_2d ──► synchronized_callback(         │
│  │                               lines, _cached_depth,        │
│  │                               _cached_info)                │
│  └──────────────────────────────────┘                          │
│                    │                                            │
│                    ▼                                            │
│             /vision/weld_lines_3d                              │
│                    │                                            │
│                    ▼                                            │
│  path_generator (B-spline + orientation)                       │
│             /vision/welding_path/generated                     │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 4 — Path Holding & Execution                 │
│                                                                 │
│  path_holder  (sole TRANSIENT_LOCAL publisher)                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Watches: /vision/welding_path/generated             │       │
│  │ Watches: /vision/welding_path/injected              │       │
│  │ Publishes: /vision/welding_path (TRANSIENT_LOCAL)   │       │
│  │ Service: /path_holder/set_source  → force republish │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  moveit_controller (TRANSIENT_LOCAL subscriber)                 │
│  → Plans Home → Approach → Cartesian weld path → Return home   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Real-Time Feedback System

We replaced optimistic, timer-based status messages with a truthful monitoring system:

1. **Process Polling**: The GUI uses a 2-second `QTimer` to poll the OS process state (`_proc.poll()`) for every `NodeWorker`. This provides a "truthful" Running/Offline indicator in the sidebar.
2. **ROS Confirmation Callbacks**:
   - `[SCAN] Depth confirmed`: Subscribed to `/vision/captured_image_depth`. Status only updates when a new depth frame actually hits the DDS buffer.
   - `[PATH] Ready`: Subscribed to `/vision/welding_path`. Status only updates when a valid weld path is successfully held by `path_holder`.
```

---

## 3. New Nodes

### `path_holder.py`

The **single authoritative publisher** for `/vision/welding_path`.

- Subscribes to two staging topics:
  - `/vision/welding_path/generated` — output from `path_generator`
  - `/vision/welding_path/injected` — output from `inject_path_node` (GUI injection)
- Caches both paths independently
- Republishes the active source to `/vision/welding_path` with `TRANSIENT_LOCAL` durability
- Exposes `/path_holder/set_source` service (Trigger) to force-republish the cached path — the GUI calls this before triggering MoveIt execution so late-joining `moveit_controller` receives the path

### `inject_path_node.py`

Allows the GUI to inject a path bypassing the camera pipeline. The GUI publishes a `Path` message to `/vision/inject_path` (VOLATILE), `inject_path_node` re-latches it to `/vision/welding_path/injected` (TRANSIENT_LOCAL) so it survives node restarts.

---

## 4. Changed Nodes

### `capture_images_node.py`

| What changed | Why |
|---|---|
| `captured_image_depth` publisher: `10` → `TRANSIENT_LOCAL RELIABLE` | depth_matcher may start *after* the user pressed Capture. TRANSIENT_LOCAL delivers the last captured depth immediately to any late joiner |
| `captured_camera_info` publisher: `10` → `TRANSIENT_LOCAL RELIABLE` | Same reason as above |
| `rclpy.shutdown()` in `main()` guarded with `if rclpy.ok()` | The signal handler already calls shutdown on SIGINT. The unconditional `finally` call raised `RCLError: rcl_shutdown already called`, causing a noisy exit crash |

### `depth_matcher.py`

| What changed | Why |
|---|---|
| Removed `ApproximateTimeSynchronizer` | Timestamps of captured depth vs. hand-drawn line differ by seconds/minutes — syncer never fired |
| Added cache-based approach | `_on_depth()` and `_on_info()` cache the latest received depth/info; `_on_lines()` immediately calls the processing pipeline using cached data |
| `captured_image_depth` subscriber: `10` → `TRANSIENT_LOCAL RELIABLE` | Must match publisher QoS for late-join delivery |
| `captured_camera_info` subscriber: `10` → `TRANSIENT_LOCAL RELIABLE` | Must match publisher QoS |
| Added `DELETEALL` marker before publishing new markers | Prevents marker accumulation in RViz when the weld line is moved |
| Clears markers when 0 valid 3D lines produced | Stale markers no longer linger after a bad/failed frame |
| Added **0.5 s rate-limit gate** in `_on_lines()` | `path_generator` had a gate but `depth_matcher` had none. Since `depth_matcher` is upstream, it was still firing on every `weld_lines_2d` frame (~camera rate), generating fresh `weld_lines_3d` that bypassed the downstream gate entirely. Adding the gate here is the true fix for the message storm |

### `path_optimizer.py`

| What changed | Why |
|---|---|
| `weld_lines_2d.header.frame_id` always set to `kinect2_rgb_optical_frame` if empty | The annotated image from the GUI canvas has no frame_id. An empty frame_id causes `depth_matcher`'s `lookupTransform` to fail with "Invalid argument empty string" |
| Suppressed Phantom Noise Contours (`< 5` points) | Prevents the generator and matcher from log-spamming and processing microscopic noise contours generated by the thinning/cv2 pipeline. |

### `path_holder.py`

| What changed | Why |
|---|---|
| Content Hash Deduplication | `_publish` now compares the hash of the `(length, first_xyz, last_xyz)` to the previously published path. This prevents spamming the TRANSIENT_LOCAL `/vision/welding_path` topic when a user rapidly clicks "Send" on the exact same path. |

### `path_generator.py`

| What changed | Why |
|---|---|
| Output topic: `/vision/welding_path` → `/vision/welding_path/generated` | Decoupled from direct publication; `path_holder` is now the sole publisher of `/vision/welding_path` |
| Added **0.5 s rate-limit gate** in `callback()` | `manual_line_node` publishes on every GUI frame, causing `weld_lines_3d` to arrive continuously. Without a gate, path_generator → path_holder → moveit_controller all fire 10+ times/second for an unchanged line, flooding logs and wasting CPU |
| `rclpy.shutdown()` guarded with `rclpy.ok()` | Prevents `RCLError: rcl_shutdown already called` on exit |
| **Dynamic Waypoint Capping** | Added `max_waypoints` (default 80); if `num_waypoints > max_waypoints`, path is dynamically downsampled. This directly prevents OMPL from failing due to extreme trajectory complexity on long continuous strokes. |
| **Dynamic ROS Parameters** | Exposed `waypoint_spacing` and `max_waypoints` as fully dynamic parameters that can be tuned live via the `SetParameters` GUI interface without restarting the node. |

### `kinect2_bridge_gpu.yaml`

| What changed | Why |
|---|---|
| `publish_tf` kept `true` (unchanged) | kinect2_bridge must publish its internal chain `kinect2 → kinect2_link → optical frames` |
| Our static TF child changed from `kinect2_link` → `kinect2` | We previously targeted `kinect2_link` directly, competing with kinect2_bridge which also claimed to be its parent. Changed to target `kinect2` (the bridge's root frame) so each frame has exactly one parent publisher |

### `moveit_controller.py`

| What changed | Why |
|---|---|
| `/vision/welding_path` subscriber QoS: `VOLATILE` → `TRANSIENT_LOCAL RELIABLE` | Required to receive the latched path from `path_holder` |
| `approach_distance` default: `0.05` → `0.15` m | Weld seam projected at z ≈ 0.045 m in `base_link`. With 5 cm offset, approach was at z=0.095 m — below workspace_min z=0.10 m. OMPL rejected all goal samples. At 15 cm, approach is at z ≈ 0.195 m, safely inside the reachable zone |
| **Path Offset Injection** | Added `path_offset_x/y/z` parameters. These are applied in `path_callback` by shifting every waypoint pose before storage. Allows for +/- 50mm fine-tuning of the weld bead without re-scanning. |

### `vision_pipeline_gui.py`

| What changed | Why |
|---|---|
| Aggressive Ghost Node Extermination (`pkill -f`) | Before launching any ROS node subprocess, the GUI issues a `pkill -f "name"` to unconditionally murder any zombie/orphaned processes of the same name. This completely eradicates DDS topic conflicts, stale node subscriptions, and multiple instances of the Kinect capture driver. |
| **1-Click Execution Dashboard** | Re-organized the left sidebar to feature a unified simple run panel for running the entire pipeline dynamically without tracking individual micro-node states. |
| **Dynamic Parameters Tab** | Implemented a dedicated settings tab using the ROS2 `SetParameters` service to tune `max_waypoints`, `approach_distance`, `weld_velocity`, and `waypoint_spacing` on-the-fly. |
| **Save/Load Stroke Profiles JSON** | Hand-drawn manual lines can be saved to disk as JSON parameters and reloaded via native OS dialgos. |
| Removed subprocess-based path injection | Replaced with persistent ROS publisher (`inject_path_node`) |
| **Manual Auto-Aligner Node** | Tightly incorporated and visualised the `manual_line_aligner_node` providing bounding box tracking and cross-frame stroke alignments out-of-the-box. |
| **XYZ Path Offset Controls** | Added dedicated spinboxes in the "ROS Parameters" tab and integrated them into the execution flow. Values are in mm for user convenience and auto-converted to meters for ROS. |

---

## 5. Camera TF Fix

**Old values (wrong — child was `kinect2_link` with incorrect position):**
```
x=1.2, y=0.0, z=0.65
roll=-1.5708, pitch=0.0, yaw=1.5708
base_link → kinect2_link   ← WRONG: competed with kinect2_bridge's own kinect2→kinect2_link TF
```

**New values (correct physical measurements, correct frame target):**
```
x=0.646, y=0.1225, z=1.015
roll=-3.14159, pitch=0.0, yaw=1.603684
base_link → kinect2        ← CORRECT: we own only the root connection
```

**Resulting TF tree (no conflicts, each frame has exactly one parent):**
```
world
  └── base_link                   ← demo.launch.py static TF
        └── kinect2                ← OUR static TF (the one physical measurement we own)
              └── kinect2_link         ← kinect2_bridge internal (from calibration)
                    ├── kinect2_rgb_optical_frame
                    ├── kinect2_ir_optical_frame
                    └── (other optical frames)
```

Updated in **both**:
- `parol6_vision/launch/live_pipeline.launch.py` — `static_tf_camera` node
- `parol6_moveit_config/launch/demo.launch.py` — `static_transform_publisher_camera` node

> **Why it matters:** The old TF placed the camera 1.2 m in front of the robot at 0.65 m height. All 3D weld path waypoints were projected into open space well outside the robot's reachable workspace, causing OMPL to fail with "Unable to sample any valid states for goal tree" for every single waypoint. The competing TF publisher also split the tree making RViz unable to trace `L6 → base_link`.

---

## 6. QoS Contract Summary

| Topic | Publisher | Subscriber |
|---|---|---|
| `/vision/captured_image_depth` | TRANSIENT_LOCAL RELIABLE depth=1 | TRANSIENT_LOCAL RELIABLE depth=1 |
| `/vision/captured_camera_info` | TRANSIENT_LOCAL RELIABLE depth=1 | TRANSIENT_LOCAL RELIABLE depth=1 |
| `/vision/weld_lines_2d` | VOLATILE (default 10) | VOLATILE (default 10) |
| `/vision/weld_lines_3d` | VOLATILE (default 10) | VOLATILE (default 10) |
| `/vision/welding_path/generated` | VOLATILE (default 10) | VOLATILE (default 10) |
| `/vision/welding_path/injected` | TRANSIENT_LOCAL RELIABLE depth=1 | TRANSIENT_LOCAL RELIABLE depth=1 |
| `/vision/welding_path` | TRANSIENT_LOCAL RELIABLE depth=1 (**path_holder only**) | TRANSIENT_LOCAL RELIABLE depth=1 |

> ⚠️ **Rule:** Any subscriber that may start *after* the publisher must use matching `TRANSIENT_LOCAL` QoS or it will never receive the message. This is the most common silent failure mode in ROS2 latched topics.

---

## 7. Correct Operating Workflow

```
1. Start MoveIt:  launch_moveit_fake.sh  (or real_hw)
   → publishes base_link TF + correct base_link → kinect2 static TF
   → kinect2_bridge publishes kinect2 → kinect2_link → optical frames
   → full TF chain: world → base_link → kinect2 → kinect2_link → optical frames

2. Start Kinect:  Live Kinect Camera (GUI)
   → kinect2_bridge streams /kinect2/sd/image_depth_rect etc.

3. Start pipeline (GUI "🚀 Launch Full Pipeline"):
   → capture_images, crop_image, path_optimizer,
     depth_matcher, path_generator, path_holder

4. Start MoveIt Controller (GUI Stage 4)

5. Press "S" (Capture):
   → captured_image_depth LATCHED (depth_matcher receives immediately)

6. Draw red line in Manual Red Line mode

7. Press "Apply / Send Strokes":
   → manual_line_node publishes annotated_image
   → path_optimizer detects line → publishes weld_lines_2d
   → depth_matcher uses cached depth → publishes weld_lines_3d
   → path_generator fits spline → publishes to /welding_path/generated
   → path_holder latches → /vision/welding_path
   → moveit_controller receives path automatically

8. Press "📡 Send Path → MoveIt":
   → GUI calls /path_holder/set_source (republish cached path)
   → waits 1 s → calls /moveit_controller/execute_welding_path
   → MoveIt plans: Home → Approach → Weld path → Return home
```

---

## 8. Troubleshooting Quick Reference

| Symptom | Cause | Fix |
|---|---|---|
| `weld_lines_2d received but no captured depth cached yet` | depth_matcher started with VOLATILE QoS or capture not done yet | Ensure QoS matches; capture image first |
| `Could not transform ... to base_link: frame does not exist` | MoveIt (robot_state_publisher) not running | Start `launch_moveit_fake.sh` first |
| `Invalid argument "" passed to lookupTransform source_frame` | `weld_lines_2d` has empty `frame_id` | `path_optimizer` now always sets `kinect2_rgb_optical_frame` |
| Markers accumulate in RViz | Old markers never cleared | `depth_matcher.publish_markers()` now sends `DELETEALL` first |
| `path_holder not available, trying direct execute` | `path_holder` not running | Start Path Holder in Stage 3 before executing |
| `No path received yet` in moveit_controller | QoS mismatch or path_holder not running | Check path_holder is up; check `/vision/welding_path` has 1 publisher |
| `Unable to sample any valid states for goal tree` | Camera TF wrong → path waypoints outside workspace | Verify `demo.launch.py` has updated TF values |
| Double shutdown `RCLError` on node exit | `rclpy.shutdown()` called twice | All nodes now guard with `if rclpy.ok(): rclpy.shutdown()` |
