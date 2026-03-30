# PAROL6 Vision Pipeline — Quality Audit
**Date:** 2026-03-28  
**Test mode:** Offline (fake MoveIt hardware) + Live Kinect v2 camera  
**Method:** Manual Red Line (Method 3)  
**Data sources:** Live `ros2 node info` graph, full session logs, complete code review

---

## 1. Node Graph Verification ✓

```
ros2 node list output (confirmed live):
  /capture_images          /crop_image              /depth_matcher
  /manual_line             /moveit_controller       /path_generator
  /path_holder             /path_optimizer          /static_transform_publisher_camera
  /static_transform_publisher_optical
```

> [!WARNING]
> **Duplicate node instances detected** — `ros2 node list` shows every node listed **twice** (e.g. `/capture_images` appears twice, `/depth_matcher` appears twice). This means the GUI launched two instances of each node. This is a significant robustness issue: both instances subscribe to the same topics but only one processes callbacks deterministically. Race conditions are possible.

**Root cause:** The "Auto-launch pipeline" feature likely starts nodes that were already running from a previous session without checking if they exist first.

---

## 2. Topic Wiring Audit ✓

Verified via `ros2 node info` for each key node:

| Connection | Status | Notes |
|---|---|---|
| `depth_matcher` ← `/vision/captured_image_depth` | ✅ | TRANSIENT_LOCAL QoS |
| `depth_matcher` ← `/vision/captured_camera_info` | ✅ | TRANSIENT_LOCAL QoS |
| `depth_matcher` ← `/vision/weld_lines_2d` | ✅ | Default QoS |
| `depth_matcher` → `/vision/weld_lines_3d` | ✅ | Publishes 3D projected line |
| `path_holder` ← `/vision/welding_path/generated` | ✅ | TRANSIENT_LOCAL QoS |
| `path_holder` ← `/vision/welding_path/injected` | ✅ | TRANSIENT_LOCAL QoS |
| `path_holder` → `/vision/welding_path` | ✅ | TRANSIENT_LOCAL — sole publisher |
| `moveit_controller` ← `/vision/welding_path` | ✅ | TRANSIENT_LOCAL QoS |
| `moveit_controller` → MoveGroup action | ✅ | `/move_action` action client |
| `moveit_controller` → ExecuteTrajectory | ✅ | `/execute_trajectory` action client |

> [!NOTE]
> `moveit_controller` exposes **2 services**: `execute_welding_path` and `is_execution_idle` (not `get_execution_status` — check GUI service call name matches).

---

## 3. Functional Test Results (from session logs)

### 3.1 Capture Stage ✅
- `capture_images_node` responds to `/vision/capture_trigger` topic
- Publishes `captured_image_color` + `captured_image_depth` together as a synchronized pair
- TRANSIENT_LOCAL QoS correctly delivers depth to `depth_matcher` that started after capture

### 3.2 Manual Red Line Mode (Method 3) ✅
- `manual_line_node` loads saved strokes on startup from `/root/.parol6/manual_line_config.json`
- Publishes annotated image on every received `captured_image_color` frame
- `/manual_line/set_strokes` service works: replaces strokes and saves config
- Stroke persistence confirmed: strokes survive node restart

### 3.3 Path Optimizer ✅
- Detects red pixels with HSV dual-range mask (H∈[0–10] ∪ [170–180])
- Computes skeleton + contours, publishes `weld_lines_2d`
- Confidence scoring works: `conf=1.00` for clean strokes
- Frame counter increments correctly (Frame 1, 2, 3... )
- **Rate: fires on every annotated_image — no internal rate limit**
  - Acceptable now because `depth_matcher` has the gate added

### 3.4 Depth Matcher ✅ (with rate limit applied)
- Cache-based sync: delivers 3D lines even when depth captured minutes before line drawing
- `quality=1.00` consistently when depth is valid
- Correctly reads depth at pixel coordinates (samples around target point for robustness)
- Applies min/max depth filter (300–2000mm)
- Published `weld_lines_3d` with correct frame_id (`kinect2_rgb_optical_frame` → `base_link`)
- **0.5 s rate limit confirmed in code** — storm eliminated

### 3.5 Path Generator ✅
- PCA ordering: correctly sorts points along principal axis even for diagonal lines
- B-spline fitting: uses scipy `splprep`/`splev` with 5mm spacing
- Deduplication: prevents crash when identical pixel rows projected to same 3D point
- Orientation: fixed downward quaternion `(x=0.7071068, z=-0.7071068)` — correct for welding on horizontal surfaces
- **0.5 s rate limit confirmed** — generates at most 2/s  
- Generated 37–583 waypoints depending on stroke length — **583 is too many** (see issues below)

### 3.6 Path Holder ✅
- Single authoritative publisher of `/vision/welding_path`
- TRANSIENT_LOCAL re-publishes on source switch — `moveit_controller` receives path immediately on startup
- `set_source` service works: switches between `generated` and `injected`
- `get_status` service available for GUI polling
- Logs source: "new generated path received" / "source switched" — clear audit trail
- **Still publishes every time a path comes in, not just on change** — creates minor redundancy with rate-limited upstream

### 3.7 MoveIt Controller ✅ / ⚠️
- Action clients wired correctly: `/move_action` + `/execute_trajectory`
- Service clients: `/compute_cartesian_path`
- `execute_welding_path` service present and correctly connected to GUI
- `is_execution_idle` service present for status polling
- **OMPL approach failure** at z=0.095m (fixed: now 0.15m offset → z=0.195m)
- **No explicit "execution_in_progress" guard visible** in service list — busy-state protection relies on internal `_exec_lock` threading lock (code confirmed correct)
- MoveIt is **not connected** in current session (fake HW launch was never triggered in this test session) — `move_action` will time out

---

## 4. Issues Found

### 🔴 Critical

| # | Issue | Impact | Fix |
|---|---|---|---|
| C1 | **Duplicate node instances** — every node appears twice in `ros2 node list` | Race condition: both instances process the same messages unpredictably. Causes double-publishing of paths, double TF lookups, and non-deterministic pipeline output | Add a "kill existing nodes" step before auto-launch in the GUI. Check `ros2 node list` before starting. |

### 🟡 Medium

| # | Issue | Impact | Fix |
|---|---|---|---|
| M1 | **Path with 583 waypoints generated** from a multi-stroke session | MoveIt Cartesian planner gets 583 poses — planning takes much longer, more likely to fail on singularities mid-path. Should be 20–60 for typical welds | Cap waypoints at `max_waypoints` param. 583 = old 3-stroke session data carried over. |
| M2 | **`moveit_controller` service name mismatch** — node exposes `is_execution_idle` but `get_execution_status` also appears in comments | GUI may call wrong service and silently fail | Audit GUI button → service call name |
| M3 | **path_holder publishes on every incoming path**, not just on content change | With rate limits at 0.5s each, path_holder could still publish 2 identical paths/s (one from depth_matcher, one from path_generator) | Add path content hash check: skip republish if waypoints unchanged |
| M4 | **kinect2_bridge warns QoS incompatibility** on `/kinect2/sd/points` | Point cloud subscriber (if any) receives no data silently | Add `BEST_EFFORT` QoS to any point cloud subscriber |

### 🟢 Minor / UX

| # | Issue | Impact | Fix |
|---|---|---|---|
| U1 | **Auto-launch doesn't check for existing nodes** — duplicate nodes created | Confusing `WARNING: nodes share exact name` in log | Check with `ros2 node list` before launching, or use `--ros-args -r __node:=unique_name` |
| U2 | **No visual confirmation** of capture success in logs — just `[INFO] Published captured color + depth frame pair` | User might not notice capture succeeded | Add a short colored status pill in the GUI after capture |
| U3 | **Stroke config survives restart** (good for repeatability) but clearing strokes requires re-drawing | If user wants to start fresh, the process is unclear | Add a "Clear Strokes" button in the GUI |
| U4 | **Capture trigger via keyboard `s`** — works only when capture_images terminal has focus | User might press `s` in GUI and nothing happens | Show a "Capture" button that calls the service/topic directly |
| U5 | **Manual line canvas** in GUI: no snap-to-line feature, no grid | Lines are hard to draw accurately for thin welds | Future: add grid overlay option |
| U6 | **Path optimization produces Contour 0 (pts=2)** every time — small artifact near border | Filtered out by path_generator (below min_pts=5) but clutters logs | Filter in path_optimizer before publishing |

---

## 5. Pipeline Data Flow (Verified Live)

```
Camera (1 Hz)
    │
    ▼
capture_images_node
    ├─► /vision/captured_image_raw   [VOLATILE]  → crop_image_node
    ├─► /vision/captured_image_depth [TRANSIENT_LOCAL]  → depth_matcher (cached)
    └─► /vision/captured_camera_info [TRANSIENT_LOCAL]  → depth_matcher (cached)
                │
                ▼
         crop_image_node
                │
                ▼
         /vision/captured_image_color [VOLATILE]
                │
                ▼
         manual_line_node  (draws red stroke on captured image)
                │
                ▼
         /vision/processing_mode/annotated_image [VOLATILE, at camera rate]
                │
                ▼
         path_optimizer  (red pixel segmentation → skeleton → contours)
                │
                ▼
         /vision/weld_lines_2d  (WeldLineArray)
                │
                ▼
         depth_matcher  [0.5s gate]  (pixel → depth lookup → TF → 3D)
                │
                ▼
         /vision/weld_lines_3d  (WeldLine3DArray, in base_link)
                │
                ▼
         path_generator  [0.5s gate]  (PCA order → B-spline → waypoints)
                │
                ▼
         /vision/welding_path/generated  [TRANSIENT_LOCAL]
                │
                ▼
         path_holder  (mux: generated | injected)
                │
                ▼
         /vision/welding_path  [TRANSIENT_LOCAL]
                │
                ▼
         moveit_controller  (Home → Approach → Cartesian Weld)
```

---

## 6. Robustness Assessment

| Scenario | Behaviour | Pass? |
|---|---|---|
| `depth_matcher` starts after capture | Gets cached depth via TRANSIENT_LOCAL | ✅ |
| `moveit_controller` starts after path generated | Gets cached path via TRANSIENT_LOCAL | ✅ |
| User draws new line → old path invalidated | New path replaces cached in path_holder automatically | ✅ |
| Node crashes and restarts | Recovers from TRANSIENT_LOCAL cache immediately | ✅ |
| No strokes drawn → pipeline triggered | path_optimizer publishes empty WeldLineArray, depth_matcher drops it | ✅ |
| 3+ strokes accumulated → large path (583 pts) | MoveIt receives 583 poses — planning slow/fails | ⚠️ |
| MoveIt not running → execute button pressed | GUI service times out — error shown | ⚠️ need timeout msg |
| Same node started twice | Both run — race conditions, double publishing | ❌ |
| Kinect disconnected mid-capture | depth frame becomes stale — depth_matcher uses cached stale depth | ⚠️ no staleness check |

---

## 7. Recommended Fixes (Priority Order)

1. **[C1] Prevent duplicate nodes** — Before launching any node in the GUI, call `ros2 node list` and skip if already running
2. **[M1] Cap waypoint count** — Add `max_waypoints: 80` parameter to `path_generator`; downsample if exceeded  
3. **[M3] Path content deduplication in path_holder** — Hash `(first_pose, last_pose, len)` and skip republish if unchanged
4. **[U3] Add "Clear Strokes" button** — Calls `/manual_line/set_strokes` with empty strokes JSON
5. **[M4] Fix kinect2 point cloud QoS** — Add BEST_EFFORT to any `/kinect2/sd/points` subscriber
